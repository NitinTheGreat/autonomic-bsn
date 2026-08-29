"""Scoring — the ONLY module that reads ground truth.

It sees the monitor's output and the true labels. The monitor never sees this
module, and this module never feeds anything back into the monitor. That
one-way arrangement is what keeps the detection metrics non-circular.

Scoring uses `meta["realised"]`, never `meta["requested"]`. For displacement the
two differ by up to 6x (Phase 3 section 6.1): a requested 15 degrees shows up as
14.89 deg on a lying ankle and 2.33 deg on a standing chest. Scoring against the
requested value would credit the monitor for missing something that was never
observable, or penalise it for the same.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from health.diagnose import DIAGNOSES

# Which realised metric expresses "how much actually happened".
REALISED_KEY = {
    "dropout": "frac_nan",
    "packet_loss": "loss_rate",
    "rate_degradation": "realised_hz",
    "clock_desync": "delta_ms",
    "displacement": "realised_angle_deg",
}


class DetectionTally:
    """Accumulates detection and diagnosis outcomes across windows."""

    def __init__(self):
        self.rows: list[dict] = []

    # ------------------------------------------------------------------ add --
    def add(self, monitor_out: dict, truth_window, dataset: str) -> None:
        """Record one window. `truth_window` is the ORIGINAL Window."""
        meta = truth_window.meta or {}
        true_type = meta.get("failure_type")          # None on clean windows
        target = meta.get("target_node")
        realised = meta.get("realised", {}) or {}
        sev = meta.get("severity")

        for node, verdict in monitor_out["nodes"].items():
            is_target = (node == target)
            flagged = verdict["state"] != "HEALTHY"
            expected_dx = true_type if is_target else "healthy"

            self.rows.append({
                "dataset": dataset,
                "activity": truth_window.label,
                "node": node,
                "is_target": is_target,
                "true_failure": true_type,
                "severity": sev,
                "expected_diagnosis": expected_dx or "healthy",
                "predicted_diagnosis": verdict["diagnosis"],
                "flagged": flagged,
                "state": verdict["state"],
                "min_sub_score": verdict["min_sub_score"],
                "realised": realised,
                "realised_magnitude": realised.get(
                    REALISED_KEY.get(true_type or "", ""), None),
                "clock_source": realised.get("clock_source"),
                "alignment": realised.get("gravity_axis_alignment"),
                "window_start": truth_window.start_sec,
            })

    # -------------------------------------------------------------- metrics --
    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> dict:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}

    def detection_by_failure(self) -> dict:
        """Precision/recall/F1 for 'flagged the right node as non-healthy'."""
        out = {}
        types = sorted({r["true_failure"] for r in self.rows
                        if r["true_failure"]})
        for ft in types:
            for sev in sorted({r["severity"] for r in self.rows
                               if r["true_failure"] == ft}):
                sub = [r for r in self.rows
                       if r["true_failure"] == ft and r["severity"] == sev]
                tp = sum(1 for r in sub if r["is_target"] and r["flagged"])
                fn = sum(1 for r in sub if r["is_target"] and not r["flagged"])
                fp = sum(1 for r in sub if not r["is_target"] and r["flagged"])
                out.setdefault(ft, {})["sev%d" % sev] = self._prf(tp, fp, fn)
        # pooled across severities
        for ft in types:
            sub = [r for r in self.rows if r["true_failure"] == ft]
            tp = sum(1 for r in sub if r["is_target"] and r["flagged"])
            fn = sum(1 for r in sub if r["is_target"] and not r["flagged"])
            fp = sum(1 for r in sub if not r["is_target"] and r["flagged"])
            out[ft]["all"] = self._prf(tp, fp, fn)
        return out

    def diagnosis_confusion(self) -> dict:
        """6x6 confusion over diagnosis classes, target nodes only.

        Distinct from detection and strictly more demanding: flagging the right
        node as unhealthy is necessary but not sufficient -- the monitor also
        has to name the right failure.
        """
        idx = {d: i for i, d in enumerate(DIAGNOSES)}
        m = [[0] * len(DIAGNOSES) for _ in DIAGNOSES]
        for r in self.rows:
            if not r["is_target"] and r["true_failure"]:
                continue          # untouched nodes scored under FPR instead
            e, p = r["expected_diagnosis"], r["predicted_diagnosis"]
            if e in idx and p in idx:
                m[idx[e]][idx[p]] += 1
        correct = sum(m[i][i] for i in range(len(DIAGNOSES)))
        total = sum(sum(row) for row in m)
        per_class = {}
        for i, d in enumerate(DIAGNOSES):
            n = sum(m[i])
            per_class[d] = {"n": n, "accuracy": (m[i][i] / n) if n else None,
                            "confused_with": {
                                DIAGNOSES[j]: m[i][j]
                                for j in range(len(DIAGNOSES))
                                if j != i and m[i][j]}}
        return {"labels": list(DIAGNOSES), "matrix": m,
                "overall_accuracy": correct / total if total else 0.0,
                "n": total, "per_class": per_class}

    def false_positive_rate(self) -> dict:
        """FPR on CLEAN windows -- the first number a reviewer asks for."""
        clean = [r for r in self.rows if not r["true_failure"]]
        n = len(clean)
        fp = sum(1 for r in clean if r["flagged"])
        by_node: dict = defaultdict(lambda: {"n": 0, "fp": 0})
        by_dx: dict = defaultdict(int)
        for r in clean:
            by_node[r["node"]]["n"] += 1
            if r["flagged"]:
                by_node[r["node"]]["fp"] += 1
                by_dx[r["predicted_diagnosis"]] += 1
        return {
            "n_clean_node_windows": n,
            "false_positives": fp,
            "false_positive_rate": fp / n if n else 0.0,
            "by_node": {k: {**v, "rate": v["fp"] / v["n"] if v["n"] else 0.0}
                        for k, v in by_node.items()},
            "misdiagnosed_as": dict(by_dx),
        }

    def displacement_by_node_activity(self) -> dict:
        """Displacement recall split by (node, activity) -- see Phase 3 6.1."""
        out: dict = {}
        for r in self.rows:
            if r["true_failure"] != "displacement" or not r["is_target"]:
                continue
            key = "%s/%s" % (r["node"], r["activity"])
            e = out.setdefault(key, {"n": 0, "detected": 0, "diagnosed": 0,
                                     "realised_deg": [], "alignment": []})
            e["n"] += 1
            e["detected"] += int(r["flagged"])
            e["diagnosed"] += int(r["predicted_diagnosis"] == "displacement")
            if r["realised_magnitude"] is not None:
                e["realised_deg"].append(r["realised_magnitude"])
            if r["alignment"] is not None:
                e["alignment"].append(r["alignment"])
        for k, e in out.items():
            e["recall"] = e["detected"] / e["n"] if e["n"] else 0.0
            e["diagnosis_accuracy"] = e["diagnosed"] / e["n"] if e["n"] else 0.0
            e["mean_realised_deg"] = (sum(e["realised_deg"]) / len(e["realised_deg"])
                                      if e["realised_deg"] else None)
            e["mean_alignment"] = (sum(e["alignment"]) / len(e["alignment"])
                                   if e["alignment"] else None)
            e["undetectable"] = e["recall"] < 0.5
            del e["realised_deg"], e["alignment"]
        return out

    def clock_desync_by_source(self) -> dict:
        """Split by measured vs derived clock -- the two are not comparable."""
        out: dict = {}
        for r in self.rows:
            if r["true_failure"] != "clock_desync" or not r["is_target"]:
                continue
            src = r["clock_source"] or "unknown"
            e = out.setdefault(src, {"n": 0, "detected": 0, "diagnosed": 0})
            e["n"] += 1
            e["detected"] += int(r["flagged"])
            e["diagnosed"] += int(r["predicted_diagnosis"] == "clock_desync")
        for e in out.values():
            e["recall"] = e["detected"] / e["n"] if e["n"] else 0.0
            e["diagnosis_accuracy"] = e["diagnosed"] / e["n"] if e["n"] else 0.0
        out["_note"] = ("MHEALTH's timestamps are DERIVED from the row index at "
                        "50 Hz, so its desync results are not comparable to "
                        "PAMAP2's measured clock.")
        return out

    def summary(self) -> dict:
        return {
            "n_rows": len(self.rows),
            "detection_by_failure": self.detection_by_failure(),
            "diagnosis_confusion": self.diagnosis_confusion(),
            "false_positives": self.false_positive_rate(),
            "displacement_by_node_activity": self.displacement_by_node_activity(),
            "clock_desync_by_clock_source": self.clock_desync_by_source(),
            "scoring_note": ("Scored against meta['realised'], never "
                             "'requested'."),
        }


def detection_latency(monitor_outputs: list[dict], onset_sec: Optional[float],
                      target_node: str) -> Optional[dict]:
    """Frames/seconds from injection onset to the node leaving HEALTHY.

    Within a single 2.56 s window the monitor emits one verdict, so latency is
    reported at window granularity: the offset from onset to the end of the
    first window in which the node is no longer HEALTHY.
    """
    if onset_sec is None:
        return None
    for out in monitor_outputs:
        v = out["nodes"].get(target_node)
        if v and v["state"] != "HEALTHY" and out["end_sec"] >= onset_sec:
            return {"onset_sec": onset_sec,
                    "detected_by_sec": out["end_sec"],
                    "latency_sec": max(0.0, out["end_sec"] - onset_sec),
                    "granularity": "one verdict per window"}
    return {"onset_sec": onset_sec, "detected_by_sec": None,
            "latency_sec": None, "granularity": "one verdict per window"}
