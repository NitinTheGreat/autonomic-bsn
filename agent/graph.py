"""System S1 — the simplest correct pipeline. Exactly one LLM call.

    ingest (py) -> context_build (py) -> har_agent (LLM) -> confidence (py)

S1 deliberately contains NO mitigation. No trust reweighting, no abstention,
no re-querying. Those are Phase 7's S2/S3 and adding them here would make S1
useless as the baseline they are measured against.

Perception vs action
--------------------
`context_build` DOES put the Phase 4 health verdict into the prompt -- the
agent can SEE that a node is degraded. It does not yet ACT on that: nothing
reweights, drops or abstains. Separating perception from action is what lets
Phase 7 add S2/S3 as feature-flagged edges on this same graph rather than as a
rewrite.

Extension points for Phase 7 are declared in `FLAGS` and wired as conditional
edges. Adding trust reweighting means flipping a flag and adding a node, not
restructuring.

Blind by construction
---------------------
Every window entering the graph is wrapped as a BlindWindow before features or
health are computed. If the agent's prompt were built from ground-truth-bearing
data it would inherit exactly the channel the monitor was denied, and every
confidence number downstream would be circular.

Phase 4 measured displacement detection at ~0.46 recall, so S1 assumes nothing
about the monitor catching displacement; the health annotation is context, not
a guarantee.

LangGraph is used when installed; otherwise an equivalent sequential runner
executes the same node functions in the same order. The node functions are the
single source of truth either way.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from agent.llm_client_adapter import LLMClientAdapter  # noqa: E402
from features.extractors import render, window_features  # noqa: E402
from health.diagnose import diagnose_window  # noqa: E402
from health.window_view import blind  # noqa: E402

PROMPT_DIR = os.path.join(_ROOT, "agent", "prompts")
SYSTEM_V1_PATH = os.path.join(PROMPT_DIR, "system_v1.txt")
FEWSHOT_PATH = os.path.join(PROMPT_DIR, "fewshot.yaml")
PROMPT_VERSION = "system_v1"

# Phase 7 extension points. S1 runs with all of these OFF; flipping one adds a
# node/edge rather than changing the ones below.
FLAGS = {
    "trust_reweighting": False,   # S2: weight node evidence by trust_weight
    "abstention": False,          # S3: refuse to answer below a threshold
    "requery": False,             # S3: re-ask with degraded nodes excluded
}


def load_system_prompt() -> str:
    with open(SYSTEM_V1_PATH, encoding="utf-8") as fh:
        return fh.read()


def load_fewshot() -> dict:
    with open(FEWSHOT_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_fewshot_block(doc: dict, eval_subjects: Optional[list] = None) -> str:
    """Render the few-shot examples, asserting subject disjointness."""
    fs_subjects = set(doc.get("fewshot_subjects", []))
    ev = set(eval_subjects or doc.get("eval_subjects", []))
    overlap = fs_subjects & ev
    if overlap:
        raise ValueError(
            "few-shot subjects overlap the evaluation set: %s. A leak here "
            "inflates accuracy invisibly." % sorted(overlap))
    block = ""
    for e in doc.get("examples", []):
        block += "%s\nAnswer: %s\n\n" % (e["features"], e["letter"])
    return ("Here are labelled examples.\n\n" + block) if block else ""


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
def node_ingest(state: dict) -> dict:
    """Normalise the incoming window and attach node metadata. Python only."""
    w = state["window"]
    bw = blind(w)                      # ground truth removed here, once
    state["blind_window"] = bw
    state["node_ids"] = bw.node_ids
    state["node_meta"] = {
        n: {"channels_present": bw.frames[n][0].meta.get("channels_present", []),
            "has_gyro": bw.frames[n][0].gyro_dps is not None}
        for n in bw.node_ids if bw.frames[n]
    }
    return state


def node_context_build(state: dict) -> dict:
    """Features + health annotation -> the prompt. Python only.

    The health verdict is SHOWN to the agent but not acted upon: S1 perceives,
    it does not mitigate.
    """
    bw = state["blind_window"]
    feats = window_features(bw)
    state["features"] = feats

    health = diagnose_window(bw, state.get("gravity_references"))
    state["health"] = health

    lines = [render(feats)]
    ann = []
    for node, v in health["nodes"].items():
        # evidence carries a NAMED rule -- readable as context, auditable in
        # the paper.
        ann.append("%-6s HEALTH %s (%s) -- %s"
                   % (node, v["state"], v["diagnosis"],
                      v["evidence"].get("rule", "")))
    if ann:
        lines.append("")
        lines.append("Node health, inferred by a separate monitor from signal "
                     "statistics:")
        lines.extend(ann)
    state["query_block"] = "\n".join(lines)
    state["health_annotations"] = ann

    doc = state.get("fewshot_doc") or load_fewshot()
    state["fewshot_doc"] = doc
    state["prompt"] = (state.get("system_prompt") or load_system_prompt()).format(
        legend=state["legend"],
        few_shot=build_fewshot_block(doc, state.get("eval_subjects")),
        query=state["query_block"])
    return state


def node_har_agent(state: dict) -> dict:
    """The ONLY model call in S1."""
    adapter: LLMClientAdapter = state["adapter"]
    state["llm"] = adapter.classify(state["prompt"], state["label_set"])
    return state


def node_confidence(state: dict) -> dict:
    """Assemble the confidence readouts. Python only."""
    r = state["llm"]
    primary = state.get("primary_confidence_signal") or "max_p"
    state["result"] = {
        "predicted": r["predicted"],
        "distribution": r["distribution"],
        "max_p": r["max_p"],
        "log_margin": r["log_margin"],
        "entropy": r["entropy"],
        "primary_confidence_signal": primary,
        "primary_confidence_value": r.get(primary),
        "confidence_method": r["confidence_method"],
        "provider": r["provider"],
        "model": r["model"],
        "prompt_version": PROMPT_VERSION,
        "flags": dict(FLAGS),
        "health_states": {n: v["state"]
                          for n, v in state["health"]["nodes"].items()},
        "health_diagnoses": {n: v["diagnosis"]
                             for n, v in state["health"]["nodes"].items()},
        # Present for Phase 7 to consume; S1 does NOT use it.
        "trust_weights": {n: v["trust_weight"]
                          for n, v in state["health"]["nodes"].items()},
    }
    return state


NODES: list[tuple[str, Callable[[dict], dict]]] = [
    ("ingest", node_ingest),
    ("context_build", node_context_build),
    ("har_agent", node_har_agent),
    ("confidence", node_confidence),
]


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_graph():
    """LangGraph StateGraph when available, else a sequential equivalent.

    Both execute the same node functions in the same order, so results do not
    depend on which is installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    g = StateGraph(dict)
    for name, fn in NODES:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    for (a, _), (b, _) in zip(NODES, NODES[1:]):
        g.add_edge(a, b)
    # Phase 7 hangs its S2/S3 nodes off `confidence` behind FLAGS; S1 ends here.
    g.add_edge("confidence", END)
    return g.compile()


class S1Agent:
    """Runs the S1 pipeline over windows."""

    def __init__(self, adapter: Optional[LLMClientAdapter] = None,
                 label_set: Optional[list] = None,
                 legend: Optional[str] = None,
                 gravity_references: Optional[dict] = None,
                 primary_confidence_signal: str = "max_p",
                 eval_subjects: Optional[list] = None):
        self.adapter = adapter or LLMClientAdapter()
        self.label_set = label_set or list("ABCDEFGH")
        self.legend = legend or ""
        self.gravity_references = gravity_references or {}
        self.primary_confidence_signal = primary_confidence_signal
        self.eval_subjects = eval_subjects
        self.system_prompt = load_system_prompt()
        self.fewshot_doc = load_fewshot()
        # Fail now, not mid-run, if the few-shot set leaks the eval subjects.
        build_fewshot_block(self.fewshot_doc, eval_subjects)
        self.compiled = build_graph()
        self.runtime = "langgraph" if self.compiled else "sequential"

    def run(self, window) -> dict:
        state: dict[str, Any] = {
            "window": window,
            "adapter": self.adapter,
            "label_set": self.label_set,
            "legend": self.legend,
            "gravity_references": self.gravity_references,
            "primary_confidence_signal": self.primary_confidence_signal,
            "eval_subjects": self.eval_subjects,
            "system_prompt": self.system_prompt,
            "fewshot_doc": self.fewshot_doc,
        }
        if self.compiled is not None:
            state = self.compiled.invoke(state)
        else:
            for _, fn in NODES:
                state = fn(state)
        return state["result"]
