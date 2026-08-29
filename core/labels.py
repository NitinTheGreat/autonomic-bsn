"""Label sets and dataset-specific ID maps.

There are TWO label sets and the distinction is deliberate, not redundancy:

  PAMAP2_8     the primary experimental set. Includes the two stair classes,
               which are the hardest to separate and therefore the most
               informative about degradation-induced confusion.

  CANONICAL_6  used ONLY for cross-dataset comparison. It is the intersection
               of what both datasets can express.

Why MHEALTH cannot support the 8-class set
------------------------------------------
MHEALTH id 5 is "climbing stairs" with NO ascending/descending distinction, so
PAMAP2's separate ids 12 (ascending) and 13 (descending) have no counterpart.
Mapping both onto id 5, or id 5 onto either one, would silently invent a
distinction the data does not contain.

MHEALTH id 10 "jogging" is deliberately EXCLUDED rather than folded into
running (id 11). MHEALTH labels them as distinct activities; merging them would
change the class definition relative to PAMAP2's "running" and quietly make the
two datasets' running classes non-comparable.

Both facts are encoded below as explicit constants with reasons attached, so a
future reader sees decisions rather than omissions.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# label sets
# --------------------------------------------------------------------------- #
PAMAP2_8 = ["lying", "sitting", "standing", "walking", "running", "cycling",
            "ascending_stairs", "descending_stairs"]        # primary experiments

CANONICAL_6 = ["lying", "sitting", "standing", "walking", "running", "cycling"]
#                                                            cross-dataset only

LABEL_SETS = {"PAMAP2_8": PAMAP2_8, "CANONICAL_6": CANONICAL_6}

# --------------------------------------------------------------------------- #
# dataset id maps
# --------------------------------------------------------------------------- #
PAMAP2_ID_MAP = {1: "lying", 2: "sitting", 3: "standing", 4: "walking",
                 5: "running", 6: "cycling",
                 12: "ascending_stairs", 13: "descending_stairs"}

MHEALTH_ID_MAP = {3: "lying", 2: "sitting", 1: "standing", 4: "walking",
                  11: "running", 9: "cycling"}

ID_MAPS = {"pamap2": PAMAP2_ID_MAP, "mhealth": MHEALTH_ID_MAP}

# Which label sets each dataset may legitimately be asked for.
DATASET_LABEL_SETS = {
    "pamap2": ["PAMAP2_8", "CANONICAL_6"],
    "mhealth": ["CANONICAL_6"],          # NOT PAMAP2_8 -- see module docstring
}

# --------------------------------------------------------------------------- #
# deliberate exclusions -- recorded so they read as decisions, not oversights
# --------------------------------------------------------------------------- #
MHEALTH_EXCLUDED_IDS = {
    5: ("climbing_stairs",
        "MHEALTH id 5 has NO ascending/descending distinction, unlike PAMAP2's "
        "separate ids 12/13. Mapping it to either would invent a distinction "
        "the data does not contain, so it is excluded entirely."),
    10: ("jogging",
         "MHEALTH labels jogging (10) and running (11) as distinct activities. "
         "Folding 10 into running would redefine the class relative to "
         "PAMAP2's running and make the two datasets non-comparable."),
    6: ("waist_bends_forward", "Gym exercise with no PAMAP2 counterpart."),
    7: ("frontal_elevation_arms", "Gym exercise with no PAMAP2 counterpart."),
    8: ("knees_bending_crouching", "Gym exercise with no PAMAP2 counterpart."),
    12: ("jump_front_back", "Gym exercise with no PAMAP2 counterpart."),
}

PAMAP2_EXCLUDED_IDS = {
    7: ("nordic_walking",
        "Distinct gait with poles; no MHEALTH counterpart and not in the "
        "8-class experimental set."),
    16: ("vacuum_cleaning", "Household activity, outside the 8-class set."),
    17: ("ironing", "Household activity, outside the 8-class set."),
    24: ("rope_jumping", "Outside the 8-class set."),
}

EXCLUDED_IDS = {"pamap2": PAMAP2_EXCLUDED_IDS, "mhealth": MHEALTH_EXCLUDED_IDS}

# The transient / null class present in both datasets, always dropped.
NULL_IDS = {"pamap2": 0, "mhealth": 0}

# Full documented activity lists, used to tell "deliberately excluded" apart
# from "undocumented, so our label map is probably wrong".
PAMAP2_ALL_ACTIVITIES = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking", 9: "watching_tv",
    10: "computer_work", 11: "car_driving", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry", 19: "house_cleaning", 20: "playing_soccer",
    24: "rope_jumping",
}

MHEALTH_ALL_ACTIVITIES = {
    0: "null_transient", 1: "standing", 2: "sitting", 3: "lying", 4: "walking",
    5: "climbing_stairs", 6: "waist_bends_forward", 7: "frontal_elevation_arms",
    8: "knees_bending_crouching", 9: "cycling", 10: "jogging", 11: "running",
    12: "jump_front_back",
}

ALL_ACTIVITIES = {"pamap2": PAMAP2_ALL_ACTIVITIES,
                  "mhealth": MHEALTH_ALL_ACTIVITIES}


class LabelSetError(ValueError):
    """Raised when a dataset is asked for a label set it cannot express."""


def permitted_label_sets(dataset: str) -> list[str]:
    """Label set names this dataset may legitimately be asked for."""
    key = dataset.lower()
    if key not in DATASET_LABEL_SETS:
        raise LabelSetError(
            "Unknown dataset %r. Known: %s"
            % (dataset, sorted(DATASET_LABEL_SETS)))
    return list(DATASET_LABEL_SETS[key])


def resolve_label_set(dataset: str, label_set: str) -> list[str]:
    """Return the class list, or RAISE if this dataset cannot express it.

    Failing loudly is the point: silently returning six classes where eight
    were expected would corrupt every downstream accuracy and confusion
    figure without any visible signal.
    """
    key = dataset.lower()
    allowed = permitted_label_sets(key)
    if label_set not in LABEL_SETS:
        raise LabelSetError(
            "Unknown label set %r. Known: %s" % (label_set, sorted(LABEL_SETS)))
    if label_set not in allowed:
        raise LabelSetError(
            "Dataset %r cannot express label set %r (permitted: %s).\n"
            "MHEALTH has no ascending/descending stair distinction -- its id 5 "
            "'climbing stairs' is undirected, so PAMAP2's ids 12/13 have no "
            "counterpart. Refusing to silently return %d classes where %d were "
            "requested.\nUse CANONICAL_6 for cross-dataset work."
            % (dataset, label_set, allowed,
               len(LABEL_SETS[allowed[0]]), len(LABEL_SETS[label_set])))
    return list(LABEL_SETS[label_set])


def id_map_for(dataset: str, label_set: str) -> dict[int, str]:
    """Activity-ID -> canonical label, restricted to the requested label set."""
    classes = set(resolve_label_set(dataset, label_set))
    full = ID_MAPS[dataset.lower()]
    return {i: name for i, name in full.items() if name in classes}
