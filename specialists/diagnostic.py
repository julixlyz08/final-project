"""
DIAGNOSTIC SPECIALIST: why. Compares members and ranks drivers, still with aggregates only.

Template tools: compare_members, rank_drivers.
Team tools (Step 4, Salam): segment_driver (which segment carries a member's extra failures),
driver_signs_all_members (where a driver flips sign across members).
Every number is computed inside one member's rows; results hold counts, rates and correlations only.
"""
import numpy as np

from agent import tools as T
from . import _data as D


def _correlations(c):
    """Correlation of every feature with the outcome, inside one member only."""
    y = c.y_train - c.y_train.mean()
    X = c.X_train - c.X_train.mean(axis=0)
    denom = (np.sqrt((X ** 2).sum(axis=0)) * np.sqrt((y ** 2).sum())) + 1e-9
    return (X * y[:, None]).sum(axis=0) / denom


@T.tool("compare_members", "Compare two members: outcome rates and the features whose means differ most (in sd units).",
        T.obj_schema({"member_a": ("string", "first member"), "member_b": ("string", "second member")}), kind="analyze")
def compare_members(member_a, member_b):
    a, b = D.member(member_a), D.member(member_b)
    feats = D.features()
    diff = a.X_train.mean(axis=0) - b.X_train.mean(axis=0)      # standardized, so already in sd units
    order = np.argsort(-np.abs(diff))[:5]
    return {"member_a": {"name": member_a, "rows": a.n, "outcome_rate": round(float(a.y_train.mean()), 3)},
            "member_b": {"name": member_b, "rows": b.n, "outcome_rate": round(float(b.y_train.mean()), 3)},
            "largest_feature_differences_sd": [{"feature": feats[i], "a_minus_b_sd": round(float(diff[i]), 2)} for i in order],
            "reading": "A rate gap with small feature differences points to a base-rate (label) difference; large feature differences point to a different population."}


@T.tool("rank_drivers", "Within one member, which features move with the outcome (correlation with the label, sign and strength).",
        T.obj_schema({"member": ("string", "member name"), "top_k": ("integer", "how many features to return, default 5")}, required=["member"]), kind="analyze")
def rank_drivers(member, top_k=5):
    c = D.member(member)
    feats = D.features()
    corr = _correlations(c)
    order = np.argsort(-np.abs(corr))[:int(top_k)]
    return {"member": member, "rows": c.n,
            "drivers": [{"feature": feats[i], "correlation_with_outcome": round(float(corr[i]), 3)} for i in order],
            "reading": "Compare the sign of a driver across members. A feature that is positive at every member but negative at one is a concept difference: the charter must decide what to do for that member."}


@T.tool("segment_driver",
        "Within one member, which course level or modality carries that member's extra failures: for each segment value, "
        "its rows, its failure rate, the member's overall rate, and the excess failures above what the overall rate predicts. "
        "Use for 'is it the students or the gateway course?' questions. Aggregates only.",
        T.obj_schema({"member": ("string", "member name"),
                      "segment": ("string", "optional: course_level or modality; if omitted, both are checked")}, required=["member"]),
        kind="analyze")
def segment_driver(member, segment=None):
    c = D.member(member)
    lab = D.label()
    segs = D.world().segments
    if segment is not None and segment not in segs:
        return {"error": f"{segment} is not a segment column; choose from {segs}"}
    overall = float(c.y_train.mean())
    by_segment = {}
    best = None                                     # (excess, segment, value)
    for seg in ([segment] if segment else segs):
        rows, fails = {}, {}
        for r in c.rows:
            v = r[seg]
            rows[v] = rows.get(v, 0) + 1
            fails[v] = fails.get(v, 0) + int(r[lab])
        values = []
        for v in rows:
            excess = fails[v] - rows[v] * overall       # failures above what the member's own rate predicts
            values.append({"value": v, "rows": rows[v], "outcome_rate": round(fails[v] / rows[v], 3),
                           "excess_failures": round(excess, 1)})
            if best is None or excess > best[0]:
                best = (excess, seg, v, round(fails[v] / rows[v], 3))
        by_segment[seg] = sorted(values, key=lambda x: -x["outcome_rate"])
    excess, seg, val, rate = best
    return {"member": member, "rows": c.n, "member_outcome_rate": round(overall, 3), "by_segment": by_segment,
            "carries_the_gap": {"segment": seg, "value": val, "outcome_rate": rate, "member_outcome_rate": round(overall, 3),
                                "excess_failures": round(excess, 1)},
            "reading": f"{val} ({seg}) fails at {rate} against {round(overall, 3)} for {member} overall. If one segment carries "
                       "most of the excess, the cause is the course or delivery (for lower_division, the gateway course), not the whole "
                       "student population. Differences under about 0.02 on a few hundred rows can be noise."}


@T.tool("driver_signs_all_members",
        "For every member, the sign and strength of each feature's correlation with the outcome, computed inside each member "
        "separately, and the features whose sign at one member is opposite to the others (a concept difference the charter "
        "must handle). Optionally limit to one feature, for example hours_worked_per_week. Aggregates only.",
        T.obj_schema({"feature": ("string", "optional: one feature name; if omitted, all features are checked")}, required=[]),
        kind="analyze")
def driver_signs_all_members(feature=None):
    feats = D.features()
    if feature is not None and feature not in feats:
        return {"error": f"{feature} is not a feature; choose from {feats}"}
    members = [c.name for c in D.world().clients]
    corr = {c.name: _correlations(c) for c in D.world().clients}     # one member at a time
    check = [feature] if feature else feats
    table, flips = {}, []
    for f in check:
        i = feats.index(f)
        vals = {m: round(float(corr[m][i]), 3) for m in members}
        table[f] = vals
        pos = [m for m in members if vals[m] > 0]
        neg = [m for m in members if vals[m] < 0]
        odd = neg if len(neg) < len(pos) else pos if len(pos) < len(neg) else []
        if len(odd) == 1:                               # one member against all the others
            m = odd[0]
            others = [abs(vals[o]) for o in members if o != m]
            flips.append({"feature": f, "member": m, "correlation_there": vals[m],
                          "others_sign": "positive" if m in neg else "negative",
                          "strength": round(min(abs(vals[m]), min(others)), 3)})
    flips.sort(key=lambda x: -x["strength"])            # strongest flip first; tiny ones are likely noise
    return {"members": members, "rows": {c.name: c.n for c in D.world().clients},
            "correlation_by_member": table, "sign_flips": flips,
            "reading": "A feature that moves the same way everywhere except at one member is a concept difference: one shared model "
                       "learns the majority direction and can be wrong for that member. The charter must decide (for example a "
                       "fine-tuned model or a tighter floor for that member). Flips with |correlation| under about 0.05 can be noise."}