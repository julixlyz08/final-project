"""
DESCRIPTIVE SPECIALIST: what happened. Aggregates only; a member's rows never leave its file.

Tools: summarize_member, segment_breakdown, trend_by_month, segment_rates_all_members,
compare_month_over_month.
"""
from collections import Counter, defaultdict

from agent import tools as T
from . import _data as D


@T.tool("summarize_member", "Row count, outcome rate and segment mix for one member. Aggregate only.",
        T.obj_schema({"member": ("string", "member name exactly as in the data card")}), kind="read")
def summarize_member(member):
    c = D.member(member)
    lab = D.label()
    out = {"member": member, "rows": c.n, "outcome_rate": round(float(c.y_train.mean()), 3), "segments": {}}
    for seg in D.world().segments:
        cnt = Counter(r[seg] for r in c.rows)
        out["segments"][seg] = {k: v for k, v in cnt.most_common()}
    return out


@T.tool("segment_breakdown", "Outcome rate by one segment column for one member (for example by ward or by claim_type).",
        T.obj_schema({"member": ("string", "member name"), "segment": ("string", "a segment column from the data card")}), kind="read")
def segment_breakdown(member, segment):
    c = D.member(member)
    if segment not in D.world().segments:
        return {"error": f"{segment} is not a segment column; choose from {D.world().segments}"}
    lab = D.label()
    tot, pos = Counter(), Counter()
    for r in c.rows:
        tot[r[segment]] += 1
        pos[r[segment]] += int(r[lab])
    return {"member": member, "segment": segment,
            "rates": {k: {"rows": tot[k], "outcome_rate": round(pos[k] / tot[k], 3)} for k in sorted(tot, key=lambda k: -pos[k] / tot[k])}}


@T.tool("trend_by_month", "Outcome rate by month (1 to 12) for one member.",
        T.obj_schema({"member": ("string", "member name")}), kind="read")
def trend_by_month(member):
    c = D.member(member)
    lab = D.label()
    tot, pos = defaultdict(int), defaultdict(int)
    for r in c.rows:
        m = int(r["month"])
        tot[m] += 1
        pos[m] += int(r[lab])
    rates = {m: round(pos[m] / tot[m], 3) for m in sorted(tot)}
    hi = max(rates, key=rates.get)
    lo = min(rates, key=rates.get)
    return {"member": member, "rate_by_month": rates, "highest_month": hi, "lowest_month": lo}


@T.tool(
    "segment_rates_all_members",
    "Failure rates by one segment column for every department. Each department's rates are "
    "computed only from its own rows; departments are never pooled. Call once for each segment, "
    "such as course_level and modality. Aggregate counts and rates only.",
    T.obj_schema({"segment": ("string", "a segment column, such as course_level or modality")}),
    kind="analyze",
)
def segment_rates_all_members(segment):
    world = D.world()
    if segment not in world.segments:
        return {"error": f"{segment} is not a segment column; choose from {world.segments}"}

    lab = D.label()
    members = []
    for client in world.clients:
        c = D.member(client.name)
        total, failures = Counter(), Counter()
        for row in c.rows:
            value = row[segment]
            total[value] += 1
            failures[value] += int(row[lab])
        members.append({
            "member": client.name,
            "rows": c.n,
            "failure_rate": round(float(c.y_train.mean()), 3),
            "rates": {
                value: {
                    "rows": total[value],
                    "failures": failures[value],
                    "failure_rate": round(failures[value] / total[value], 3),
                }
                for value in sorted(total)
            },
        })
    return {"segment": segment, "members": members}


@T.tool(
    "compare_month_over_month",
    "Compare one member's failure rate in a specified month with the immediately preceding "
    "month. Returns aggregate enrollment and failure counts only.",
    T.obj_schema({
        "member": ("string", "member name exactly as in the data card"),
        "month": ("integer", "month to compare, from 2 through 12"),
    }),
    kind="analyze",
)
def compare_month_over_month(member, month):
    month = int(month)
    if month < 2 or month > 12:
        return {"error": "month must be from 2 through 12 so a preceding month is available"}

    c = D.member(member)
    lab = D.label()
    month_counts = defaultdict(lambda: {"rows": 0, "failures": 0})
    for row in c.rows:
        row_month = int(row["month"])
        if row_month in (month - 1, month):
            month_counts[row_month]["rows"] += 1
            month_counts[row_month]["failures"] += int(row[lab])

    previous, current = month_counts[month - 1], month_counts[month]
    if not previous["rows"] or not current["rows"]:
        missing = [m for m, result in ((month - 1, previous), (month, current)) if not result["rows"]]
        return {"error": f"no enrollment data for month(s) {missing} at {member}"}

    previous_rate = previous["failures"] / previous["rows"]
    current_rate = current["failures"] / current["rows"]
    difference = current_rate - previous_rate
    return {
        "member": member,
        "previous_month": {
            "month": month - 1,
            **previous,
            "failure_rate": round(previous_rate, 3),
        },
        "month": {
            "month": month,
            **current,
            "failure_rate": round(current_rate, 3),
        },
        "change": {
            "failure_rate": round(difference, 3),
            "percentage_points": round(difference * 100, 1),
            "direction": "increased" if difference > 0 else "decreased" if difference < 0 else "unchanged",
        },
    }
