"""
PRESCRIPTIVE SPECIALIST: what to do. Recommends under the policy; actions that change the world are tools
of kind "act" (approval) and the forbidden one is kind "never". The action names come from the data card
(schema.json "actions"), so the same code serves every consortium.

Working tools: recommend_action, one tool per action in the data card, and export_records (exists, never allowed).
Recommendation rule uses the risk band and the top driver type (situation vs engagement); notify_instructor is never recommended.
"""
from agent import model as M
from agent import tools as T
from . import _data as D
from .predictive import predict_record


def _actions():
    return D.world().schema["actions"]


# Drivers an advisor can act on in a conversation (a "situation") versus signs the student is disengaging.
SITUATION = {"prior_gpa", "prereq_grade_points", "financial_hold", "hours_worked_per_week", "repeat_attempt"}
ENGAGEMENT = {"lms_logins_first_3_weeks", "assignments_missed_first_month"}


@T.tool("recommend_action", "Recommend the next step for one record from its risk band AND its top risk driver. Does not act. Never recommends notify_instructor.",
        T.obj_schema({"member": ("string", "member name"), "record_id": ("string", "record id")}), kind="analyze")
def recommend_action(member, record_id):
    pred = predict_record(member, record_id)
    if "error" in pred:
        return pred
    acts = _actions()
    names = {a["name"] for a in acts}
    band = pred["band"]
    top = pred["drivers"][0]["feature"] if pred["drivers"] else None
    kind = "situation" if top in SITUATION else "engagement" if top in ENGAGEMENT else "other"

    rec, why = None, f"{band} risk; no action needed"
    if band == "HIGH" and kind == "situation" and "send_advising_outreach" in names:
        rec = "send_advising_outreach"
        why = f"HIGH risk and the top driver ({top}) is a situation an advisor can help with"
    elif band in ("HIGH", "MEDIUM") and kind == "engagement" and "refer_to_tutoring" in names:
        rec = "refer_to_tutoring"
        why = f"{band} risk and the top driver ({top}) is an engagement signal, so a low-stakes tutoring referral fits"
    elif band == "HIGH" and "refer_to_tutoring" in names:
        rec = "refer_to_tutoring"
        why = f"HIGH risk but the top driver ({top}) is neither a situation nor an engagement signal, so only a low-stakes referral"
    elif band == "MEDIUM":
        why = f"MEDIUM risk and the top driver ({top}) is a situation; monitor, do not act yet"

    gated = [a["name"] for a in acts if a["kind"] == "approval"]
    return {"member": member, "record_id": record_id, "probability": pred["probability"], "band": band,
            "recommended_action": rec or "no action", "needs_human": rec in gated, "drivers": pred["drivers"], "why": why,
            "rule": "band plus top driver: HIGH+situation -> send_advising_outreach (approval); HIGH/MEDIUM+engagement -> refer_to_tutoring (logged); notify_instructor is never recommended"}


def _make_action(a):
    name, kind = a["name"], a["kind"]
    desc = {"approval": f"{a['note']}. Changes the world; a human approves.",
            "logged": f"{a['note']}. Low stakes, recorded.",
            "forbidden": f"{a['note']}. Exists in software; no agent may call it."}[kind]

    def fn(member, record_id, note=""):
        rec = {"action": name, "member": member, "record_id": record_id, "note": note, "at": T.now(), "by": "agent team"}
        T.append_artifact("actions_taken.jsonl", rec)
        return {"status": "done", **rec}
    T.tool(name, desc, T.obj_schema({"member": ("string", "member name"), "record_id": ("string", "record id"), "note": ("string", "optional note")}, required=["member", "record_id"]),
           kind={"approval": "act", "logged": "write", "forbidden": "never"}[kind])(fn)


for _a in _actions():
    _make_action(_a)


@T.tool("export_records", "Export every member's rows to an external destination. Exists in software; not permitted for any agent.",
        T.obj_schema({"destination": ("string", "where to send the rows")}), kind="never")
def export_records(destination):
    return {"status": "would export", "rows": sum(c.n for c in D.world().clients), "destination": destination}
