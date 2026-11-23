import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import specialists  # noqa: F401
import config
from agent import tools as T

MIN_TOOLS = 8


def test_enough_tools_registered():
    assert len(T.TOOLS) >= MIN_TOOLS


def test_every_tool_has_a_kind_and_schema():
    for name, t in T.TOOLS.items():
        assert t["kind"] in {"read", "analyze", "model", "write", "act", "never"}, name
        assert t["parameters"]["type"] == "object", name


def test_summarize_home_member_runs():
    r = T.call("summarize_member", {"member": config.HOME_MEMBER})
    assert r["rows"] > 0 and 0 <= r["outcome_rate"] <= 1


def test_model_card_facts_available():
    r = T.call("model_card_facts", {})
    assert "neutral_test_accuracy" in r and r["gate_decision"] == "PASS"


def test_compare_month_over_month_runs():
    r = T.call("compare_month_over_month", {"member": config.HOME_MEMBER, "month": 6})
    assert r["change"]["direction"] in {"increased", "decreased", "unchanged"}


def test_segment_rates_all_members_covers_every_member():
    r = T.call("segment_rates_all_members", {"segment": "modality"})
    assert len(r["members"]) == 4


def test_segment_driver_names_a_segment_value():
    r = T.call("segment_driver", {"member": config.HOME_MEMBER, "segment": "course_level"})
    assert r["carries_the_gap"]["value"] in {x["value"] for x in r["by_segment"]["course_level"]}


def test_driver_signs_finds_the_real_estate_flip():
    r = T.call("driver_signs_all_members", {"feature": "hours_worked_per_week"})
    assert [f["member"] for f in r["sign_flips"]] == ["Real Estate"]


def test_calibration_check_returns_rates():
    r = T.call("calibration_check", {"member": "Real Estate"})
    assert 0 <= r["predicted_rate"] <= 1 and 0 <= r["actual_rate"] <= 1


def test_recommend_action_never_recommends_notify_instructor():
    ranked = T.call("rank_records_by_risk", {"member": config.HOME_MEMBER, "top_k": 10})
    for rec in ranked["records"]:
        r = T.call("recommend_action", {"member": config.HOME_MEMBER, "record_id": rec["record_id"]})
        assert r["recommended_action"] != "notify_instructor" and "why" in r
