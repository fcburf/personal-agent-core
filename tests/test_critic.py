"""Critic 조건 검사·커뮤니케이터 문구 테스트 (LLM 호출 없음). 실행: python tests/test_critic.py  또는  pytest tests

- 코드 검사(critic_rules.py): 조건별로 '걸려야 할 예시 / 걸리면 안 될 예시'
- 구매 상담은 서경 님 purchase_advisor 실제 출력(가짜 LLM)을 그대로 Critic에 넣어 본다 (오사카·교토 데모 데이터)
- critic_nodes.run_critic: LLM 해석 검사를 끄거나(CRITIC_LLM=off) 가짜 LLM으로 바꿔서 검사
- 그래프: CRITIC_MODE=real 로 실제 Critic을 끼워 구매 상담 경로가 승인 단계까지 가는지
"""
import contextlib
import copy
import io
import os
import sys
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
for p in (str(ROOT), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import critic_rules as cr      # noqa: E402
import critic_nodes as cn      # noqa: E402
from test_purchase_advisor import advise, demo_context, INTENT, CHOICE   # noqa: E402  (서경 님 데모 데이터 재사용)

os.environ["CRITIC_LLM"] = "off"   # 기본은 코드 검사만 (LLM 단계는 아래에서 가짜 LLM으로 따로 검사)
EVENING = datetime(2026, 12, 2, 20, 30)


def rules(vs):
    return {v["rule"].split("_")[0] for v in vs}


# ---------- 여행 일정 정리 (T) ----------
TRIP_CTX = {"trip": {"start_date": "2026-12-01", "end_date": "2026-12-03",
                     "wishlist": {"places": ["후시미 이나리 신사", "도톤보리"]}}}


def trip_plan():
    def day(d, lv, title):
        return {"date": d, "walking_level": lv, "schedule_blocks": [
            {"id": f"{d}-1", "title": title, "start_time": "10:00", "end_time": "12:00", "task_type": "important", "walking_level": lv},
            {"id": f"{d}-2", "title": "점심", "start_time": "12:00", "end_time": "13:00", "task_type": "break", "walking_level": "적음"}]}
    return {"days": [day("2026-12-01", "많음", "후시미 이나리 신사"), day("2026-12-02", "많음", "USJ"),
                     day("2026-12-03", "많음", "도톤보리")],
            "heavy_walking_dates": ["2026-12-01", "2026-12-02", "2026-12-03"], "unresolvable_warnings": []}


def test_trip_plan_passes():
    assert cr.check_trip_plan(trip_plan(), TRIP_CTX) == []


def test_trip_missing_day_is_T1_critical_and_wrong_heavy_dates_is_T4():
    p = trip_plan()
    p["days"].pop()
    p["heavy_walking_dates"] = ["2026-12-01"]
    vs = cr.check_trip_plan(p, TRIP_CTX)
    assert {"T1", "T2", "T4"} <= rules(vs)          # 3일 차가 빠짐 → 도톤보리 누락(T2)도 함께
    assert cr.max_severity(vs) == "critical"


def test_trip_overlap_is_A4():
    p = trip_plan()
    p["days"][0]["schedule_blocks"][1]["start_time"] = "11:30"
    assert "A4" in rules(cr.check_trip_plan(p, TRIP_CTX))


# ---------- 아침 브리핑 (A·B·C·D) ----------
MORNING_CTX = {"today": "2026-12-02", "active_hours": "08:00-23:00",
               "calendar_events": [{"id": "usj", "title": "USJ", "start_time": "09:30", "end_time": "19:00",
                                    "is_fixed": True, "is_adjustable": False, "location": "USJ"}],
               "tasks": []}


def morning_plan(start="09:30"):
    return {"schedule_blocks": [
        {"id": "usj", "title": "USJ", "start_time": start, "end_time": "19:00", "task_type": "fixed", "status": "kept", "location": "USJ"},
        {"id": "break_1", "title": "휴식", "start_time": "19:30", "end_time": "20:00", "task_type": "break", "location": ""}],
        "changed_items": [], "postponed_tasks": []}


def test_morning_plan_passes():
    assert cr.check_morning_briefing(morning_plan(), MORNING_CTX) == []


def test_moving_unadjustable_fixed_event_is_A3_1_critical():
    vs = cr.check_morning_briefing(morning_plan("10:30"), MORNING_CTX)
    assert "A3-1" in rules(vs) and cr.max_severity(vs) == "critical"


def test_postponing_without_rest_request_is_C1_and_urgent_is_A5():
    ctx = dict(MORNING_CTX, tasks=[{"id": "t1", "title": "과제", "deadline": "2026-12-03 23:59", "estimated_minutes": 60}])
    p = morning_plan()
    p["postponed_tasks"] = [{"id": "t1", "title": "과제", "new_scheduled_date": "2026-12-04", "reason": "피곤해서"}]
    assert {"C1", "A5"} <= rules(cr.check_morning_briefing(p, ctx))


# ---------- 구매 상담 (PA·PB·PD) — 서경 님 어드바이저 실제 출력 ----------
def test_demo_purchase_passes_with_only_a_low_note():
    p = advise()                                     # 남은 도보 '많음' 2일 × 하루 6매 = 12매 → 18매입 1개
    vs = cr.check_purchase(p, demo_context(), EVENING)
    assert cr.max_severity(vs) in ("low",) and not rules(vs) - {"PA5"}


def test_smallest_pack_overbuy_is_low_note_not_rework():
    """11/28 결정: 필요량 4매인데 가장 작은 포장이 18매 → 2배를 넘어도 PA5 예외(참고 메모만)"""
    p = advise(choice={**CHOICE, "daily_usage": 2})
    vs = cr.check_purchase(p, demo_context(), EVENING)
    assert [v["rule"] for v in vs] == ["PA5_최소포장_남는양"] and vs[0]["severity"] == "low"


def test_buying_three_boxes_is_PA5_high():
    p = advise()
    p["recommended"]["quantity"] = 3                 # 18매 × 3 = 54매, 필요량 12매
    vs = cr.check_purchase(p, demo_context(), EVENING)
    assert "PA5" in rules(vs) and cr.max_severity(vs) == "high"


def test_too_few_is_PA5_high():
    ctx = demo_context(products=[{"id": "S", "name": "휴족시간 6매입", "price": 400, "currency": "JPY", "unit_count": 6}])
    p = advise(ctx, intent={**INTENT, "relevant_product_ids": ["S"]}, choice={**CHOICE, "chosen_id": "S"})
    p["recommended"]["quantity"] = 1                 # 6매 < 필요량 12매
    assert "PA5" in rules(cr.check_purchase(p, ctx, EVENING))


def test_over_trip_budget_is_PA1_critical_using_converted_total():
    """11/28 결정: 원화 예산 ↔ 엔화 상품은 구매 에이전트의 total_in_trip_currency로 비교"""
    p = advise()
    ctx = demo_context(spending={"records": [], "trip_spent": 795000, "data_quality": "ok"})   # 남은 예산 5,000원
    vs = cr.check_purchase(p, ctx, EVENING)                                                  # 932엔 ≈ 8,854원
    assert "PA1" in rules(vs) and cr.max_severity(vs) == "critical"


def test_missing_converted_total_is_only_a_note():
    p = advise()
    p["recommended"]["total_in_trip_currency"] = None
    assert "PA1_예산비교_불가" in [v["rule"] for v in cr.check_purchase(p, demo_context(), EVENING)]


def test_recommending_despite_impulse_signal_is_PA2():
    p = advise()
    assert "PA2" in rules(cr.check_purchase(p, demo_context(now="23:30"), EVENING))
    urgent = copy.deepcopy(p)
    urgent["intent"]["is_urgent_need"] = True        # 구급약 등 긴급 필요 품목은 PA2 제외
    assert "PA2" not in rules(cr.check_purchase(urgent, demo_context(now="23:30"), EVENING))


def test_advisor_hold_on_impulse_passes_critic():
    ctx = demo_context(now="23:30")
    p = advise(ctx)                                  # 어드바이저가 스스로 보류 + 재검토 시점
    assert p["decision"] == "hold" and cr.check_purchase(p, ctx, EVENING) == []


def test_wrong_heavy_dates_is_PB4_and_unknown_product_is_PB1():
    p = advise()
    p["quantity_detail"]["heavy_walking_dates"] = ["2026-12-02"]
    p["recommended"]["name"] = "휴족시간 100매입"
    assert {"PB4", "PB1"} <= rules(cr.check_purchase(p, demo_context(), EVENING))


def test_hold_without_revisit_is_PD1():
    p = {"decision": "hold", "item": "x", "category": "기타", "recommended": None, "revisit_at": None}
    assert rules(cr.check_purchase(p, {}, EVENING)) == {"PD1"}


# ---------- 저녁 회고 (RC) ----------
def test_recall_too_many_questions_is_RC1_and_bad_stage_is_RC2():
    vs = cr.check_recall({"recall_stage": "2_weeks", "questions": ["a", "b", "c"]}, {})
    assert rules(vs) == {"RC1", "RC2"}
    assert cr.check_recall({"recall_stage": "1_week", "questions": ["a"]}, {}) == []


# ---------- run_critic (코드 + LLM 단계) ----------
class FakeLLM:
    def __init__(self, findings=None, fail=False):
        self.findings, self.fail = findings or [], fail

    def with_structured_output(self, schema):
        outer = self

        class _R:
            def invoke(self, _):
                if outer.fail:
                    raise RuntimeError("API down")
                return schema(findings=outer.findings)
        return _R()


@contextlib.contextmanager
def fake_llm(llm):
    saved, env = sys.modules.get("config"), os.environ.get("CRITIC_LLM")
    m = types.ModuleType("config")
    m.llm_large = m.llm_small = llm
    sys.modules["config"] = m
    os.environ["CRITIC_LLM"] = "on"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        os.environ["CRITIC_LLM"] = env or "off"
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def purchase_state(p=None, ctx=None):
    return {"request_type": "purchase_consultation", "draft_proposal": p or advise(),
            "context_data": ctx or demo_context(), "messages": [], "rejected_history": []}


def test_run_critic_fills_result_fields():
    with contextlib.redirect_stdout(io.StringIO()):
        res = cn.run_critic(purchase_state())
    assert set(res) >= {"violated_rules", "severity", "feedback", "details", "llm_check"}
    assert res["severity"] in ("low", "medium") and res["llm_check"] == "off"


def test_llm_findings_keep_table_severity_and_drop_unknown_ids():
    llm = FakeLLM([{"rule_id": "PD2", "reason": "같은 이유로 거절한 추천", "fix": "다른 상품 제안"},
                   {"rule_id": "Z9", "reason": "목록 밖", "fix": "-"}])
    with fake_llm(llm):
        res = cn.run_critic(purchase_state())
    assert res["llm_check"] == "checked"
    assert "PD2_거절추천_반복" in res["violated_rules"] and not any(r.startswith("Z9") for r in res["violated_rules"])
    assert res["severity"] == "high"                 # 심각도는 LLM이 아니라 조건표 값


def test_llm_failure_falls_back_to_code_checks():
    with fake_llm(FakeLLM(fail=True)):
        res = cn.run_critic(purchase_state())
    assert res["llm_check"].startswith("failed") and res["severity"] in ("low", "medium")


def test_llm_skipped_when_code_already_rejects():
    p = advise()
    p["recommended"]["quantity"] = 3
    with fake_llm(FakeLLM(fail=True)):
        res = cn.run_critic(purchase_state(p))
    assert res["llm_check"].startswith("skipped") and res["severity"] == "high"


# ---------- 커뮤니케이터 문구 ----------
def test_purchase_message_conclusion_reason_alternative():
    s = purchase_state()
    with contextlib.redirect_stdout(io.StringIO()):
        s["critic_result"] = cn.run_critic(s)
    s["critic_verdict"] = "pass"
    msg = cn.build_message(s, after_rejection=False)
    assert msg.startswith("결론: 휴족시간 18매입 1개") and "이유:" in msg and "대안: 샤론파스" in msg
    assert "8,854원" in msg                           # 엔화 상품 + 원화 환산


def test_keep_original_and_rejection_messages_differ_by_scenario():
    base = {"critic_verdict": "keep_original", "critic_result": {"details": [
        {"rule": "PA1_여행예산_초과", "severity": "critical", "message": "[PA1] 남은 예산을 넘습니다."}]}}
    buy = cn.build_message({**base, "request_type": "purchase_consultation"}, False)
    assert "보류" in buy and "남은 예산" in buy and "[PA1]" not in buy
    assert "확정하지" in cn.build_message({**base, "request_type": "trip_planning"}, False)
    assert "원래 일정" in cn.build_message({"request_type": "morning_briefing"}, True)
    assert "넣어 둘게요" in cn.build_message({"request_type": "purchase_consultation"}, True)


def test_warn_message_starts_with_warning_and_trip_lists_days():
    s = {"request_type": "trip_planning", "critic_verdict": "warn", "draft_proposal": trip_plan(),
         "context_data": TRIP_CTX, "critic_result": {"details": [
             {"rule": "T3_입력에없는_장소", "severity": "high", "message": "[T3] 입력에 없는 장소"}]}}
    msg = cn.build_message(s, False)
    assert msg.startswith("⚠️") and "3일 일정" in msg and "2026-12-02" in msg


# ---------- 그래프에서 실제 Critic 사용 ----------
def test_graph_with_real_critic_reaches_approval():
    from helpers import load_graph, run
    g = load_graph(name="g_real_critic")
    os.environ["CRITIC_MODE"] = "real"
    try:
        cfg, n = run(g.app, "real-critic", {"request_type": "purchase_consultation"})
    finally:
        os.environ["CRITIC_MODE"] = "dummy"
    st = g.app.get_state(cfg)
    assert n.count("critic") == 1 and st.next == ("user_approval",)    # 데모 제안은 한 번에 통과
    assert st.values["critic_verdict"] == "pass" and st.values["critic_result"]["target_agent"] is None
    assert st.values["final_response"].startswith("결론:")


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:
            failed += 1; print("FAIL", name, "->", repr(e)[:300])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
