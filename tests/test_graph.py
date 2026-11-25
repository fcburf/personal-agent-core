"""그래프 경로·판정 테스트 (LLM 호출 없음). 실행: python tests/test_graph.py  또는  pytest tests"""
from helpers import load_graph, run, quiet
from state import get_last_user_message

g = load_graph()
app = g.app


# ---------- Critic 판정 ----------
def test_verdict_rules():
    d = g.decide_critic_verdict
    assert d("medium", 0) == (True, "pass")
    assert d("high", 1) == (False, "rework") and d("high", 2) == (False, "rework")
    assert d("high", 3) == (False, "warn")
    assert d("critical", 3) == (False, "keep_original")


def test_critical_after_two_reworks_keeps_original_without_approval():
    gc = load_graph("critical", "g_critical")
    cfg, n = run(gc.app, "crit", {"request_type": "morning_briefing"})
    st = gc.app.get_state(cfg)
    assert n.count("planner") == 3 and n.count("critic") == 3      # 재작업 2회 = 플래너 3번
    assert st.values["critic_verdict"] == "keep_original"
    assert st.next == ()                                             # 승인 없이 종료


def test_high_after_two_reworks_warns_and_asks_approval():
    gh = load_graph("high", "g_high")
    cfg, _ = run(gh.app, "high", {"request_type": "morning_briefing"})
    st = gh.app.get_state(cfg)
    assert st.values["critic_verdict"] == "warn"
    assert st.next == ("user_approval",)
    assert st.values["final_response"].startswith("⚠️")


# ---------- 시나리오 경로 ----------
def test_morning_briefing_approve_path():
    cfg, n = run(app, "morning", {"request_type": "morning_briefing"}, [{"status": "approved"}])
    assert n[:4] == ["supervisor", "context_collector", "condition_analyzer", "planner"]
    assert n.count("planner") == 2                                   # 임시 Critic: 1회 반려 후 통과
    assert n[-1] == "action_executor" and app.get_state(cfg).next == ()


def test_trip_planning_uses_planner_and_writes_on_approval():
    cfg, n = run(app, "trip", {"request_type": "trip_planning"}, [{"status": "approved"}])
    assert "planner" in n and "purchase_advisor" not in n
    assert n[-1] == "action_executor"
    assert "days" in app.get_state(cfg).values["draft_proposal"]


def test_purchase_and_review_rework_stay_with_owner_agent():
    for req, owner in [("purchase_consultation", "purchase_advisor"), ("retrospective", "recall_interview")]:
        cfg, n = run(app, f"rw-{req}", {"request_type": req})
        assert n.count(owner) == 2 and "planner" not in n, n
        st = dict(app.get_state(cfg).values)
        st.update(critic_verdict="rework", critic_result={"target_agent": "planner"})
        assert quiet(g.route_after_critic, st) == owner              # 잘못된 target_agent 무시


def test_general_ends_without_approval():
    cfg, n = run(app, "general", {"request_type": "general"})
    assert n == ["supervisor", "communicator"] and app.get_state(cfg).next == ()


# ---------- 사용자 결정 ----------
def test_modify_records_history_resets_counter_and_adds_message():
    cfg, n = run(app, "modify", {"request_type": "morning_briefing"},
                 [{"status": "modify", "feedback": "3시 일정은 내일로 미뤄줘"}])
    st = app.get_state(cfg).values
    assert n[n.index("user_approval") + 1] == "planner"
    assert len(st["rejected_history"]) == 1
    assert set(st["rejected_history"][0]["changes"]) == {"changed_items", "postponed_tasks"}
    assert get_last_user_message(st) == "3시 일정은 내일로 미뤄줘"
    assert n.count("planner") == 4                                   # 처음 2번 + 수정 후 2번 (카운터 초기화)


def test_reject_ends_with_keep_original_message():
    cfg, n = run(app, "reject", {"request_type": "morning_briefing"}, [{"status": "rejected"}])
    st = app.get_state(cfg)
    assert n[-2:] == ["user_approval", "communicator"] and st.next == ()
    assert "원래 일정" in st.values["final_response"]


def test_new_request_resets_previous_decision():
    run(app, "reset", {"request_type": "morning_briefing"}, [{"status": "rejected"}])
    cfg, _ = run(app, "reset", {"request_type": "morning_briefing"})
    st = app.get_state(cfg)
    assert st.values["user_decision"] == {} and st.next == ("user_approval",)


def test_review_modify_goes_to_recall_interview():
    cfg, n = run(app, "review-mod", {"request_type": "retrospective"}, [{"status": "modify"}])
    assert n[n.index("user_approval") + 1] == "recall_interview"


# ---------- 거절 이력 형식 ----------
def test_purchase_rejection_history_format():
    cfg, _ = run(app, "buy-reject", {"request_type": "purchase_consultation"},
                 [{"status": "rejected", "feedback": "너무 비싸", "reason_tags": ["가격"]}])
    h = app.get_state(cfg).values["rejected_history"][-1]
    assert set(h["changes"]) == {"decision", "item", "category", "recommended"}
    assert h["reject_reason_tags"] == ["가격"]
    assert h["purchase_context"]                                    # 구매하려던 이유·상황


def test_trip_plan_history_keeps_block_ids():
    changes = g.extract_changes({"days": [{"date": "2026-11-30", "schedule_blocks": [
        {"id": "evt_1", "title": "오사카성", "start_time": "13:00", "end_time": "16:00", "location": "x"}]}]})
    assert changes == {"days": [{"date": "2026-11-30", "blocks": [
        {"id": "evt_1", "title": "오사카성", "start_time": "13:00", "end_time": "16:00"}]}]}


# ---------- 맥락 형식 ----------
def test_dummy_context_has_agreed_keys():
    ctx = g.build_dummy_context()
    for k in ("today", "active_hours", "trip", "calendar_events", "tasks", "trip_schedule",
              "biometrics", "weather", "spending", "products"):
        assert k in ctx, k
    assert all(e["date"] for e in ctx["trip_schedule"])
    assert all("walking_level" in e for e in ctx["calendar_events"])
    assert len(ctx["biometrics"]["recent"]) == 3


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:
            failed += 1; print("FAIL", name, "->", repr(e)[:200])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
