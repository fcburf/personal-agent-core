"""채린 님 플래너와 그래프의 연결 테스트 (가짜 LLM 사용, Gemini 호출 없음)"""
import io
import contextlib
from helpers import ROOT  # noqa: F401  (sys.path 설정)
from langchain_core.messages import HumanMessage, AIMessage
import planner_nodes as pn
import graph_core

ctx = graph_core.build_dummy_context()
EMPTY = pn.PlannerDraftOutput(schedule_blocks=[], changed_items=[], postponed_tasks=[])


class _FakeLLM:
    def __init__(self):
        self.calls, self.last_prompt = 0, ""

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls += 1
        self.last_prompt = "\n".join(str(m.content) for m in messages)
        return EMPTY


def _run_planner(state):
    fake = _FakeLLM()
    original = pn.llm_large
    pn.llm_large = fake
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = pn.planner_agent_node(state)
    finally:
        pn.llm_large = original
    return fake, out


def test_rest_request_read_from_last_human_message_even_if_ai_replied():
    fake, out = _run_planner({"messages": [HumanMessage(content="오늘은 쉴래"), AIMessage(content="네")],
                              "context_data": ctx, "condition_score": {}})
    assert out["draft_proposal"]["user_requested_rest"] is True


def test_critic_feedback_reaches_planner_prompt():
    fake, _ = _run_planner({"messages": [HumanMessage(content="오늘 일정 알려줘")], "context_data": ctx,
                            "condition_score": {}, "critic_verdict": "rework",
                            "critic_result": {"feedback": "A3-1 고정 일정 삭제: USJ 예약을 유지할 것"}})
    assert "A3-1 고정 일정 삭제" in fake.last_prompt


def test_internal_retries_capped():
    fake, out = _run_planner({"messages": [HumanMessage(content="오늘 일정 알려줘")],
                              "context_data": ctx, "condition_score": {}})
    assert fake.calls == pn.PLANNER_MAX_ATTEMPTS == 2
    assert out["draft_proposal"]["remaining_violations"]           # 남은 위반은 Critic으로 넘어감


def test_dummy_trip_context_drives_rule_checks():
    v = pn._validate_plan(EMPTY, ctx["calendar_events"], ctx["tasks"], ctx["today"], False)
    assert any("[A3-1]" in x and "evt_d2_1" in x for x in v), v   # USJ 예약(조정 불가)을 빼면 감지


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
