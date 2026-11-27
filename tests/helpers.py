"""테스트 공용 도우미: LLM을 호출하지 않는 그래프를 만들어 경로(라우팅)만 검사한다."""
import importlib.util
import io
import contextlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fake_planner_module():
    """채린 님 planner_nodes 대신 쓰는 가짜 모듈 (Gemini 호출 없음)"""
    m = types.ModuleType("planner_nodes")
    m.condition_analyzer_node = lambda s: {"condition_score": {"morning": 80, "afternoon": 40}}

    def planner(state):
        if state.get("request_type") == "trip_planning":
            return {"draft_proposal": {"trip_id": "trip_osaka", "days": [
                {"date": "2026-11-30", "schedule_blocks": [{"id": "evt_d1_1", "title": "오사카성",
                                                            "start_time": "13:00", "end_time": "16:00"}]}]}}
        return {"draft_proposal": {"changed_items": [{"id": "evt_d2_2", "title": "도톤보리 저녁"}],
                                   "postponed_tasks": [], "schedule_blocks": []}}
    m.planner_agent_node = planner
    return m


def fake_purchase_module():
    """서경 님 purchase_nodes 대신 쓰는 가짜 모듈 (Gemini 호출 없음). 형식은 state.PurchaseProposal"""
    m = types.ModuleType("purchase_nodes")
    m.purchase_advisor_node = lambda s: {"draft_proposal": {
        "decision": "recommend", "item": "다리 피로 회복 패치", "category": "건강·위생",
        "purchase_context": "유니버설 스튜디오에서 약 2만 보를 걸어 다리 통증",
        "recommended": {"name": "휴족시간 18매입", "price": 932, "currency": "JPY", "quantity": 1},
        "quantity_basis": "남은 일정 중 도보 '많음' 2일 × 하루 2매 = 4매 → 18매입 1개",
        "criteria": [], "candidates": [], "impulse_signals": [], "evidence": [], "revisit_at": None}}
    return m


def load_graph(critic_always_severity=None, name="graph_under_test"):
    """graph_core를 새로 불러와서 반환 (가짜 플래너 사용).
    critic_always_severity="critical"/"high"면 임시 Critic이 항상 그 심각도로 반려하도록 바꾼다."""
    src_path = ROOT / "graph_core.py"
    src = src_path.read_text(encoding="utf-8").split("# [4] 로컬 자가 검증용")[0]
    if critic_always_severity:
        src = src.replace("if prev_count < 1:", "if True:")
        src = src.replace('"severity": "high",', f'"severity": "{critic_always_severity}",', 1)

    saved = sys.modules.get("planner_nodes")
    saved_purchase = sys.modules.get("purchase_nodes")
    sys.modules["planner_nodes"] = fake_planner_module()
    sys.modules["purchase_nodes"] = fake_purchase_module()
    try:
        mod = types.ModuleType(name)
        mod.__file__ = str(src_path)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, str(src_path), "exec"), mod.__dict__)
    finally:
        if saved_purchase is not None:
            sys.modules["purchase_nodes"] = saved_purchase
        else:
            sys.modules.pop("purchase_nodes", None)
        if saved is not None:
            sys.modules["planner_nodes"] = saved
        else:
            sys.modules.pop("planner_nodes", None)
    return mod


def run(app, thread_id, inp, decisions=()):
    """그래프를 실행하고, 사용자 결정(decisions)을 차례로 넣어 재개. 지나간 노드 목록을 돌려줌"""
    cfg = {"configurable": {"thread_id": thread_id}}
    nodes = []

    def go(x):
        for ev in app.stream(x, cfg):
            nodes.extend(k for k in ev if k != "__interrupt__")

    with contextlib.redirect_stdout(io.StringIO()):
        go(inp)
        for d in decisions:
            app.update_state(cfg, {"user_decision": d})   # as_node 없이 (README 규칙)
            go(None)
    return cfg, nodes


def quiet(f, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args)
