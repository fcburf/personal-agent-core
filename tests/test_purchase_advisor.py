"""구매 어드바이저 테스트 (LLM 호출 없음). 실행: python tests/test_purchase_advisor.py  또는  pytest tests

가짜 LLM을 끼워 넣어 '코드가 계산하는 부분'(남은 일정 → 수량, 여행 예산, 충동 신호, 거절 이력)만 검사한다.
데모 데이터: 지인 님 오사카·교토 2박 3일 (2일 차 저녁, 예산 800,000 KRW, 상품은 JPY).
"""
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeLLM:
    """with_structured_output(schema).invoke(prompt) → 미리 정한 값을 schema로 돌려준다."""
    def __init__(self, answers=None, fail=False):
        self.answers, self.fail, self.prompts = answers or {}, fail, []

    def with_structured_output(self, schema):
        outer = self

        class _Runner:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                if outer.fail:
                    raise RuntimeError("fake LLM failure")
                return schema(**outer.answers[schema.__name__])
        return _Runner()


# config(Gemini 초기화)를 가짜로 바꾼 뒤 purchase_nodes를 불러온다 → API 키 없이 실행
_fake_config = types.ModuleType("config")
_fake_config.llm_small, _fake_config.llm_large = FakeLLM(), FakeLLM()
_saved_config = sys.modules.get("config")
sys.modules["config"] = _fake_config
sys.modules.pop("purchase_nodes", None)
import purchase_nodes as pn  # noqa: E402
if _saved_config is not None:
    sys.modules["config"] = _saved_config
else:
    sys.modules.pop("config", None)

from langchain_core.messages import HumanMessage  # noqa: E402

DEMO_MESSAGE = ("어제 교토에서도 많이 걷고 오늘 유니버설에서도 하루 종일 걸어서 다리가 아파. "
                "휴족시간을 사려고 하는데 몇 장 정도 사는 게 좋을까? 내일도 많이 걸어야 해?")

INTENT = {"item": "다리 피로 회복 패치", "category": "건강·위생",
          "purchase_context": "교토·USJ에서 이틀 연속 많이 걸어 다리 통증",
          "relevant_product_ids": ["A", "B"], "usage_driver": "walking_days", "is_urgent_need": False}
CHOICE = {"chosen_id": "A", "daily_usage": 6, "daily_usage_basis": "양쪽 종아리·발바닥에 3매씩",
          "criteria": ["남은 일정 도보량", "여행 예산"], "candidate_notes": [{"id": "A", "note": "매수가 필요량에 가까움"}]}


def demo_context(**over):
    ctx = {
        "today": "2026-12-02", "now": "20:30",
        "trip": {"trip_id": "trip_osaka", "destination": "일본 오사카·교토", "start_date": "2026-12-01",
                 "end_date": "2026-12-03", "budget": 800000, "currency": "KRW"},
        "trip_schedule": [
            {"date": "2026-12-01", "events": [{"id": "d1", "title": "후시미 이나리 신사", "walking_level": "많음"}]},
            {"date": "2026-12-02", "events": [{"id": "d2a", "title": "난바 → USJ 이동", "walking_level": "보통"},
                                              {"id": "d2b", "title": "슈퍼 닌텐도 월드", "walking_level": "많음"}]},
            {"date": "2026-12-03", "events": [{"id": "d3a", "title": "아침 식사", "walking_level": "적음"},
                                              {"id": "d3b", "title": "도톤보리 관광·기념품 쇼핑", "walking_level": "많음"}]},
        ],
        "biometrics": {"recent": [
            {"date": "2026-11-30", "sleep_type": "manual", "sleep_hours": 7.33, "fatigue_level": 2},
            {"date": "2026-12-01", "sleep_type": "manual", "sleep_hours": 5.5, "fatigue_level": 4},
            {"date": "2026-12-02", "sleep_type": "manual", "sleep_hours": 5.17, "fatigue_level": 5}]},
        "spending": {"records": [], "trip_spent": 310000, "data_quality": "ok"},
        "products": [
            {"id": "A", "name": "휴족시간 18매입", "price": 932, "currency": "JPY", "unit_count": 18, "source": "dummy"},
            {"id": "B", "name": "샤론파스 40매입", "price": 1180, "currency": "JPY", "unit_count": 40, "source": "dummy"}],
    }
    ctx.update(over)
    return ctx


def advise(ctx=None, intent=None, choice=None, fail=False, **state):
    pn.llm_small = FakeLLM({"PurchaseIntent": intent or INTENT}, fail=fail)
    pn.llm_large = FakeLLM({"AdvisorChoice": choice or CHOICE}, fail=fail)
    s = {"messages": [HumanMessage(content=DEMO_MESSAGE)], "context_data": ctx or demo_context(), **state}
    with contextlib.redirect_stdout(io.StringIO()):
        return pn.purchase_advisor_node(s)["draft_proposal"]


# ---------- 데모 시나리오 ----------
def test_demo_recommends_one_box_based_on_remaining_heavy_days():
    p = advise()
    assert p["decision"] == "recommend" and p["category"] == "건강·위생"
    assert p["recommended"]["name"] == "휴족시간 18매입" and p["recommended"]["quantity"] == 1
    q = p["quantity_detail"]
    assert q["heavy_walking_dates"] == ["2026-12-02", "2026-12-03"]      # 오늘(밤) + 내일, 지난 1일 차는 제외
    assert q["days"] == 2 and q["needed_units"] == 12 and q["total_units"] == 18
    assert "2일" in p["quantity_basis"] and "12매" in p["quantity_basis"]
    assert p["impulse_signals"] == [] and p["revisit_at"] is None      # 5.5h·5.17h지만 3일 연속은 아님


def test_proposal_has_every_field_critic_and_history_read():
    p = advise()
    for k in ("decision", "item", "category", "recommended", "quantity_basis", "purchase_context",
              "criteria", "candidates", "impulse_signals", "evidence", "revisit_at"):
        assert k in p, k
    assert {"name", "price", "currency", "quantity"} <= set(p["recommended"])
    assert all(c["unit_count"] for c in p["candidates"])


def test_budget_is_compared_in_trip_currency():
    p = advise()
    assert p["budget_check"]["remaining"] == 490000 and p["budget_check"]["currency"] == "KRW"
    assert p["recommended"]["total_in_trip_currency"] == round(932 * 9.5)   # 시연용 고정 환율
    assert all(c["within_budget"] for c in p["candidates"])


# ---------- 수량 ----------
def test_quantity_grows_with_need_and_stays_within_twice_the_need():
    ctx = demo_context(products=[{"id": "S", "name": "휴족시간 6매입", "price": 400, "currency": "JPY", "unit_count": 6}])
    p = advise(ctx, intent={**INTENT, "relevant_product_ids": ["S"]}, choice={**CHOICE, "chosen_id": "S"})
    assert p["recommended"]["quantity"] == 2 and p["quantity_detail"]["total_units"] == 12   # 필요량 12매 = 6매입 2개
    assert p["quantity_detail"]["total_units"] <= 2 * p["quantity_detail"]["needed_units"]


def test_overbuy_is_flagged_when_smallest_pack_exceeds_twice_the_need():
    p = advise(choice={**CHOICE, "chosen_id": "B", "daily_usage": 2})
    # 필요량 4매인데 후보 A(18매)·B(40매) 모두 2배 초과 → 덜 남는 쪽을 고르고 근거에 표시
    assert p["quantity_detail"]["needed_units"] == 4
    assert all(c["overbuy"] for c in p["candidates"]) and "넘음" in p["quantity_basis"]
    assert p["recommended"]["name"] == "휴족시간 18매입"


def test_no_heavy_walking_left_means_hold():
    ctx = demo_context(today="2026-12-03")
    ctx["trip_schedule"][2]["events"] = [{"id": "d3", "title": "공항 이동", "walking_level": "보통"}]
    p = advise(ctx)
    assert p["decision"] == "hold" and p["recommended"] is None
    assert p["revisit_at"] == "2026-12-04" and p["revisit_condition"]          # PD1


def test_unknown_walking_level_is_not_guessed():
    ctx = demo_context()
    for day in ctx["trip_schedule"]:
        for ev in day["events"]:
            ev["walking_level"] = "unknown"
    p = advise(ctx)
    assert p["quantity_detail"]["days"] is None and p["recommended"]["quantity"] == 1
    assert "도보량" in p["quantity_basis"] and any("알 수 없는" in e for e in p["evidence"])


def test_one_time_item_is_quantity_one():
    ctx = demo_context(products=[{"id": "K", "name": "키링", "price": 800, "currency": "JPY", "unit_count": 1}])
    p = advise(ctx, intent={**INTENT, "category": "기념품", "usage_driver": "one_time", "relevant_product_ids": ["K"]},
               choice={**CHOICE, "chosen_id": "K"})
    assert p["recommended"]["quantity"] == 1 and p["quantity_detail"]["needed_units"] is None


# ---------- 예산 (PA1) ----------
def test_over_budget_candidate_is_never_recommended():
    ctx = demo_context(spending={"records": [], "trip_spent": 790000, "data_quality": "ok"})   # 남은 예산 10,000원
    p = advise(ctx, choice={**CHOICE, "chosen_id": "B"})                                       # LLM이 비싼 B를 골라도
    assert p["recommended"]["name"] == "휴족시간 18매입"                                         # 8,854원 → 예산 안
    assert next(c for c in p["candidates"] if c["id"] == "B")["within_budget"] is False


def test_all_over_budget_means_hold():
    p = advise(demo_context(spending={"records": [], "trip_spent": 799000, "data_quality": "ok"}))
    assert p["decision"] == "hold" and p["recommended"] is None
    assert any("예산" in r for r in p["hold_reasons"])


def test_budget_not_set_skips_check():
    ctx = demo_context()
    ctx["trip"]["budget"] = None
    p = advise(ctx)
    assert p["decision"] == "recommend" and p["budget_check"]["remaining"] is None


# ---------- 충동 신호 (PA2) ----------
def test_late_night_holds_with_revisit():
    p = advise(demo_context(now="23:20"))
    assert p["decision"] == "hold" and p["impulse_signals"] and p["revisit_at"] == "2026-12-03"


def test_three_short_sleeps_is_a_signal_but_urgent_need_is_exempt():
    ctx = demo_context()
    ctx["biometrics"]["recent"][0]["sleep_hours"] = 5.0
    assert advise(ctx)["decision"] == "hold"
    p = advise(ctx, intent={**INTENT, "item": "해열제", "is_urgent_need": True})
    assert p["decision"] == "recommend" and p["impulse_signals"] == []


# ---------- 거절 이력 (PD2) ----------
def _rejection(tags=("가격",)):
    return {"status": "rejected", "feedback": "너무 비싸", "reject_reason_tags": list(tags),
            "purchase_context": "교토·USJ에서 이틀 연속 많이 걸어 다리 통증", "decided_at": "2026-12-02T20:40:00",
            "changes": {"decision": "recommend", "item": "다리 피로 회복 패치", "category": "건강·위생",
                        "recommended": {"name": "휴족시간 18매입", "price": 932, "currency": "JPY", "quantity": 1}}}


def test_rejected_product_is_not_recommended_again():
    p = advise(rejected_history=[_rejection()])
    assert p["recommended"]["name"] != "휴족시간 18매입"
    assert any("거절" in e for e in p["evidence"])


def test_rejection_in_other_category_is_ignored():
    r = _rejection()
    r["changes"]["category"] = "전자기기"
    assert advise(rejected_history=[r])["recommended"]["name"] == "휴족시간 18매입"


def test_all_rejected_means_hold():
    r2 = _rejection()
    r2["changes"]["recommended"] = {"name": "샤론파스 40매입", "price": 1180, "currency": "JPY", "quantity": 1}
    p = advise(rejected_history=[_rejection(), r2])
    assert p["decision"] == "hold" and p["recommended"] is None


# ---------- 중복 구매 (PA4) ----------
def test_recent_same_product_purchase_holds():
    ctx = demo_context(spending={"trip_spent": 310000, "data_quality": "ok", "records": [
        {"item_name": "휴족시간 18매입", "category": "건강·위생", "price": 932, "purchase_date": "2026-12-01"},
        {"item_name": "이어폰", "category": "전자기기", "price": 30000, "purchase_date": "2026-12-01"}]})
    p = advise(ctx)
    assert p["decision"] == "hold" and any("중복" in e for e in p["evidence"])
    assert not any("이어폰" in e for e in p["evidence"])                       # 다른 카테고리 이력은 쓰지 않음 (PC1)


# ---------- 재작업·수정 요청·오류 ----------
def test_modify_request_keeps_original_intent():
    first = advise()
    pn.llm_small = FakeLLM(fail=True)     # 의도를 다시 파싱하면 실패하도록
    pn.llm_large = FakeLLM({"AdvisorChoice": {**CHOICE, "chosen_id": "B"}})
    s = {"messages": [HumanMessage(content=DEMO_MESSAGE), HumanMessage(content="샤론파스로 바꿔줘")],
         "context_data": demo_context(), "draft_proposal": first, "user_decision": {"status": "modify"}}
    with contextlib.redirect_stdout(io.StringIO()):
        p = pn.purchase_advisor_node(s)["draft_proposal"]
    assert p["item"] == first["item"] and p["recommended"]["name"] == "샤론파스 40매입"
    assert "샤론파스로 바꿔줘" in pn.llm_large.prompts[-1]


def test_critic_feedback_is_passed_on_rework():
    first = advise()
    pn.llm_large = FakeLLM({"AdvisorChoice": CHOICE})
    s = {"messages": [HumanMessage(content=DEMO_MESSAGE)], "context_data": demo_context(), "draft_proposal": first,
         "critic_verdict": "rework", "critic_result": {"feedback": "PA5: 수량이 필요량보다 많음"}}
    with contextlib.redirect_stdout(io.StringIO()):
        pn.purchase_advisor_node(s)
    assert "PA5: 수량이 필요량보다 많음" in pn.llm_large.prompts[-1]


def test_llm_failure_still_produces_a_proposal():
    p = advise(fail=True)
    assert p["decision"] == "recommend" and p["recommended"]["name"] == "휴족시간 18매입"
    assert p["category"] == "건강·위생" and p["quantity_detail"]["days"] == 2


def test_no_products_or_no_trip_does_not_crash():
    p = advise(demo_context(products=[]))
    assert p["decision"] == "hold" and p["recommended"] is None
    p = advise(demo_context(trip=None, trip_schedule=[]))
    assert p["recommended"]["quantity"] == 1 and "일정 정보가 없어" in p["quantity_basis"]


# ---------- 저장 ----------
def test_save_purchase_decision_writes_row():
    import db_init
    saved = db_init.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        db_init.DB_PATH = Path(tmp) / "test.db"
        try:
            p = advise()
            row_id = pn.save_purchase_decision({"draft_proposal": p, "context_data": demo_context(),
                                                "user_decision": {"status": "approved"}})
            conn = db_init.get_connection()
            row = conn.execute("SELECT trip_id, item, decision, user_response FROM purchase_decisions WHERE id=?",
                               (row_id,)).fetchone()
            conn.close()
            assert row == ("trip_osaka", "다리 피로 회복 패치", "recommend", "approved")
        finally:
            db_init.DB_PATH = saved


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
