"""구매 어드바이저 노드 (구매 에이전트 · 서경) — 여행 시나리오 구매 상담

흐름 (purchase_agent_detail 여행 버전):
  ① 구매 의도 파싱 (llm_small)      : 품목·카테고리·구매하려던 이유·비교 대상·수량 기준
  ② 관련 이력만 추리기 (코드)        : 같은 카테고리 구매 이력·거절 이력만 사용 (PC1)
  ③ 후보별 계산 (코드)              : 남은 일정 도보 '많음' 날 수 → 필요량 → 상자 수 → 여행 예산 비교
  ④ 충동 신호 검사 (코드)            : 23시 이후 / 최근 3일 연속 수면 6시간 미만 (PA2)
  ⑤ 후보 비교·선택 (llm_large)      : 코드가 계산한 표 안에서만 고르고 하루 사용량·비교 이유 작성
  ⑥ PurchaseProposal 확정 (코드)    : 수량·금액·보류 여부는 코드가 최종 결정

원칙: 숫자(수량·금액·날짜)는 LLM이 아니라 코드가 계산한다. LLM은 '어느 후보가 맞는지'와 '왜'만 쓴다.
     LLM 호출이 실패해도 코드만으로 제안을 완성한다(데모 중 API 오류 대비).
출력: state.PurchaseProposal (+ Critic·화면이 쓰기 좋은 quantity_detail, budget_check, revisit_condition)
"""
import json
import math
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from categories import DEFAULT_CATEGORY, PURCHASE_CATEGORIES, URGENT_NEED_EXAMPLES, normalize_category
from config import llm_large, llm_small
from state import get_last_user_message
from contextlib import closing

# ---------------------------------------------------------------------
# 설정값 (Critic 조건표와 같은 기준. 바꿀 때는 원형 님 critic_rules.py와 함께)
# ---------------------------------------------------------------------
HEAVY_WALKING = "많음"
LATE_NIGHT_HOUR = 23             # PA2: 23시 이후 상담
SHORT_SLEEP_HOURS = 6.0          # PA2: 최근 3일 연속 수면 6시간 미만
SHORT_SLEEP_DAYS = 3
MAX_OVERBUY_RATIO = 2            # PA5: 추천 매수가 필요량의 2배를 넘으면 과다
DEFAULT_DAILY_USAGE = 2          # LLM이 하루 사용량을 못 정했을 때 (예: 종아리 양쪽 1매씩)
DUPLICATE_LOOKBACK_DAYS = 7      # PA4: 최근 같은 품목 구매 확인 기간

# 시연용 고정 환율 (1 외화 = N원). 실제 환율이 아니므로 .env의 FX_JPY_KRW 등으로 덮어쓸 것.
# 여행 예산(trip.currency)과 상품 가격(product.currency)의 통화가 다를 때만 쓴다.
_FX_DEFAULT_TO_KRW = {"KRW": 1.0, "JPY": 9.5, "USD": 1400.0}


def _fx_to_krw(currency: str) -> Optional[float]:
    cur = (currency or "").upper()
    env = os.getenv(f"FX_{cur}_KRW")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _FX_DEFAULT_TO_KRW.get(cur)


def convert_currency(amount: float, from_cur: str, to_cur: str) -> Optional[float]:
    """통화 변환. 환율을 모르는 통화면 None (→ 예산 비교를 건너뛰고 근거에 표시)."""
    from_cur, to_cur = (from_cur or "").upper(), (to_cur or "").upper()
    if not from_cur or not to_cur or from_cur == to_cur:
        return float(amount)
    a, b = _fx_to_krw(from_cur), _fx_to_krw(to_cur)
    if not a or not b:
        return None
    return amount * a / b


# ---------------------------------------------------------------------
# LLM 구조화 출력 스키마
# ---------------------------------------------------------------------
class PurchaseIntent(BaseModel):
    item: str = Field(description="사용자가 사려는 품목을 일반 명사로 (예: '다리 피로 회복 패치')")
    category: str = Field(description=f"다음 중 하나: {', '.join(PURCHASE_CATEGORIES)}")
    purchase_context: str = Field(description="사용자가 말한 구매하려던 이유·상황 한 줄. 사용자가 말하지 않은 내용은 넣지 말 것")
    relevant_product_ids: List[str] = Field(
        default_factory=list,
        description="제공된 상품 목록 중 이번 고민과 관련된 상품 id. 사용자가 특정 상품만 물었으면 그 상품만",
    )
    usage_driver: Literal["walking_days", "trip_days", "one_time"] = Field(
        description="필요 수량이 무엇에 비례하는가. walking_days: 많이 걷는 날마다 쓰는 것(파스·휴족시간 등) / "
                    "trip_days: 남은 여행 날마다 쓰는 것 / one_time: 한 개면 되는 것(기념품·기기 등)"
    )
    is_urgent_need: bool = Field(
        description=f"건강·안전에 꼭 필요한 긴급 구매인가 (예: {', '.join(URGENT_NEED_EXAMPLES[:5])}). 피로 회복 용품은 False"
    )


class CandidateNote(BaseModel):
    id: str = Field(description="후보 상품 id (제공된 표의 id 그대로)")
    note: str = Field(description="이 후보의 장단점 한 줄. 제공된 표의 가격·매수·남은 일정만 근거로 쓸 것")


class AdvisorChoice(BaseModel):
    chosen_id: Optional[str] = Field(default=None, description="추천할 후보 id. 고를 수 있는 후보(selectable=true)가 없으면 null")
    daily_usage: int = Field(default=DEFAULT_DAILY_USAGE, ge=1, le=20, description="선택한 상품의 하루 사용량(매·개). 사용자가 말한 양이 있으면 그 값")
    daily_usage_basis: str = Field(default="", description="하루 사용량을 그렇게 본 이유 한 줄 (예: '양쪽 종아리에 1매씩')")
    criteria: List[str] = Field(default_factory=list, description="이번 선택에 쓴 기준 (예: '남은 일정 도보량', '여행 예산', '매당 가격'). 구매 이력에서 온 기준이면 어떤 이력인지 함께")
    candidate_notes: List[CandidateNote] = Field(default_factory=list, description="후보별 비교 메모")


# ---------------------------------------------------------------------
# ① 구매 의도 파싱
# ---------------------------------------------------------------------
_WALKING_WORDS = ("걷", "걸어", "걸었", "다리", "발", "파스", "휴족", "종아리", "물집")


def _fallback_intent(user_message: str, products: List[dict]) -> dict:
    """LLM 없이 쓰는 최소 의도 파악 (API 오류 대비)."""
    mentioned = [str(p.get("id")) for p in products
                 if any(tok and tok in user_message for tok in str(p.get("name", "")).split())]
    walking = any(w in user_message for w in _WALKING_WORDS)
    return {
        "item": (user_message.strip()[:30] or "구매 상담"),
        "category": "건강·위생" if walking else DEFAULT_CATEGORY,
        "purchase_context": user_message.strip()[:80],
        "relevant_product_ids": mentioned or [str(p.get("id")) for p in products],
        "usage_driver": "walking_days" if walking else "one_time",
        "is_urgent_need": False,
    }


def parse_purchase_intent(user_message: str, products: List[dict]) -> dict:
    product_list = [{"id": str(p.get("id")), "name": p.get("name")} for p in products]
    prompt = f"""당신은 여행 중 구매 상담 에이전트의 '구매 의도 파악' 단계입니다.
사용자 메시지에서 무엇을, 왜 사려는지만 뽑으세요. 추천이나 판단은 하지 않습니다.

[규칙]
- purchase_context는 사용자가 실제로 말한 이유·상황만 한 줄로 적습니다. 없는 사정을 지어내지 마세요.
- category는 반드시 다음 중 하나: {', '.join(PURCHASE_CATEGORIES)}
- relevant_product_ids는 아래 상품 목록의 id만 쓸 수 있습니다.

[사용자 메시지]
{user_message}

[비교할 수 있는 상품 목록]
{json.dumps(product_list, ensure_ascii=False)}
"""
    try:
        res: PurchaseIntent = llm_small.with_structured_output(PurchaseIntent).invoke(prompt)
        intent = res.model_dump()
    except Exception as e:  # API 오류·구조화 실패 → 코드 기본값
        print(f" -> [경고] 구매 의도 LLM 파싱 실패, 기본 규칙 사용: {e}")
        intent = _fallback_intent(user_message, products)

    intent["category"] = normalize_category(intent.get("category"))
    valid_ids = {str(p.get("id")) for p in products}
    intent["relevant_product_ids"] = [i for i in intent.get("relevant_product_ids", []) if i in valid_ids]
    return intent


# ---------------------------------------------------------------------
# ③ 남은 일정 → 필요량
# ---------------------------------------------------------------------
def _parse_date(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def remaining_schedule_summary(trip_schedule: List[dict], today_str: str) -> dict:
    """오늘(오늘 밤 포함) 이후 여행 일정에서 도보 '많음'인 날을 센다. (Critic PA5·PB4와 같은 기준)

    반환: {remaining_dates, heavy_dates, unknown_dates, has_schedule}
    - unknown_dates: walking_level이 비었거나 'unknown'이라 판단할 수 없는 날 (지어내지 않고 그대로 알림)
    """
    today = _parse_date(today_str) or date.today()
    remaining, heavy, unknown = [], [], []
    for day in trip_schedule or []:
        d = _parse_date(day.get("date"))
        if d is None or d < today:
            continue
        remaining.append(d.isoformat())
        levels = [str(ev.get("walking_level") or "").strip() for ev in day.get("events", [])]
        if HEAVY_WALKING in levels:
            heavy.append(d.isoformat())
        elif levels and all(lv in ("", "unknown") for lv in levels):
            unknown.append(d.isoformat())
    return {
        "remaining_dates": remaining,
        "heavy_dates": heavy,
        "unknown_dates": unknown,
        "has_schedule": bool(remaining),
    }


def usage_days(usage_driver: str, schedule: dict) -> Optional[int]:
    """수량의 기준이 되는 날 수. None이면 일정 정보로 판단 불가."""
    if usage_driver == "one_time":
        return None
    if not schedule["has_schedule"]:
        return None
    if usage_driver == "trip_days":
        return len(schedule["remaining_dates"])
    # walking_days: 도보량을 알 수 없는 날만 남았으면 판단 불가
    if not schedule["heavy_dates"] and schedule["unknown_dates"]:
        return None
    return len(schedule["heavy_dates"])


def plan_quantity(needed_units: Optional[int], unit_count: int) -> dict:
    """필요량을 채우는 최소 상자 수. 필요량을 모르면 1상자."""
    unit_count = max(int(unit_count or 1), 1)
    if not needed_units:
        return {"quantity": 1, "total_units": unit_count, "overbuy": False}
    quantity = max(math.ceil(needed_units / unit_count), 1)
    total = quantity * unit_count
    return {"quantity": quantity, "total_units": total, "overbuy": total > needed_units * MAX_OVERBUY_RATIO}


# ---------------------------------------------------------------------
# ④ 충동 신호 (PA2)
# ---------------------------------------------------------------------
def detect_impulse_signals(context_data: dict, now: Optional[datetime] = None) -> List[str]:
    signals = []
    now_hhmm = context_data.get("now")  # 시연·테스트용으로 맥락에 "HH:MM"을 넣으면 그 시각 기준
    hour = None
    if now_hhmm:
        try:
            hour = int(str(now_hhmm).split(":")[0])
        except ValueError:
            hour = None
    if hour is None:
        hour = (now or datetime.now()).hour
    if hour >= LATE_NIGHT_HOUR:
        signals.append(f"심야 상담 ({LATE_NIGHT_HOUR}시 이후)")

    recent = (context_data.get("biometrics") or {}).get("recent") or []
    recent = sorted((r for r in recent if r.get("date")), key=lambda r: r["date"])[-SHORT_SLEEP_DAYS:]
    hours = [0.0 if r.get("sleep_type") == "all_nighter" else r.get("sleep_hours") for r in recent]
    if len(hours) == SHORT_SLEEP_DAYS and all(h is not None and h < SHORT_SLEEP_HOURS for h in hours):
        signals.append(f"최근 {SHORT_SLEEP_DAYS}일 연속 수면 {SHORT_SLEEP_HOURS:g}시간 미만 ({', '.join(f'{h:g}시간' for h in hours)})")
    return signals


# ---------------------------------------------------------------------
# ② 관련 이력 (같은 카테고리만, PC1) · 거절 이력 (PD2)
# ---------------------------------------------------------------------
def related_purchase_records(spending: dict, category: str) -> List[dict]:
    if (spending or {}).get("data_quality") == "insufficient":
        return []
    return [r for r in (spending or {}).get("records", []) if normalize_category(r.get("category")) == category]


def recent_duplicates(records: List[dict], product_names: List[str], today_str: str) -> List[dict]:
    """PA4: 최근 같은 상품을 이미 샀는지 (이름이 서로 포함되는 경우만. 추측하지 않음)."""
    today = _parse_date(today_str) or date.today()
    found = []
    for r in records:
        d = _parse_date(r.get("purchase_date"))
        if d is None or (today - d).days > DUPLICATE_LOOKBACK_DAYS or (today - d).days < 0:
            continue
        name = str(r.get("item_name", ""))
        if name and any(name in p or p in name for p in product_names if p):
            found.append(r)
    return found


def rejected_for_category(rejected_history: List[dict], category: str) -> List[dict]:
    """같은 카테고리에서 사용자가 '거절'한 구매 추천만 (수정 요청 modify는 요청 문장으로 따로 반영)."""
    out = []
    for h in rejected_history or []:
        ch = h.get("changes")
        if h.get("status") != "rejected" or not isinstance(ch, dict) or "decision" not in ch:
            continue
        if normalize_category(ch.get("category")) == category:
            out.append(h)
    return out


# ---------------------------------------------------------------------
# ③ 후보 표 만들기
# ---------------------------------------------------------------------
def build_candidates(products: List[dict], needed_units: Optional[int], trip: dict, spending: dict,
                     rejected: List[dict]) -> tuple[List[dict], dict]:
    trip = trip or {}
    trip_currency = (trip.get("currency") or "").upper()
    budget = trip.get("budget")
    spent = (spending or {}).get("trip_spent") or 0
    remaining_budget = (budget - spent) if budget else None   # 예산 미설정(None·0)이면 검사 생략 (PA1)

    rejected_names = {str((h.get("changes") or {}).get("recommended", {}).get("name"))
                      for h in rejected if (h.get("changes") or {}).get("recommended")}
    rejected_tags = {t for h in rejected for t in (h.get("reject_reason_tags") or [])}

    candidates = []
    for p in products:
        price = p.get("price")
        if price is None:
            continue
        cur = (p.get("currency") or trip_currency or "KRW").upper()
        q = plan_quantity(needed_units, p.get("unit_count") or 1)
        total_price = price * q["quantity"]
        total_trip = convert_currency(total_price, cur, trip_currency or cur)
        within = None if (remaining_budget is None or total_trip is None) else total_trip <= remaining_budget
        was_rejected = str(p.get("name")) in rejected_names
        candidates.append({
            "id": str(p.get("id")),
            "name": p.get("name"),
            "price": price,
            "currency": cur,
            "unit_count": p.get("unit_count") or 1,
            "quantity": q["quantity"],
            "total_units": q["total_units"],
            "total_price": total_price,
            "total_in_trip_currency": None if total_trip is None else round(total_trip),
            "unit_price": round(price / max(p.get("unit_count") or 1, 1), 1),
            "within_budget": within,
            "overbuy": q["overbuy"],
            "previously_rejected": was_rejected,
            "source": p.get("source"),
            "source_url": p.get("source_url"),
            "selectable": (within is not False) and not was_rejected,
        })

    budget_check = {
        "budget": budget, "trip_spent": spent, "remaining": remaining_budget,
        "currency": trip_currency or None, "rejected_reason_tags": sorted(rejected_tags),
    }
    return candidates, budget_check


def _default_choice(candidates: List[dict]) -> Optional[dict]:
    """코드 기본 선택: 고를 수 있는 후보 중 과다 구매가 아닌 것 우선, 그다음 총액이 가장 싼 것."""
    pool = [c for c in candidates if c["selectable"]]
    if not pool:
        return None
    return min(pool, key=lambda c: (c["overbuy"],
                                    c["total_in_trip_currency"] if c["total_in_trip_currency"] is not None else c["total_price"]))


# ---------------------------------------------------------------------
# ⑤ 후보 비교·선택 (LLM)
# ---------------------------------------------------------------------
def choose_with_llm(user_message: str, intent: dict, candidates: List[dict], schedule: dict, days: Optional[int],
                    budget_check: dict, history: List[dict], rejected: List[dict], critic_feedback: str) -> Optional[dict]:
    table = [{k: c[k] for k in ("id", "name", "price", "currency", "unit_count", "unit_price", "within_budget",
                                "previously_rejected", "selectable")} for c in candidates]
    prompt = f"""당신은 여행 중 구매 상담 에이전트입니다. 아래 표의 후보 중에서 사용자에게 가장 맞는 것을 고르세요.
수량과 금액 계산은 시스템이 합니다. 당신은 (1) 어느 후보인지 (2) 하루 사용량 (3) 비교 이유만 정합니다.

[절대 규칙 — Critic 조건표]
- 표에 없는 가격·매수·성분·효능·리뷰를 쓰지 마세요 (PB1). 상품 효과의 우열을 단정하지 마세요.
- selectable=false 인 후보는 고르지 마세요 (예산 초과 PA1, 사용자가 거절한 추천 PD2).
- 남은 일정과 반대되는 이유를 쓰지 마세요 (PB4). 남은 일정은 아래 값이 전부입니다.
- 같은 카테고리 구매 이력만 근거로 쓰고, 이력이 없으면 취향을 지어내지 마세요 (PC1, PC2).
  이력이 1~2건이면 '항상', '늘' 같은 확정 표현을 쓰지 마세요 (PC3).
- 거절 이력의 이유(reject_reason_tags, feedback)를 반영하세요. 예: '가격' → 더 싼 후보, '수량' → 더 적은 매수.

[사용자 메시지]
{user_message}

[구매 의도]
{json.dumps({k: intent[k] for k in ("item", "category", "purchase_context", "usage_driver")}, ensure_ascii=False)}

[후보 표]
{json.dumps(table, ensure_ascii=False)}

[남은 일정] 도보 '많음'인 날: {schedule["heavy_dates"] or "없음"} / 남은 날짜: {schedule["remaining_dates"] or "일정 정보 없음"} / 수량 기준 날 수: {days if days is not None else "판단 불가"}
[여행 예산] {json.dumps(budget_check, ensure_ascii=False)}
[같은 카테고리 구매 이력] {json.dumps(history, ensure_ascii=False) if history else "없음"}
[같은 카테고리 거절 이력] {json.dumps(rejected, ensure_ascii=False) if rejected else "없음"}
"""
    if critic_feedback:
        prompt += f"\n[⚠️ 이전 Critic 반려 피드백 — 반드시 반영]\n{critic_feedback}\n"
    try:
        res: AdvisorChoice = llm_large.with_structured_output(AdvisorChoice).invoke(prompt)
        return res.model_dump()
    except Exception as e:
        print(f" -> [경고] 후보 비교 LLM 실패, 코드 기본 선택 사용: {e}")
        return None


# ---------------------------------------------------------------------
# 포맷 도우미
# ---------------------------------------------------------------------
def _money(amount, currency) -> str:
    if amount is None:
        return "금액 정보 없음"
    return f"{round(amount):,} {currency}" if currency else f"{round(amount):,}"


def _tomorrow(today_str: str) -> str:
    return ((_parse_date(today_str) or date.today()) + timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------
# 노드 본체
# ---------------------------------------------------------------------
def purchase_advisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[구매 어드바이저] 남은 일정·여행 예산 기준으로 구매 상담 초안 작성 중...")
    ctx = state.get("context_data") or {}
    today_str = ctx.get("today") or date.today().isoformat()
    trip = ctx.get("trip") or {}
    spending = ctx.get("spending") or {}
    products = ctx.get("products") or []
    user_message = get_last_user_message(state)

    prev = state.get("draft_proposal") or {}
    is_purchase_prev = "decision" in prev and "item" in prev
    decision_status = (state.get("user_decision") or {}).get("status")
    is_rework = state.get("critic_verdict") == "rework" and is_purchase_prev
    is_modify = decision_status == "modify" and is_purchase_prev
    critic_feedback = (state.get("critic_result") or {}).get("feedback", "") if is_rework else ""

    # ① 구매 의도 — 재작업·수정 요청이면 같은 상담이므로 앞서 파악한 의도를 그대로 쓴다
    #    (수정 요청일 때 마지막 사용자 메시지는 "더 싼 걸로" 같은 수정 문장이라 품목이 들어 있지 않음)
    if (is_rework or is_modify) and prev.get("intent"):
        intent = dict(prev["intent"])
    else:
        intent = parse_purchase_intent(user_message, products)
    category = intent["category"]

    # ② 같은 카테고리 이력만
    history = related_purchase_records(spending, category)
    rejected = rejected_for_category(state.get("rejected_history") or [], category)

    # ③ 남은 일정 → 날 수
    schedule = remaining_schedule_summary(ctx.get("trip_schedule") or [], today_str)
    days = usage_days(intent["usage_driver"], schedule)

    relevant_ids = set(intent.get("relevant_product_ids") or [])
    pool = [p for p in products if str(p.get("id")) in relevant_ids] or products

    # ⑤ LLM 선택 (하루 사용량을 알아야 수량이 나오므로 1차 표는 기본 사용량으로 만든다)
    def table_for(daily: int):
        needed = (days * daily) if days else None
        cands, bcheck = build_candidates(pool, needed, trip, spending, rejected)
        return needed, cands, bcheck

    needed_units, candidates, budget_check = table_for(DEFAULT_DAILY_USAGE)
    choice = None
    if candidates:
        choice = choose_with_llm(user_message, intent, candidates, schedule, days, budget_check,
                                 history, rejected, critic_feedback)
    daily_usage = int((choice or {}).get("daily_usage") or DEFAULT_DAILY_USAGE)
    daily_basis = (choice or {}).get("daily_usage_basis") or "기본값 (양쪽에 1매씩)"
    if intent["usage_driver"] == "one_time":
        daily_usage = 1
    needed_units, candidates, budget_check = table_for(daily_usage)

    by_id = {c["id"]: c for c in candidates}
    chosen = by_id.get(str((choice or {}).get("chosen_id")))
    if not chosen or not chosen["selectable"]:
        chosen = _default_choice(candidates)   # LLM이 못 고르거나 고를 수 없는 후보를 골랐으면 코드가 고른다
    elif chosen["overbuy"] and not is_modify:
        # 필요량의 2배를 넘는 선택(PA5)이면 덜 남거나 더 싼 후보로 바꾼다.
        # 단, 사용자가 수정 요청으로 직접 고른 상품이면 그대로 두고 quantity_basis에 남는 양만 표시한다.
        chosen = _default_choice(candidates)
    notes = {n["id"]: n["note"] for n in (choice or {}).get("candidate_notes", []) if n.get("id") in by_id}
    for c in candidates:
        c["note"] = notes.get(c["id"], "")

    # ④ 충동 신호 (긴급 필요 품목은 적용하지 않음)
    impulse_signals = [] if intent.get("is_urgent_need") else detect_impulse_signals(ctx)

    # 근거 (코드가 실제 데이터에서만 만든다 → PB1·PB4 방지)
    evidence: List[str] = []
    if not schedule["has_schedule"]:
        evidence.append("trip_schedule: 남은 여행 일정 정보가 없음")
    else:
        evidence.append(f"trip_schedule: 오늘 포함 남은 {len(schedule['remaining_dates'])}일 중 도보 '많음' "
                        f"{len(schedule['heavy_dates'])}일 ({', '.join(schedule['heavy_dates']) or '없음'})")
        if schedule["unknown_dates"]:
            evidence.append(f"trip_schedule: 도보량을 알 수 없는 날 {', '.join(schedule['unknown_dates'])}")
    if budget_check["remaining"] is not None:
        evidence.append(f"trip.budget {_money(budget_check['budget'], budget_check['currency'])} 중 "
                        f"{_money(budget_check['trip_spent'], budget_check['currency'])} 사용 → "
                        f"남은 예산 {_money(budget_check['remaining'], budget_check['currency'])}")
    else:
        evidence.append("trip.budget: 여행 예산 미설정 → 예산 비교 생략")
    for c in candidates:
        src = c.get("source_url") or c.get("source") or "출처 없음"
        evidence.append(f"products[{c['id']}]: {c['name']} {_money(c['price'], c['currency'])} / {c['unit_count']}매·개 (출처: {src})")
    duplicates = recent_duplicates(history, [c["name"] for c in candidates] + [intent["item"]], today_str)
    for r in duplicates:
        evidence.append(f"spending.records: 최근 구매 '{r.get('item_name')}' ({r.get('purchase_date')}) — 중복 구매 여부 확인 필요")
    for h in rejected:
        evidence.append(f"rejected_history: 같은 카테고리 추천 거절 (이유: {', '.join(h.get('reject_reason_tags') or []) or h.get('feedback') or '미기재'})")

    # ⑥ 최종 결정
    hold_reasons: List[str] = []
    revisit_condition = None
    if not candidates:
        hold_reasons.append("비교할 상품 정보(products)가 없어 가격·매수를 확인할 수 없음")
        revisit_condition = "상품 이름과 가격을 알려 주면 다시 비교"
    elif chosen is None:
        if any(c["within_budget"] is False for c in candidates):
            hold_reasons.append("모든 후보가 남은 여행 예산을 넘음")
            revisit_condition = "더 저렴한 상품이 있거나 예산을 조정하면 다시 검토"
        else:
            hold_reasons.append("같은 상황에서 이미 거절한 추천뿐이라 다시 권하지 않음")
            revisit_condition = "다른 상품이나 달라진 상황을 알려 주면 다시 검토"
    if intent["usage_driver"] == "walking_days" and schedule["has_schedule"] and days == 0:
        hold_reasons.append("남은 일정에 도보 '많음'인 날이 없어 필요성이 낮음")
        revisit_condition = revisit_condition or "일정이 바뀌어 많이 걷는 날이 생기면 다시 검토"
    if impulse_signals:
        hold_reasons.append("충동구매 신호: " + "; ".join(impulse_signals))
        revisit_condition = revisit_condition or "내일 아침에도 필요하면 다시 상담"
    if duplicates and chosen is not None:
        hold_reasons.append("최근 같은 상품을 구매한 기록이 있음 — 남은 양을 먼저 확인")
        revisit_condition = revisit_condition or "남은 양이 부족하면 다시 상담"

    decision = "hold" if hold_reasons else "recommend"

    # 수량 근거 문장 (Critic PA5·PB4가 읽는 quantity_basis)
    if chosen is None:
        quantity_basis = None
    elif intent["usage_driver"] == "one_time":
        quantity_basis = "한 번 사면 되는 품목 → 1개"
    elif days is None:
        reason = "남은 일정의 도보량 정보가 없어" if schedule["has_schedule"] else "남은 일정 정보가 없어"
        quantity_basis = f"{reason} 필요량을 계산하지 못함 → 최소 수량 {chosen['quantity']}개({chosen['total_units']}매)"
    else:
        unit_label = "도보 '많음'" if intent["usage_driver"] == "walking_days" else "여행"
        quantity_basis = (f"오늘 포함 남은 일정 중 {unit_label} {days}일 × 하루 {daily_usage}매({daily_basis}) = 필요량 {needed_units}매 "
                          f"→ {chosen['unit_count']}매입 {chosen['quantity']}개({chosen['total_units']}매)")
        if chosen["overbuy"]:
            quantity_basis += f" ※ 가장 작은 포장이 필요량의 {MAX_OVERBUY_RATIO}배를 넘음 (남는 양 {chosen['total_units'] - needed_units}매)"

    recommended = None
    if chosen is not None and decision == "recommend":
        recommended = {
            "name": chosen["name"], "price": chosen["price"], "currency": chosen["currency"],
            "quantity": chosen["quantity"], "unit_count": chosen["unit_count"],
            "total_price": chosen["total_price"],
            "total_in_trip_currency": chosen["total_in_trip_currency"],
        }

    criteria = [c for c in (choice or {}).get("criteria", []) if c] or (
        ["남은 일정 도보량", "여행 예산", "매당 가격"] if intent["usage_driver"] == "walking_days" else ["여행 예산", "가격"])

    proposal = {
        "decision": decision,
        "item": intent["item"],
        "category": category,
        "recommended": recommended,
        "quantity_basis": quantity_basis,
        "purchase_context": intent["purchase_context"],
        "criteria": criteria,
        "candidates": candidates,
        "impulse_signals": impulse_signals,
        "evidence": evidence,
        "revisit_at": _tomorrow(today_str) if decision == "hold" else None,
        # --- 아래는 PurchaseProposal 기본 칸 외에 Critic·화면이 쓰기 좋게 덧붙인 값 ---
        "revisit_condition": revisit_condition if decision == "hold" else None,
        "hold_reasons": hold_reasons,
        "considered": None if chosen is None else {"id": chosen["id"], "name": chosen["name"], "quantity": chosen["quantity"]},
        "quantity_detail": {
            "usage_driver": intent["usage_driver"],
            "heavy_walking_dates": schedule["heavy_dates"],
            "days": days,
            "daily_usage": daily_usage if days is not None else None,
            "needed_units": needed_units,
            "total_units": None if chosen is None else chosen["total_units"],
        },
        "budget_check": budget_check,
        "intent": intent,   # 재작업·수정 요청 때 같은 상담을 이어 가기 위한 값
    }
    print(f" -> 결정: {decision}" + (f" / {recommended['name']} × {recommended['quantity']}" if recommended else "")
          + (f" / 보류 이유: {hold_reasons}" if hold_reasons else ""))
    return {"draft_proposal": proposal}


# ---------------------------------------------------------------------
# 승인·거절 후 저장 (실행 에이전트가 호출) — purchase_decisions 테이블
# ---------------------------------------------------------------------
def save_purchase_decision(state: Dict[str, Any], user_response: str = "approved") -> int:
    """구매 상담 결과 한 건을 장기 메모리에 저장하고 행 id를 돌려준다.
    리드 action_executor의 purchase_consultation 분기에서 호출:  save_purchase_decision(state)"""
    from db_init import get_connection, init_db

    proposal = state.get("draft_proposal") or {}
    decision = state.get("user_decision") or {}
    trip = (state.get("context_data") or {}).get("trip") or {}
    init_db()
    with closing(get_connection()) as conn, conn:
        cur = conn.execute(
            """INSERT INTO purchase_decisions
               (trip_id, item, category, decision, recommended, user_response,
                reject_reason_tags, purchase_context, evidence, revisit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trip.get("trip_id"),
                proposal.get("item") or "미상",
                proposal.get("category"),
                proposal.get("decision"),
                json.dumps(proposal.get("recommended"), ensure_ascii=False),
                user_response,
                json.dumps(decision.get("reason_tags") or [], ensure_ascii=False),
                proposal.get("purchase_context"),
                json.dumps(proposal.get("evidence") or [], ensure_ascii=False),
                proposal.get("revisit_at"),
            ),
        )
        return cur.lastrowid
