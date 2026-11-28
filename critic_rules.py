"""Critic 공용 규칙 (검증·소통·화면 · 원형) — 코드로 검사할 수 있는 판정 조건

- Critic 노드(graph_core.critic_node)와 역할 에이전트(플래너·구매 어드바이저)가 같은 기준을 쓰도록 모은 파일.
- 조건 ID·심각도는 「Critic 판정 조건표」(https://claude.ai/artifact/2GgFQeE1iTA9FvJANWRRL7)와 같다.
  조건을 바꿀 때는 조건표와 이 파일을 함께 고치고 팀에 공지한다.
- 입력은 dict(draft_proposal, context_data)이고, pydantic 모델을 넣어도 동작한다
  (플래너가 나중에 `from critic_rules import check_morning_briefing`으로 자체 검사를 바꿔 쓸 수 있게).
- LLM이 필요한 해석 검사(B1, B2, C2, PB2 등)는 여기 없고 graph_core.critic_node의 LLM 단계가 맡는다.

반환 형식: [{"rule": "A1_마감_누락", "severity": "critical", "message": "무엇이, 왜, 어떻게"}]
"""
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# 조건 ID → (violated_rules에 넣을 이름, 심각도)   ※ 조건표와 같은 값
# ---------------------------------------------------------------------
RULES: Dict[str, tuple] = {
    # 여행 일정 정리 (trip_planning)
    "T1": ("T1_여행기간_불일치", "critical"),
    "T2": ("T2_가고싶은곳_누락", "medium"),
    "T3": ("T3_입력에없는_장소", "high"),
    "T4": ("T4_도보많음날짜_불일치", "high"),
    "T5": ("T5_비예보날_야외일정", "medium"),
    "T6": ("T6_연속도보_휴식없음", "medium"),
    # 아침 브리핑 (morning_briefing)
    "A1": ("A1_마감_누락", "critical"),
    "A2": ("A2_마감후_배치", "critical"),
    "A3-1": ("A3-1_고정일정_확정변경", "critical"),
    "A3-2": ("A3-2_조정불가일정_이동제안", "high"),
    "A4": ("A4_시간_겹침", "high"),
    "A5": ("A5_연기규칙_위반", "critical"),
    "A6": ("A6_긴박업무_분량축소", "critical"),
    "B1": ("B1_없는_데이터", "high"),
    "B2": ("B2_데이터와_반대", "high"),
    "B3": ("B3_이유_누락", "medium"),
    "C1": ("C1_요청없는_연기", "high"),
    "C2": ("C2_무관한_기억", "high"),
    "C3": ("C3_확인안된_패턴", "medium"),
    "D1": ("D1_휴식_없음", "medium"),
    "D2": ("D2_근거없는_수용", "medium"),
    "D3": ("D3_이동시간_없음", "medium"),
    "D4": ("D4_활동시간_밖", "medium"),
    "D5": ("D5_지나치게_쪼갬", "low"),
    "D6": ("D6_거절제안_반복", "high"),
    # 구매 상담 (purchase_consultation)
    "PA1": ("PA1_여행예산_초과", "critical"),
    "PA2": ("PA2_충동신호_무시", "high"),
    "PA3": ("PA3_후회이유_무시", "high"),
    "PA4": ("PA4_중복_구매", "high"),
    "PA5": ("PA5_수량_과다과소", "high"),
    "PB1": ("PB1_검색결과에_없는_정보", "high"),
    "PB2": ("PB2_이력과_반대", "high"),
    "PB3": ("PB3_기준_근거없음", "medium"),
    "PB4": ("PB4_남은일정과_반대", "high"),
    "PC1": ("PC1_무관한_카테고리", "high"),
    "PC2": ("PC2_취향_지어냄", "high"),
    "PC3": ("PC3_취향_단정", "medium"),
    "PD1": ("PD1_재검토시점_없음", "medium"),
    "PD2": ("PD2_거절추천_반복", "high"),
    # 저녁 회고 (retrospective)
    "RA1": ("RA1_근거없는_해석저장", "critical"),
    "RA2": ("RA2_한번답으로_패턴", "high"),
    "RA3": ("RA3_완료율_불일치", "high"),
    "RB1": ("RB1_이력에없는_물건", "high"),
    "RB2": ("RB2_유도_질문", "high"),
    "RC1": ("RC1_질문_2개초과", "medium"),
    "RC2": ("RC2_회상시점_규칙밖", "medium"),
    "RD1": ("RD1_미룬질문_재질문", "high"),
}

# 조건표 밖의 '참고 메모'(low) — 통과시키되 화면·커뮤니케이터가 알려 줄 내용
NOTES: Dict[str, tuple] = {
    "PA5_NOTE": ("PA5_최소포장_남는양", "low"),      # 11/28 결정: 가장 작은 포장 1개는 PA5 예외, 남는 양만 안내
    "PA1_NOTE": ("PA1_예산비교_불가", "low"),        # 환산 금액이 없어 예산 비교를 못 함
    "PA5_UNKNOWN": ("PA5_도보량_확인불가", "low"),    # 남은 일정의 도보량이 전부 unknown
}

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# 판정 기준값 (구매 어드바이저 purchase_nodes.py의 값과 같아야 함)
HEAVY_WALKING = "많음"
LATE_NIGHT_HOUR = 23         # PA2: 23시 이후 상담
SHORT_SLEEP_HOURS = 6.0      # PA2: 최근 3일 연속 수면 6시간 미만
SHORT_SLEEP_DAYS = 3
MAX_OVERBUY_RATIO = 2        # PA5: 추천 매수가 필요량의 2배를 넘으면 과다
MIN_TRAVEL_GAP_MIN = 15      # D3: 장소가 다른 일정 사이 최소 이동 시간
RECALL_STAGES = ("first_run", "1_week", "1_month")
MAX_RECALL_QUESTIONS = 2     # RC1


# =====================================================================
# 공용 도우미
# =====================================================================
def _d(obj) -> Dict[str, Any]:
    """dict든 pydantic 모델이든 dict로."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def violation(rule_key: str, message: str) -> Dict[str, str]:
    name, severity = RULES.get(rule_key) or NOTES[rule_key]
    return {"rule": name, "severity": severity, "message": f"[{name.split('_')[0]}] {message}"}


def max_severity(violations: List[Dict[str, str]]) -> str:
    if not violations:
        return "low"
    return max((v["severity"] for v in violations), key=lambda s: SEVERITY_RANK.get(s, 0))


def to_messages(violations: List[Dict[str, str]]) -> List[str]:
    """플래너 자체 검사처럼 '[A1] ...' 문장 목록이 필요할 때."""
    return [v["message"] for v in violations]


def _min(t: str) -> int:
    try:
        h, m = map(int, str(t).strip().split(":")[:2])
        return h * 60 + m
    except Exception:
        return -1


def _active_window(context: dict) -> tuple:
    try:
        s, e = str(context.get("active_hours") or "08:00-23:00").split("-")
        return _min(s), _min(e)
    except ValueError:
        return 8 * 60, 23 * 60


def _date(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def detect_proposal_type(proposal: dict) -> str:
    """draft_proposal 모양으로 시나리오 판별 (조건표 「시나리오별 검사 방식」, graph_core.extract_changes와 같은 순서)."""
    p = _d(proposal)
    if "days" in p:
        return "trip_planning"
    if "changed_items" in p or "schedule_blocks" in p:
        return "morning_briefing"
    if "decision" in p and "item" in p:
        return "purchase_consultation"
    if "recall_stage" in p or "questions" in p:
        return "retrospective"
    return "unknown"


def _check_day_blocks(blocks: List[dict], window: tuple, label: str, check_travel: bool) -> List[dict]:
    """하루 블록의 A4(겹침)·D4(활동 시간 밖)·D3(이동 시간) 검사."""
    out = []
    blocks = sorted((_d(b) for b in blocks), key=lambda b: _min(b.get("start_time")))
    for i, b in enumerate(blocks):
        s, e = _min(b.get("start_time")), _min(b.get("end_time"))
        if s < window[0] or (e > window[1]):
            out.append(violation("D4", f"{label}'{b.get('title')}'({b.get('start_time')}~{b.get('end_time')})가 활동 시간 밖입니다. 활동 시간 안으로 옮기세요."))
        if i == 0:
            continue
        prev = blocks[i - 1]
        pe = _min(prev.get("end_time"))
        if s < pe:
            out.append(violation("A4", f"{label}'{prev.get('title')}'(~{prev.get('end_time')})와 '{b.get('title')}'({b.get('start_time')}~)의 시간이 겹칩니다. 둘 중 하나를 옮기세요."))
        elif (check_travel and prev.get("location") and b.get("location")
              and prev.get("location") != b.get("location") and (s - pe) < MIN_TRAVEL_GAP_MIN):
            out.append(violation("D3", f"{label}장소가 다른 '{prev.get('title')}'({prev.get('location')})와 '{b.get('title')}'({b.get('location')}) 사이에 이동 시간 {MIN_TRAVEL_GAP_MIN}분이 없습니다."))
    return out


# =====================================================================
# 1. 여행 일정 정리 (TripPlanProposal) — T1, T2, T4, T6 + 날짜별 A4·D4
# =====================================================================
def trip_dates(trip: dict) -> List[str]:
    s, e = _date((trip or {}).get("start_date")), _date((trip or {}).get("end_date"))
    if not s or not e or e < s:
        return []
    return [date.fromordinal(s.toordinal() + i).isoformat() for i in range((e - s).days + 1)]


def check_trip_plan(proposal: dict, context: dict) -> List[dict]:
    p, ctx = _d(proposal), context or {}
    trip = ctx.get("trip") or {}
    window = _active_window(ctx)
    days = [_d(d) for d in p.get("days") or []]
    out: List[dict] = []

    # T1 여행 기간과 날짜 일치
    expected = trip_dates(trip)
    actual = [str(d.get("date"))[:10] for d in days]
    if expected:
        missing = [d for d in expected if d not in actual]
        extra = [d for d in actual if d not in expected]
        if missing or extra:
            out.append(violation("T1", f"여행 기간({expected[0]}~{expected[-1]})과 일정 날짜가 다릅니다. 빠진 날={missing}, 기간 밖={extra}. 기간 안의 모든 날짜에 일정을 넣으세요."))
    for d in days:
        if not d.get("schedule_blocks"):
            out.append(violation("T1", f"{d.get('date')}의 일정이 비어 있습니다."))

    # T2 가고 싶은 곳 누락 시 이유
    places = ((trip.get("wishlist") or {}).get("places")) or []
    text = " ".join(f"{b.get('title','')} {b.get('location','')} {b.get('note','')}"
                    for d in days for b in (_d(x) for x in d.get("schedule_blocks") or []))
    warn_text = " ".join(p.get("unresolvable_warnings") or [])
    for place in places:
        if place and place not in text and place not in warn_text:
            out.append(violation("T2", f"가고 싶은 곳 '{place}'이(가) 일정에 없고, 빠진 이유도 unresolvable_warnings에 없습니다."))

    # T4 heavy_walking_dates == 도보 '많음' 날짜 (구매 수량 판단의 근거)
    # 날짜 전체의 walking_level 기준 (플래너가 heavy_walking_dates를 만드는 기준과 같음)
    heavy_actual = sorted(str(d.get("date"))[:10] for d in days if d.get("walking_level") == HEAVY_WALKING)
    heavy_listed = sorted(str(x)[:10] for x in p.get("heavy_walking_dates") or [])
    if heavy_actual != heavy_listed:
        out.append(violation("T4", f"heavy_walking_dates={heavy_listed}가 일정의 도보 '많음' 날짜 {heavy_actual}와 다릅니다. 구매 상담 수량 계산에 쓰이므로 맞추세요."))

    # T6 도보 '많음' 3일 연속이면 휴식 블록
    streak = 0
    for d in days:
        streak = streak + 1 if d.get("walking_level") == HEAVY_WALKING else 0
        if streak >= 3 and not any(_d(b).get("task_type") == "break" for b in d.get("schedule_blocks") or []):
            out.append(violation("T6", f"도보 '많음'이 3일째 이어지는 {d.get('date')}에 휴식 블록(task_type='break')이 없습니다."))

    # 날짜별 A4·D4 (여행 일정은 장소 간 이동 블록이 따로 있으므로 D3는 보지 않음)
    for d in days:
        out += _check_day_blocks(d.get("schedule_blocks") or [], window, f"{d.get('date')} ", check_travel=False)
    return out


# =====================================================================
# 2. 아침 브리핑 (PlannerProposal) — A1~A6, B3, C1, D1, D3, D4
#    (planner_nodes._validate_plan과 같은 기준을 dict 입력으로 옮김)
# =====================================================================
def check_morning_briefing(proposal: dict, context: dict, user_requested_rest: Optional[bool] = None) -> List[dict]:
    p, ctx = _d(proposal), context or {}
    blocks = [_d(b) for b in p.get("schedule_blocks") or []]
    changed = {str(_d(c).get("id")): _d(c) for c in p.get("changed_items") or []}
    postponed = {str(_d(x).get("id")): _d(x) for x in p.get("postponed_tasks") or []}
    by_id = {str(b.get("id")): b for b in blocks}
    rest = p.get("user_requested_rest") if user_requested_rest is None else user_requested_rest
    today = _date(ctx.get("today") or p.get("date")) or date.today()
    out: List[dict] = []

    # A3-1 / A3-2 / E3 고정 일정
    for ev in ctx.get("calendar_events") or []:
        if not ev.get("is_fixed", True):
            continue
        eid, title = str(ev.get("id")), ev.get("title", ev.get("id"))
        b = by_id.get(eid)
        if not b:
            out.append(violation("A3-1", f"고정 일정 '{title}'(id={eid})이 일정에서 빠졌습니다. 고정 일정은 그대로 두세요."))
            continue
        moved = b.get("start_time") != ev.get("start_time") or b.get("end_time") != ev.get("end_time")
        if not ev.get("is_adjustable", False):
            if moved and b.get("status") != "proposed":
                out.append(violation("A3-1", f"조정할 수 없는 고정 일정 '{title}'({ev.get('start_time')}~{ev.get('end_time')})의 시간을 바꿨습니다. 원래 시간으로 되돌리고, 겹치면 unresolvable_warnings에만 적으세요."))
            elif moved or b.get("status") == "proposed" or eid in changed:
                out.append(violation("A3-2", f"조정할 수 없는 고정 일정 '{title}'의 이동을 제안했습니다. 이동 제안 대신 충돌 알림만 하세요."))
        elif moved and b.get("status") != "proposed":
            out.append(violation("A3-1", f"조정 가능한 고정 일정 '{title}'을 확정으로 옮겼습니다. status='proposed'로 이동 '제안'만 하세요(E3)."))

    # A1·A2·A5·A6·C1 할 일
    for t in ctx.get("tasks") or []:
        tid, title = str(t.get("id")), t.get("title", t.get("id"))
        dl = str(t.get("deadline") or "")
        days_left, dl_min = 99, 23 * 60 + 59
        if dl:
            dd = _date(dl)
            if dd:
                days_left = (dd - today).days
            if " " in dl:
                dl_min = _min(dl.split(" ")[1])
        urgent = days_left <= 2
        b, pp = by_id.get(tid), postponed.get(tid)
        if pp and not rest:
            out.append(violation("C1", f"쉬겠다는 요청이 없는데 '{title}'을(를) 다른 날로 미뤘습니다. 오늘 안에서 시간대만 바꾸세요."))
        if pp and (urgent or t.get("is_collaborative")):
            out.append(violation("A5", f"'{title}'(마감 D-{days_left}{', 협업' if t.get('is_collaborative') else ''})은 미룰 수 없습니다. 오늘 일정에 넣으세요."))
        if days_left == 0 and not b and not pp:
            out.append(violation("A1", f"오늘 마감({dl})인 '{title}'이 일정에서 빠졌습니다. 마감 전에 배치하세요."))
        if b:
            if days_left == 0 and _min(b.get("end_time")) > dl_min:
                out.append(violation("A2", f"오늘 마감 '{title}'이 마감 시각({dl}) 이후({b.get('end_time')})에 끝나도록 배치됐습니다."))
            orig = int(t.get("estimated_minutes") or t.get("duration_minutes") or 0)
            reduced = (b.get("status") == "reduced" or (orig and int(b.get("duration_minutes") or orig) < orig)
                       or changed.get(tid, {}).get("change_type") == "reduced")
            if urgent and reduced and not rest:
                out.append(violation("A6", f"'조절' 요청에서 긴박한 '{title}'(마감 D-{days_left})의 분량을 줄였습니다. 원래 분량을 유지하고 시간대만 옮기세요."))

    # B3 변경 이유
    for c in changed.values():
        if len(str(c.get("reason") or "").strip()) < 5:
            out.append(violation("B3", f"변경한 '{c.get('title')}'에 구체적인 이유가 없습니다."))

    # A4·D3·D4
    out += _check_day_blocks(blocks, _active_window(ctx), "", check_travel=True)

    # D1 4시간 이상 활동인데 휴식 없음
    work = sum(max(_min(b.get("end_time")) - _min(b.get("start_time")), 0) for b in blocks if b.get("task_type") != "break")
    if work >= 240 and blocks and not any(b.get("task_type") == "break" for b in blocks):
        out.append(violation("D1", "일정이 4시간 이상인데 식사·휴식 블록이 없습니다."))
    return out


# =====================================================================
# 3. 구매 상담 (PurchaseProposal) — PA1, PA2, PA5, PB1(추천 제품), PB4(도보 날짜), PD1
# =====================================================================
def remaining_heavy_dates(trip_schedule: List[dict], today_str: str) -> dict:
    """오늘(오늘 밤 포함) 이후 일정에서 도보 '많음'인 날. purchase_nodes.remaining_schedule_summary와 같은 기준."""
    today = _date(today_str) or date.today()
    remaining, heavy, unknown = [], [], []
    for day in trip_schedule or []:
        d = _date(day.get("date"))
        if d is None or d < today:
            continue
        remaining.append(d.isoformat())
        levels = [str(e.get("walking_level") or "").strip() for e in day.get("events") or []]
        if HEAVY_WALKING in levels:
            heavy.append(d.isoformat())
        elif levels and all(lv in ("", "unknown") for lv in levels):
            unknown.append(d.isoformat())
    return {"remaining": remaining, "heavy": heavy, "unknown": unknown}


def impulse_signals(context: dict, now: Optional[datetime] = None) -> List[str]:
    """PA2 충동구매 신호: 23시 이후 상담 / 최근 3일 연속 수면 6시간 미만."""
    ctx = context or {}
    signals = []
    hour = None
    if ctx.get("now"):   # 시연·테스트용 "HH:MM"
        try:
            hour = int(str(ctx["now"]).split(":")[0])
        except ValueError:
            hour = None
    if hour is None:
        hour = (now or datetime.now()).hour
    if hour >= LATE_NIGHT_HOUR:
        signals.append(f"{LATE_NIGHT_HOUR}시 이후 상담")
    recent = sorted((r for r in (ctx.get("biometrics") or {}).get("recent") or [] if r.get("date")),
                    key=lambda r: r["date"])[-SHORT_SLEEP_DAYS:]
    hours = [0.0 if r.get("sleep_type") == "all_nighter" else r.get("sleep_hours") for r in recent]
    if len(hours) == SHORT_SLEEP_DAYS and all(h is not None and h < SHORT_SLEEP_HOURS for h in hours):
        signals.append(f"최근 {SHORT_SLEEP_DAYS}일 연속 수면 {SHORT_SLEEP_HOURS:g}시간 미만")
    return signals


def _find_product(products: List[dict], name: str) -> Optional[dict]:
    for p in products or []:
        if str(p.get("name")) == str(name):
            return p
    return None


def check_purchase(proposal: dict, context: dict, now: Optional[datetime] = None) -> List[dict]:
    p, ctx = _d(proposal), context or {}
    trip = ctx.get("trip") or {}
    spending = ctx.get("spending") or {}
    products = ctx.get("products") or []
    rec = _d(p.get("recommended")) if p.get("recommended") else None
    recommend = p.get("decision") == "recommend" and rec is not None
    out: List[dict] = []

    # PD1 보류인데 재검토 시점·조건 없음
    if p.get("decision") == "hold" and not (p.get("revisit_at") or p.get("revisit_condition")):
        out.append(violation("PD1", "보류했는데 언제·어떤 조건에서 다시 볼지(revisit_at)가 없습니다."))
    if not recommend:
        return out

    # PB1 추천 제품·가격이 검색 결과(products)에 있는지
    prod = _find_product(products, rec.get("name"))
    if products and prod is None:
        out.append(violation("PB1", f"추천한 '{rec.get('name')}'이(가) 상품 목록(products)에 없습니다. 목록에 있는 상품만 추천하세요."))
    elif prod is not None and prod.get("price") is not None and rec.get("price") != prod.get("price"):
        out.append(violation("PB1", f"'{rec.get('name')}' 가격이 {rec.get('price')}로 적혀 있지만 상품 목록에는 {prod.get('price')}입니다."))

    # PA1 여행 예산 (11/28 결정: 구매 에이전트가 넣은 total_in_trip_currency로 비교)
    budget = trip.get("budget")
    if budget:
        remaining = budget - (spending.get("trip_spent") or 0)
        trip_cur = (trip.get("currency") or "").upper()
        amount = rec.get("total_in_trip_currency")
        if amount is None and (rec.get("currency") or "").upper() in ("", trip_cur):
            amount = (rec.get("price") or 0) * (rec.get("quantity") or 1)
        if amount is None:
            out.append(violation("PA1_NOTE", f"추천 금액을 여행 예산 통화({trip_cur})로 바꾼 값이 없어 예산 비교를 하지 못했습니다."))
        elif amount > remaining:
            out.append(violation("PA1", f"추천 금액 {amount:,.0f} {trip_cur}이(가) 남은 여행 예산 {remaining:,.0f} {trip_cur}을(를) 넘습니다. 예산 안의 상품을 고르거나 보류하세요."))

    # PA2 충동 신호가 있는데 바로 구매 추천 (긴급 필요 품목은 제외)
    urgent = bool((p.get("intent") or {}).get("is_urgent_need"))
    signals = [] if urgent else impulse_signals(ctx, now)
    if signals:
        out.append(violation("PA2", f"충동구매 신호({', '.join(signals)})가 있는데 보류 없이 바로 구매를 권했습니다. 보류하고 재검토 시점을 주세요."))

    # PA5 수량 / PB4 남은 일정 (많이 걷는 날마다 쓰는 품목만)
    qd = p.get("quantity_detail") or {}
    driver = qd.get("usage_driver") or (p.get("intent") or {}).get("usage_driver")
    if driver == "walking_days":
        sched = remaining_heavy_dates(ctx.get("trip_schedule") or [], ctx.get("today") or "")
        listed = qd.get("heavy_walking_dates")
        if listed is not None and sorted(listed) != sorted(sched["heavy"]):
            out.append(violation("PB4", f"수량 근거의 도보 '많음' 날짜 {sorted(listed)}가 실제 남은 일정 {sched['heavy']}와 다릅니다."))
        if not sched["heavy"] and sched["unknown"]:
            out.append(violation("PA5_UNKNOWN", f"남은 일정({', '.join(sched['unknown'])})의 도보량을 알 수 없어 수량을 확인하지 못했습니다."))
        else:
            daily = qd.get("daily_usage")
            unit_count = rec.get("unit_count") or (prod or {}).get("unit_count") or 1
            quantity = rec.get("quantity") or 1
            total = unit_count * quantity
            if not daily:
                out.append(violation("PB3", "하루 사용량 기준이 없어 수량 근거를 확인할 수 없습니다. quantity_basis에 하루 몇 매인지 적으세요."))
            else:
                needed = len(sched["heavy"]) * int(daily)
                if needed == 0:
                    out.append(violation("PB4", "남은 일정에 도보 '많음'인 날이 없는데 '많이 걸으니 필요하다'는 이유로 구매를 권했습니다."))
                elif total < needed:
                    out.append(violation("PA5", f"추천 수량 {total}매가 필요량 {needed}매(도보 '많음' {len(sched['heavy'])}일 × 하루 {daily}매)보다 적습니다."))
                elif total > needed * MAX_OVERBUY_RATIO:
                    unit_counts = [c.get("unit_count") or 1 for c in (p.get("candidates") or products)]
                    smallest = min(unit_counts) if unit_counts else unit_count
                    if quantity == 1 and unit_count <= smallest:
                        # 11/28 결정: 가장 작은 포장 1개는 예외 (더 줄일 방법이 없음) → 남는 양만 안내
                        out.append(violation("PA5_NOTE", f"가장 작은 포장({unit_count}매) 1개라 필요량 {needed}매보다 {total - needed}매가 남습니다."))
                    else:
                        out.append(violation("PA5", f"추천 수량 {total}매({unit_count}매 × {quantity}개)가 필요량 {needed}매의 {MAX_OVERBUY_RATIO}배를 넘습니다. 필요량을 채우는 가장 적은 수량으로 줄이세요."))
    return out


# =====================================================================
# 4. 저녁 회고 (RecallResult) — RC1, RC2
# =====================================================================
def check_recall(proposal: dict, context: dict) -> List[dict]:
    p = _d(proposal)
    out: List[dict] = []
    qs = p.get("questions") or []
    if len(qs) > MAX_RECALL_QUESTIONS:
        out.append(violation("RC1", f"질문이 {len(qs)}개입니다. 한 번에 {MAX_RECALL_QUESTIONS}개 이하로 줄이세요."))
    if p.get("recall_stage") not in RECALL_STAGES:
        out.append(violation("RC2", f"회상 시점 '{p.get('recall_stage')}'은(는) 정해진 시점({', '.join(RECALL_STAGES)})이 아닙니다."))
    return out


# =====================================================================
# 시나리오별 코드 검사 묶음
# =====================================================================
CHECKS = {
    "trip_planning": check_trip_plan,
    "morning_briefing": check_morning_briefing,
    "purchase_consultation": check_purchase,
    "retrospective": check_recall,
}


def run_code_checks(proposal: dict, context: dict, scenario: Optional[str] = None) -> List[dict]:
    """draft_proposal 모양(또는 request_type)으로 시나리오를 골라 코드 검사를 돌린다."""
    scenario = scenario if scenario in CHECKS else detect_proposal_type(proposal)
    fn = CHECKS.get(scenario)
    return fn(proposal, context) if fn else []
