"""Critic 검사 + 커뮤니케이터 문구 (검증·소통·화면 · 원형)

graph_core.critic_node / communicator_node가 이 파일의 함수를 부른다.
판정(pass / rework / warn / keep_original)은 graph_core.decide_critic_verdict가 정하고,
여기서는 '무엇이 걸렸는지(violated_rules)·얼마나 심각한지(severity)·어떻게 고칠지(feedback)'만 만든다.

Critic 흐름
  ① 코드 검사 (critic_rules.py)  — 숫자·시간·ID로 확인할 수 있는 조건. 항상 실행
  ② LLM 해석 검사 (llm_large)     — 근거·개인화처럼 문장을 읽어야 하는 조건.
                                    ①에서 이미 high 이상이면 어차피 반려이므로 생략(시연 시간 절약)
                                    .env CRITIC_LLM=off 이면 생략 / 호출이 실패해도 ①만으로 판정
  ③ 결과 정리                      — 심각도는 LLM이 아니라 조건표 값(critic_rules.RULES)으로 고정
"""
import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

import critic_rules as cr
from state import get_last_user_message

# ---------------------------------------------------------------------
# ② LLM 해석 검사 — 시나리오별로 LLM이 볼 조건 (조건표 「시나리오별 검사 방식」)
# ---------------------------------------------------------------------
LLM_RULES: Dict[str, Dict[str, str]] = {
    "trip_planning": {
        "T3": "입력(trip.wishlist)에 없는 장소를 '사용자가 원한 곳'이라고 하거나, 근거 없이 새 장소를 넣음",
        "T5": "비 예보(weather)가 있는 날에 야외·도보 '많음' 일정을 이유(note) 없이 넣음",
        "B1": "입력 데이터에 없는 날씨·장소·사실을 근거로 씀",
    },
    "morning_briefing": {
        "B1": "입력 데이터(weather, biometrics, calendar_events, tasks)에 없는 날씨·수면·장소·사실을 근거로 씀",
        "B2": "변경 이유가 입력 데이터와 반대 (예: 맑음인데 '비가 와서')",
        "C2": "오늘 일정과 관련 없는 기억·이력(예: 과거 구매)을 근거로 씀",
        "C3": "한두 번의 기록으로 '항상 ~한다' 같은 습관을 단정함",
        "D2": "사용자 요청을 근거 없이 그대로 받아들여 무리한 일정을 만듦",
        "D6": "rejected_history에서 사용자가 거절·수정한 제안과 같은 변경을 다시 냄",
    },
    "purchase_consultation": {
        "PA3": "같은 카테고리에서 사용자가 후회한 이유(spending.records, rejected_history)를 무시하고 추천함",
        "PA4": "최근 같은 기능의 제품을 샀는데 언급 없이 추천함",
        "PB1": "근거(evidence)·후보 메모에 products 목록에 없는 가격·매수·스펙·리뷰가 있음",
        "PB2": "추천 이유가 구매 이력과 반대",
        "PB3": "구매 기준(criteria)에 어떤 데이터에서 왔는지 근거가 없음",
        "PB4": "추천·보류 이유가 남은 일정(trip_schedule)과 반대 (예: 내일 도보 '적음'인데 '내일 많이 걸으니 필요')",
        "PC1": "관련 없는 카테고리의 구매 이력을 근거로 씀",
        "PC2": "이력이 없는 카테고리인데 취향을 지어냄",
        "PC3": "1~2건의 이력으로 '항상 ~를 선호' 같은 취향을 단정함",
        "PD2": "rejected_history에서 같은 카테고리이고 구매 이유·상황(purchase_context)이 비슷한데 거절한 추천을 다시 냄. 이유가 다르면 반복이 아님",
    },
    "retrospective": {
        "RA1": "사용자 답을 근거 없이 해석해 저장하려 함",
        "RA2": "한 번의 답으로 패턴을 저장하려 함",
        "RB1": "구매 이력에 없는 물건을 물어봄",
        "RB2": "답을 유도하는 질문 (예: '별로였죠?')",
        "RD1": "사용자가 '나중에'라고 넘긴 질문을 다시 물음",
    },
}

# LLM에게 보여 줄 맥락 (시나리오에 필요한 것만 — 프롬프트 길이·시간 절약)
LLM_CONTEXT_KEYS = {
    "trip_planning": ("trip", "weather"),
    "morning_briefing": ("today", "calendar_events", "tasks", "biometrics", "weather"),
    "purchase_consultation": ("today", "trip", "trip_schedule", "spending", "products", "biometrics"),
    "retrospective": ("today", "spending"),
}


class LLMFinding(BaseModel):
    rule_id: str = Field(description="걸린 조건 ID (제공된 목록에 있는 ID만)")
    reason: str = Field(description="무엇이 왜 문제인지 한 문장. 입력 데이터의 어느 값과 어긋나는지 포함")
    fix: str = Field(description="담당 에이전트가 어떻게 고치면 되는지 한 문장")


class CriticLLMOutput(BaseModel):
    findings: List[LLMFinding] = Field(default_factory=list, description="확실한 위반만. 없으면 빈 목록")


CRITIC_SYSTEM = """당신은 개인 비서 시스템의 검증 에이전트(Critic)입니다.
역할 에이전트가 만든 제안(draft_proposal)을 입력 데이터(context)와 비교해, 아래 조건에 '확실히' 걸리는 것만 찾으세요.

[원칙]
- 제공된 조건 ID 목록에 있는 조건만 검사합니다. 목록 밖의 문제는 적지 않습니다.
- 입력 데이터에서 근거를 찾을 수 있을 때만 위반으로 적습니다. 애매하면 적지 않습니다 (거짓 반려가 더 해롭습니다).
- 숫자·시간·수량·예산은 이미 코드가 검사했으므로 다시 따지지 않습니다.
- 말투나 문장 길이는 검사하지 않습니다 (커뮤니케이터가 다듬습니다).
"""


def _llm_check(scenario: str, state: Dict[str, Any]) -> List[dict]:
    rules = LLM_RULES.get(scenario)
    if not rules:
        return []
    from config import llm_large   # 테스트에서 Gemini 없이 불러올 수 있도록 함수 안에서 import

    ctx = state.get("context_data") or {}
    small_ctx = {k: ctx.get(k) for k in LLM_CONTEXT_KEYS.get(scenario, ()) if k in ctx}
    proposal = dict(state.get("draft_proposal") or {})
    proposal.pop("intent", None)
    human = (
        f"[검사할 조건]\n" + "\n".join(f"- {k}: {v}" for k, v in rules.items())
        + f"\n\n[사용자 요청]\n{get_last_user_message(state)}"
        + f"\n\n[입력 데이터(context)]\n{json.dumps(small_ctx, ensure_ascii=False, default=str)}"
        + f"\n\n[거절·수정 이력(rejected_history)]\n{json.dumps(state.get('rejected_history') or [], ensure_ascii=False, default=str)}"
        + f"\n\n[검사할 제안(draft_proposal)]\n{json.dumps(proposal, ensure_ascii=False, default=str)}"
    )
    out: CriticLLMOutput = llm_large.with_structured_output(CriticLLMOutput).invoke(
        [("system", CRITIC_SYSTEM), ("human", human)])
    found = []
    for f in out.findings:
        key = f.rule_id.strip().split("_")[0].upper()
        if key in rules:   # 목록 밖 ID는 버림 (심각도는 조건표 값으로 고정)
            found.append(cr.violation(key, f"{f.reason} → {f.fix}".strip(" →")))
    return found


def _feedback(violations: List[dict]) -> str:
    if not violations:
        return "조건표의 모든 검사를 통과했습니다."
    ordered = sorted(violations, key=lambda v: -cr.SEVERITY_RANK.get(v["severity"], 0))
    blocking = [v for v in ordered if cr.SEVERITY_RANK[v["severity"]] >= cr.SEVERITY_RANK["high"]]
    head = "다음 문제를 고쳐 다시 제안하세요:" if blocking else "통과입니다. 다만 다음 사항을 참고하세요:"
    return head + "\n" + "\n".join(f"- ({v['severity']}) {v['message']}" for v in ordered)


def run_critic(state: Dict[str, Any]) -> Dict[str, Any]:
    """critic_result의 violated_rules·severity·feedback(+화면용 details)을 만든다. target_agent는 graph_core가 채운다."""
    proposal = state.get("draft_proposal") or {}
    ctx = state.get("context_data") or {}
    scenario = state.get("request_type")
    if scenario not in cr.CHECKS:
        scenario = cr.detect_proposal_type(proposal)

    # ① 코드 검사
    violations = cr.run_code_checks(proposal, ctx, scenario)
    print(f" -> [코드 검사] {scenario}: {len(violations)}건 {[v['rule'] for v in violations]}")

    # ② LLM 해석 검사
    llm_status = "skipped"
    if os.getenv("CRITIC_LLM", "on").lower() == "off":
        llm_status = "off"
    elif cr.SEVERITY_RANK[cr.max_severity(violations)] >= cr.SEVERITY_RANK["high"]:
        llm_status = "skipped (코드 검사에서 이미 반려)"
    else:
        try:
            found = _llm_check(scenario, state)
            violations += found
            llm_status = "checked"
            print(f" -> [LLM 해석 검사] {len(found)}건 {[v['rule'] for v in found]}")
        except Exception as e:  # noqa: BLE001 — 시연 중 API 오류로 그래프가 멈추지 않도록
            llm_status = f"failed ({type(e).__name__})"
            print(f" -> [LLM 해석 검사 실패 → 코드 검사만으로 판정] {type(e).__name__}: {str(e)[:120]}")

    # ③ 정리 (같은 조건·같은 문장 중복 제거)
    seen, unique = set(), []
    for v in violations:
        key = (v["rule"], v["message"])
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return {
        "violated_rules": sorted({v["rule"] for v in unique}),
        "severity": cr.max_severity(unique),
        "feedback": _feedback(unique),
        "details": unique,            # 화면 '자세히 보기'용: [{rule, severity, message}]
        "scenario": scenario,
        "llm_check": llm_status,
    }


# =====================================================================
# 커뮤니케이터 문구 — 결론 → 이유 → 대안 (LLM 없이 draft_proposal로 만든다: 빠르고, 없는 내용을 지어내지 않음)
# =====================================================================
def _money(amount, currency) -> str:
    if amount is None:
        return "-"
    cur = (currency or "").upper()
    unit = {"KRW": "원", "JPY": "엔", "USD": "달러"}.get(cur, f" {cur}")
    return f"{amount:,.0f}{unit}"


def _clean(msg: str) -> str:
    """'[PA5] 가장 작은…' → '가장 작은…' (사용자에게는 조건 ID를 보이지 않음)"""
    return msg.split("] ", 1)[1] if msg.startswith("[") and "] " in msg else msg


def _issues(state, min_sev="low", max_sev="critical") -> List[str]:
    lo, hi = cr.SEVERITY_RANK[min_sev], cr.SEVERITY_RANK[max_sev]
    details = (state.get("critic_result") or {}).get("details") or []
    return [_clean(v["message"]) for v in details if lo <= cr.SEVERITY_RANK.get(v["severity"], 0) <= hi]


def _trip_text(p: dict, ctx: dict) -> str:
    days = p.get("days") or []
    lines = [f"결론: {p.get('destination') or (ctx.get('trip') or {}).get('destination') or '여행'} {len(days)}일 일정을 짰어요."]
    for d in days:
        titles = [b.get("title") for b in d.get("schedule_blocks") or [] if b.get("task_type") != "break"]
        label = d.get("day_label") or d.get("date")
        lines.append(f"- {label} ({d.get('date')}, 도보 {d.get('walking_level') or '-'}): {' → '.join(t for t in titles if t)}")
    if p.get("heavy_walking_dates"):
        lines.append(f"이유: 많이 걷는 날은 {', '.join(p['heavy_walking_dates'])}이에요. 이 날짜는 구매 상담 때 파스 수량 계산에 쓰여요.")
    for w in p.get("unresolvable_warnings") or []:
        lines.append(f"넣지 못한 곳: {w}")
    lines.append("확정하시면 캘린더에 날짜별로 기록할게요.")
    return "\n".join(lines)


def _morning_text(p: dict, ctx: dict) -> str:
    lines = ["결론: 오늘 일정 브리핑이에요."]
    if p.get("briefing_note"):
        lines.append(p["briefing_note"])
    walk = {str(e.get("id")): e.get("walking_level") for e in ctx.get("calendar_events") or []}
    for b in sorted(p.get("schedule_blocks") or [], key=lambda b: cr._min(b.get("start_time"))):
        tag = " (이동 제안)" if b.get("status") == "proposed" else ""
        lv = walk.get(str(b.get("id"))) or b.get("walking_level")
        lines.append(f"- {b.get('start_time')}–{b.get('end_time')} {b.get('title')}" + (f" · 도보 {lv}" if lv else "") + tag)
    for c in p.get("changed_items") or []:
        lines.append(f"바뀐 점: {c.get('title')} {c.get('before')} → {c.get('after')} (이유: {c.get('reason')})")
    for t in p.get("postponed_tasks") or []:
        lines.append(f"미룬 일: {t.get('title')} → {t.get('new_scheduled_date')} (이유: {t.get('reason')})")
    for w in p.get("unresolvable_warnings") or []:
        lines.append(f"충돌 알림: {w}")
    lines.append("이대로 캘린더에 반영할까요?" if (p.get("changed_items") or p.get("postponed_tasks")) else "바꿀 부분이 있으면 말씀해 주세요.")
    return "\n".join(lines)


def _purchase_text(p: dict, ctx: dict) -> str:
    rec = p.get("recommended")
    trip = ctx.get("trip") or {}
    bc = p.get("budget_check") or {}
    if p.get("decision") == "recommend" and rec:
        price = _money(rec.get("price"), rec.get("currency"))
        total = rec.get("total_in_trip_currency")
        conv = f", 약 {_money(total, trip.get('currency'))}" if total is not None and (rec.get("currency") or "").upper() != (trip.get("currency") or "").upper() else ""
        lines = [f"결론: {rec.get('name')} {rec.get('quantity', 1)}개를 추천해요 ({price}{conv})."]
        if p.get("quantity_basis"):
            lines.append(f"이유: {p['quantity_basis']}")
        if bc.get("remaining") is not None:
            lines.append(f"예산: 남은 여행 예산 {_money(bc['remaining'], bc.get('currency'))} 안에서 살 수 있어요.")
        others = [c for c in p.get("candidates") or [] if c.get("name") != rec.get("name")]
        for c in others:
            why = c.get("note") or ("필요량보다 너무 많아요" if c.get("overbuy") else "")
            lines.append(f"대안: {c.get('name')} ({_money(c.get('price'), c.get('currency'))}, {c.get('unit_count')}매){' — ' + why if why else ''}")
        lines.append("이대로 살까요?")
        return "\n".join(lines)
    lines = ["결론: 지금은 사지 않는 걸 권해요 (보류)."]
    for r in p.get("hold_reasons") or []:
        lines.append(f"이유: {r}")
    when = p.get("revisit_condition") or ""
    if p.get("revisit_at"):
        when = f"{when} ({p['revisit_at']})" if when else f"{p['revisit_at']}에 다시 확인"
    if when:
        lines.append(f"다시 볼 때: {when}")
    return "\n".join(lines)


def _recall_text(p: dict, ctx: dict) -> str:
    qs = p.get("questions") or []
    return "\n".join(["오늘 하루를 돌아볼게요."] + [f"- {q}" for q in qs] + ["편할 때 답해 주시고, 넘기셔도 괜찮아요."])


BODY = {
    "trip_planning": _trip_text,
    "morning_briefing": _morning_text,
    "purchase_consultation": _purchase_text,
    "retrospective": _recall_text,
}

AFTER_REJECTION = {
    "trip_planning": "알겠어요. 여행 일정은 확정하지 않았어요. 바꾸고 싶은 날짜나 장소를 말씀해 주시면 다시 짤게요.",
    "morning_briefing": "알겠습니다. 오늘은 원래 일정을 그대로 유지할게요. 바꾸고 싶은 부분이 있으면 말씀해 주세요.",
    "purchase_consultation": "알겠어요. 이번 추천은 넣어 둘게요. 거절하신 이유는 다음 상담에 참고하고, 다른 상품이나 조건이 있으면 말씀해 주세요.",
    "retrospective": "알겠어요. 이번 답변은 저장하지 않을게요.",
}

# keep_original: 재작업 2회 후에도 critical이 남음 → 제안을 내보내지 않음 (조건표 「심각도별 처리」)
KEEP_ORIGINAL = {
    "trip_planning": "검토에서 해결되지 않은 문제가 있어 여행 일정을 확정하지 않았어요. 여행 날짜와 가고 싶은 곳을 한 번 더 확인해 주세요.",
    "morning_briefing": "검토 결과 마감·고정 일정을 지킬 수 있는 안전한 재배치안을 찾지 못해 원래 일정을 유지합니다.",
    "purchase_consultation": "검토 결과 지금은 구매를 권하지 않아요 (보류 유지).",
    "retrospective": "이번 답변은 저장하지 않고 다음 회고 때 다시 여쭤볼게요.",   # 튜토리얼 후 최종 결정
}


def build_message(state: Dict[str, Any], after_rejection: bool) -> str:
    req = state.get("request_type")
    verdict = state.get("critic_verdict")
    if req == "general":
        return "안녕하세요! 여행 일정 정리, 아침 브리핑, 구매 상담, 저녁 회고를 도와드릴 수 있어요."
    scenario = req if req in BODY else cr.detect_proposal_type(state.get("draft_proposal") or {})
    if after_rejection:
        return AFTER_REJECTION.get(scenario, AFTER_REJECTION["morning_briefing"])
    if verdict == "keep_original":
        reasons = _issues(state, "critical")
        return KEEP_ORIGINAL.get(scenario, KEEP_ORIGINAL["morning_briefing"]) + (
            "\n이유: " + " / ".join(reasons[:2]) if reasons else "")

    p = state.get("draft_proposal") or {}
    ctx = state.get("context_data") or {}
    try:
        body = BODY[scenario](p, ctx) if scenario in BODY else "제안을 정리했어요."
    except Exception as e:  # noqa: BLE001 — 형식이 조금 달라도 응답은 나가게
        print(f" -> [커뮤니케이터 경고] 문구 생성 실패({type(e).__name__}) → 기본 문구")
        body = "제안을 정리했어요. 자세한 내용은 아래를 확인해 주세요."

    notes = _issues(state, "low", "medium")
    if notes:
        body += "\n" + "\n".join(f"참고: {n}" for n in notes)
    if verdict == "warn":
        problems = _issues(state, "high")
        return "⚠️ [확인 필요] 검토에서 해결되지 않은 문제가 있어요: " + " / ".join(problems[:2]) + "\n\n" + body
    return body
