import os
import json
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# =====================================================================
# 0. LLM 초기화 헬퍼
# =====================================================================
def _get_llm(model_type: str = "large") -> ChatGoogleGenerativeAI:
    if model_type == "small":
        model_name = os.getenv("GEMINI_MODEL_SMALL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    model_name = os.getenv("GEMINI_MODEL_LARGE", "gemini-2.5-pro")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.2)


# =====================================================================
# 1. Structured Output 스키마 정의 (Critic 연동용 ID 및 규칙 필드 반영)
# =====================================================================
class ConditionAnalysisOutput(BaseModel):
    morning_score: int = Field(description="오전(08~12시) 에너지 점수 (0~100)")
    afternoon_score: int = Field(description="오후(12~18시) 에너지 점수 (0~100)")
    evening_score: int = Field(description="저녁(18~23시) 에너지 점수 (0~100)")
    summary: str = Field(description="입력된 실제 데이터만을 근거로 한 컨디션 한 줄 요약")


class ScheduleBlock(BaseModel):
    id: str = Field(description="원본 캘린더 이벤트 또는 할 일의 고유 ID (휴식/식사 등 신규 블록은 'break_1' 등 부여)")
    start_time: str = Field(description="시작 시각 (HH:MM)")
    end_time: str = Field(description="종료 시각 (HH:MM)")
    duration_minutes: int = Field(description="배치된 소요 시간(분)")
    title: str = Field(description="일정 또는 작업 제목")
    task_type: Literal["fixed", "important", "flexible", "break"] = Field(description="작업 분류")
    status: Literal["kept", "moved", "reduced", "proposed", "added"] = Field(description="상태 (고정 일정 이동 제안은 반드시 'proposed')")
    location: str = Field(default="", description="장소")


class ChangedItem(BaseModel):
    id: str = Field(description="원본 캘린더 이벤트 또는 할 일의 고유 ID")
    title: str = Field(description="변경된 항목 제목")
    change_type: Literal["moved", "reduced", "proposed"] = Field(description="변경 유형")
    before: str = Field(description="기존 시간대 또는 기존 분량")
    after: str = Field(description="변경된 시간대 또는 조정된 분량(어디까지 할지 구체적 명시)")
    reason: str = Field(description="실제 데이터(에너지 점수, 마감일 등)에 기반한 변경 이유 (없는 날씨/과거 구매기록 언급 절대 금지)")


class PostponedTask(BaseModel):
    id: str = Field(description="원본 할 일의 고유 ID")
    title: str = Field(description="연기된 작업 제목")
    original_deadline: str = Field(description="원래 마감일 (YYYY-MM-DD HH:MM)")
    new_scheduled_date: str = Field(description="새로 배치할 날짜 (YYYY-MM-DD)")
    reason: str = Field(description="연기 사유 (사용자 명시적 휴식 요청 기반)")


class PlannerDraftOutput(BaseModel):
    schedule_blocks: List[ScheduleBlock] = Field(description="하루 전체 일정 블록 (시작 시각 오름차순)")
    changed_items: List[ChangedItem] = Field(description="기존 대비 변경되거나 이동 제안된 항목 목록")
    postponed_tasks: List[PostponedTask] = Field(description="다음 날 이후로 연기된 작업 목록")
    unresolvable_warnings: List[str] = Field(
        default_factory=list,
        description="조정 불가 고정 일정과 마감 작업이 물리적으로 겹쳐 해결할 수 없는 경우의 충돌 알림 (예: '15시 회의 때문에 마감 작업 시간이 부족합니다. 확인이 필요합니다.')"
    )


# =====================================================================
# 2. Node 1: 컨디션 분석 노드 (condition_analyzer_node)
# =====================================================================
def condition_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[컨디션 분석] 실제 생체·환경 데이터 기반 시간대별 에너지 점수 산출 중...")
    context_data = state.get("context_data", {})
    bio_data = context_data.get("biometrics", context_data.get("health", {}))
    env_data = context_data.get("environment", context_data.get("weather", {}))

    if not bio_data and not env_data:
        scores = {"morning": 70, "afternoon": 60, "evening": 50, "summary": "생체/환경 데이터 미제공으로 기본 에너지 기준 적용"}
        return {"condition_score": scores}

    llm = _get_llm("small").with_structured_output(ConditionAnalysisOutput)
    prompt = f"""당신은 개인 비서 시스템의 컨디션 분석가입니다.
제공된 실제 생체 및 환경 데이터만을 근거로 오늘의 시간대별(오전/오후/저녁) 에너지 점수(0~100)를 산출하세요.
[절대 규칙 - B1, C2, C3 방지]
- 입력 데이터에 없는 날씨, 수면, 과거 기록을 절대 지어내지 마세요.

[실제 입력 데이터]
생체 데이터: {json.dumps(bio_data, ensure_ascii=False)}
환경 데이터: {json.dumps(env_data, ensure_ascii=False)}
"""
    res: ConditionAnalysisOutput = llm.invoke([HumanMessage(content=prompt)])
    scores = {
        "morning": res.morning_score,
        "afternoon": res.afternoon_score,
        "evening": res.evening_score,
        "summary": res.summary,
    }
    print(f" -> 산출 완료: 오전={scores['morning']}, 오후={scores['afternoon']}, 저녁={scores['evening']}")
    return {"condition_score": scores}


# =====================================================================
# 3. 내부 유틸리티: ID 기반 자체 Critic 조건표(A1~E3) 검증 함수
# =====================================================================
def _parse_time_to_min(t_str: str) -> int:
    try:
        h, m = map(int, t_str.strip().split(":"))
        return h * 60 + m
    except Exception:
        return -1


def _detect_explicit_rest_request(user_message: str) -> bool:
    """C1/E1/E2 판단용: 사용자가 명시적으로 휴식/연기를 요청했는지 판별 ('피곤해서 조절해줘'는 False)"""
    rest_keywords = ["쉴래", "쉬고 싶", "휴식할래", "미뤄줘", "내일로 미뤄", "연기해줘", "아파서 못", "오늘은 쉴"]
    return any(kw in user_message for kw in rest_keywords)


def _validate_plan(
    draft: PlannerDraftOutput,
    events: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    today_str: str,
    user_requested_rest: bool,
    active_hours: tuple = (8 * 60, 23 * 60),
) -> List[str]:
    """
    Critic 판정 조건표(A1~A6, B3, C1, D1, D3, D4, E1~E3)를 원본 ID 기준으로 자체 검증.
    위반 사항이 있으면 구체적 수정 지시 문자열 리스트를 반환.
    """
    violations: List[str] = []
    blocks = draft.schedule_blocks

    block_by_id = {b.id: b for b in blocks}
    changed_by_id = {c.id: c for c in draft.changed_items}
    postponed_by_id = {p.id: p for p in draft.postponed_tasks}

    # --- [A3-1, A3-2, E3] 고정 일정 검사 (ID 기준) ---
    for ev in events:
        ev_id = str(ev.get("id", ""))
        title = ev.get("title", ev_id)
        is_fixed = ev.get("is_fixed", True)
        is_adjustable = ev.get("is_adjustable", False)
        orig_start = ev.get("start_time", "")
        orig_end = ev.get("end_time", "")

        if not is_fixed:
            continue

        matched = block_by_id.get(ev_id)
        if not matched:
            violations.append(f"[A3-1] 고정 일정(id={ev_id}, '{title}')이 삭제되었습니다. 고정 일정은 반드시 유지하세요.")
            continue

        time_changed = (matched.start_time != orig_start) or (matched.end_time != orig_end)
        if not is_adjustable:
            # 조정 불가 고정 일정
            if time_changed or matched.status == "proposed" or ev_id in changed_by_id:
                violations.append(
                    f"[A3-1/A3-2] 조정 불가 고정 일정(id={ev_id}, '{title}', {orig_start}~{orig_end})을 이동하거나 제안했습니다. 원래 시간대로 고정하고, 겹치면 unresolvable_warnings에만 기록하세요."
                )
        else:
            # 조정 가능 고정 일정 (E3)
            if time_changed and matched.status != "proposed":
                violations.append(
                    f"[E3] 조정 가능 고정 일정(id={ev_id}, '{title}') 이동 시 status를 반드시 'proposed'로 표시해야 합니다."
                )

    # --- [A1, A2, A5, A6, C1, E1, E2] 할 일(Tasks) 검사 (ID 기준) ---
    today_dt = datetime.strptime(today_str, "%Y-%m-%d").date()

    for task in tasks:
        t_id = str(task.get("id", ""))
        title = task.get("title", t_id)
        deadline_str = task.get("deadline", "")
        is_collab = task.get("is_collaborative", False)
        orig_duration = int(task.get("estimated_minutes", task.get("duration_minutes", 60)))

        days_left = 99
        deadline_time_min = 23 * 60 + 59
        if deadline_str:
            try:
                dl_part = deadline_str.split(" ")
                dl_date = datetime.strptime(dl_part[0], "%Y-%m-%d").date()
                days_left = (dl_date - today_dt).days
                if len(dl_part) > 1:
                    deadline_time_min = _parse_time_to_min(dl_part[1])
            except Exception:
                pass

        is_urgent = days_left <= 2  # 오늘(0), 내일(1), 모레(2)
        scheduled_block = block_by_id.get(t_id)
        postponed_item = postponed_by_id.get(t_id)

        # C1: 명시적 휴식 요청이 없으면 날짜 연기 금지
        if postponed_item and not user_requested_rest:
            violations.append(
                f"[C1] 사용자가 휴식/연기를 명시하지 않았는데 작업(id={t_id}, '{title}')을 연기했습니다. 오늘 일정 내에서 재배치하세요."
            )

        # A5 / E1: 협업 업무 또는 긴박한 업무(마감 오늘~2일)는 연기 불가
        if postponed_item and (is_collab or is_urgent):
            violations.append(
                f"[A5] 협업 업무 또는 긴박한 업무(id={t_id}, '{title}', 마감 D-{days_left})는 연기할 수 없습니다. 오늘 일정에 배치하세요."
            )

        # A1: 오늘 마감 작업 누락 검사
        if days_left == 0 and not scheduled_block:
            violations.append(
                f"[A1] 오늘 마감인 작업(id={t_id}, '{title}')이 계획에서 누락되었습니다. 반드시 오늘 마감 시각({deadline_str}) 전에 배치하세요."
            )

        # 배치된 경우 검사 (A2, A6, E2)
        if scheduled_block:
            end_m = _parse_time_to_min(scheduled_block.end_time)
            # A2: 오늘 마감 시각 초과 여부
            if days_left == 0 and end_m > deadline_time_min:
                violations.append(
                    f"[A2] 오늘 마감 작업(id={t_id}, '{title}')이 마감 시각({deadline_str}) 이후({scheduled_block.end_time})에 배치되었습니다."
                )

            # A6: 일반 조절/브리핑 요청에서 긴박한 업무 분량 축소 금지
            is_reduced = (
                scheduled_block.status == "reduced"
                or scheduled_block.duration_minutes < orig_duration
                or (t_id in changed_by_id and changed_by_id[t_id].change_type == "reduced")
            )
            if not user_requested_rest and is_urgent and is_reduced:
                violations.append(
                    f"[A6] '조절/브리핑' 요청에서는 긴박한 업무(id={t_id}, '{title}', 마감 D-{days_left})의 분량을 줄일 수 없습니다(원래 {orig_duration}분 유지, 시간대 이동만 허용)."
                )

    # --- [B3] 변경 항목 근거 누락 검사 ---
    for item in draft.changed_items:
        if not item.reason or len(item.reason.strip()) < 5:
            violations.append(f"[B3] 변경된 항목(id={item.id}, '{item.title}')에 구체적인 변경 이유가 없습니다.")

    # --- [A4, D3, D4] 시간 겹침, 이동 시간, 활동 시간대 검사 ---
    sorted_blocks = sorted(blocks, key=lambda b: _parse_time_to_min(b.start_time))
    for i, b in enumerate(sorted_blocks):
        s_m = _parse_time_to_min(b.start_time)
        e_m = _parse_time_to_min(b.end_time)
        if s_m < active_hours[0] or e_m > active_hours[1]:
            violations.append(
                f"[D4] '{b.title}'(id={b.id}, {b.start_time}~{b.end_time})가 사용자 활동 시간(08:00~23:00) 밖에 배치되었습니다."
            )
        if i > 0:
            prev = sorted_blocks[i - 1]
            prev_e = _parse_time_to_min(prev.end_time)
            if s_m < prev_e:
                violations.append(
                    f"[A4] '{prev.title}'(id={prev.id}, ~{prev.end_time})와 '{b.title}'(id={b.id}, {b.start_time}~)의 시간이 겹칩니다."
                )
            elif (
                prev.location
                and b.location
                and prev.location != b.location
                and (s_m - prev_e) < 15
            ):
                violations.append(
                    f"[D3] 장소가 다른 '{prev.title}'({prev.location})과 '{b.title}'({b.location}) 사이에 이동 시간(최소 15분)이 부족합니다."
                )

    # --- [D1] 식사/휴식 시간 확보 검사 ---
    total_work_min = sum(
        (_parse_time_to_min(b.end_time) - _parse_time_to_min(b.start_time))
        for b in sorted_blocks
        if b.task_type != "break"
    )
    has_break = any(b.task_type == "break" for b in sorted_blocks)
    if total_work_min >= 240 and not has_break:
        violations.append("[D1] 총 일정이 4시간 이상인데 식사 또는 휴식 블록(task_type='break')이 전혀 없습니다.")

    return violations


# =====================================================================
# 4. Node 2: 일정 플래너 노드 (planner_agent_node)
# =====================================================================
def planner_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[일정 플래너] Critic 조건표(A1~E3) 준수 모드: 작업 분류 및 재배치 시작...")
    user_message = state.get("user_message", "")
    context_data = state.get("context_data", {})
    condition_score = state.get("condition_score", {"morning": 70, "afternoon": 60, "evening": 50})
    critic_result = state.get("critic_result") or {}
    rejected_history = state.get("rejected_history", [])

    events = context_data.get("calendar_events", context_data.get("events", []))
    tasks = context_data.get("tasks", context_data.get("todos", []))
    today_str = context_data.get("today", datetime.now().strftime("%Y-%m-%d"))
    active_window = context_data.get("active_hours", "08:00-23:00")

    user_requested_rest = _detect_explicit_rest_request(user_message)
    external_feedback = critic_result.get("feedback", "") if critic_result.get("verdict") == "rework" else ""

    llm = _get_llm("large").with_structured_output(PlannerDraftOutput)

    system_prompt = f"""당신은 개인 비서 시스템의 '일정 에이전트(Schedule Agent)'입니다.
컨디션 점수와 일정/할 일 데이터를 바탕으로 최적의 하루 일정을 설계하세요.
반드시 아래 규칙을 엄격히 준수해야 합니다.

[1. 원본 ID 유지 규칙 (필수)]
- 입력된 calendar_events와 tasks의 각 항목에 있는 'id' 값을 schedule_blocks, changed_items, postponed_tasks의 'id' 필드에 정확히 그대로 넣으세요. (임의 변경 금지)

[2. 작업 분류 및 사용자 요청별 처리 규칙 (A3, A5, A6, C1, E1~E3)]
- fixed (고정 일정):
  * is_adjustable=False: 절대 시간 이동/삭제/제안 금지 (status='kept'). 마감 작업과 도저히 안 맞으면 unresolvable_warnings에만 기록.
  * is_adjustable=True: 부득이하게 이동 시 status='proposed'로 이동 제안만 가능(E3).
- 긴박한 업무 (마감 오늘~2일 이내) vs 여유 있는 업무 (마감 3일 이상):
  * 일반 브리핑 / "피곤해서 조절해줘" (명시적 휴식 요청 아님, user_requested_rest={user_requested_rest}):
    - 날짜 연기 절대 금지(C1).
    - 긴박한 업무(마감 오늘~2일 이내): **분량을 절대 줄이지 말고(duration_minutes 원본 유지) 시간대만 이동**하세요(A6). 마감 시각 전 완료 필수(A1, A2).
    - 여유 있는 업무(마감 3일 이상): 순서·시간대·분량 조절 허용.
  * "아파서 쉴래" / "미뤄줘" (명시적 휴식 요청, user_requested_rest=True):
    - 개인 업무 + 마감 3일 이상(E1): 1~2일 연기 허용(postponed_tasks에 기록).
    - 협업 업무 또는 마감 오늘~2일 이내(E2): 연기 불가, 오늘 최소 분량(status='reduced')으로 축소하고 무엇을 어디까지 할지 구체 명시.

[3. 에너지 기반 배치 및 실현 가능성 (B1~B3, D1~D6)]
- 에너지가 높은 시간대에 중요한/고난도 작업을 배치하고, 낮은 시간대에는 휴식이나 가벼운 작업을 배치하세요.
- 변경된 모든 항목(changed_items)에는 반드시 실제 데이터(에너지 점수, 마감일)에 근거한 이유를 작성하세요(B3).
- 입력에 없는 날씨, 과거 구매 기록, 확인되지 않은 습관을 절대 언급하지 마세요(B1, B2, C2, C3).
- 일정 간 시간 겹침 금지(A4), 다른 장소 간 이동 시간 최소 15분 확보(D3), 활동 시간({active_window}) 준수(D4), 4시간 이상 활동 시 식사/휴식(task_type='break') 필수 포함(D1).
- 이전 거절 이력(rejected_history)과 동일한 제안 반복 금지(D6).
"""

    base_user_content = f"""[기준 날짜]: {today_str}
[사용자 요청]: "{user_message}"
[명시적 휴식/연기 요청 여부(user_requested_rest)]: {user_requested_rest}
[오늘 컨디션 점수]: {json.dumps(condition_score, ensure_ascii=False)}
[고정 캘린더 일정(calendar_events)]: {json.dumps(events, ensure_ascii=False)}
[할 일 목록(tasks)]: {json.dumps(tasks, ensure_ascii=False)}
[이전 거절 이력(rejected_history)]: {json.dumps(rejected_history, ensure_ascii=False)}
"""
    if external_feedback:
        base_user_content += f"\n[⚠️ 이전 Critic 반려 피드백 - 반드시 반영하여 수정할 것]:\n{external_feedback}\n"

    # 내부 재계획 루프 (최대 3회 시도)
    draft: Optional[PlannerDraftOutput] = None
    internal_violations: List[str] = []

    for attempt in range(1, 4):
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=base_user_content)]
        if internal_violations:
            retry_msg = "방금 생성한 계획에서 다음 규칙 위반이 발견되었습니다. 모두 수정하여 다시 출력하세요:\n- " + "\n- ".join(internal_violations)
            messages.append(HumanMessage(content=retry_msg))

        draft = llm.invoke(messages)
        internal_violations = _validate_plan(draft, events, tasks, today_str, user_requested_rest)

        if not internal_violations:
            print(f" -> [자체 Critic 검사 통과] {attempt}회차에 위반 0건으로 확정")
            break
        else:
            print(f" -> [자체 재계획 루프 {attempt}/3] 위반 {len(internal_violations)}건 감지, 재조정 중: {internal_violations}")

    # Critic 노드가 바로 참조할 수 있도록 요구된 모든 필드 포함
    draft_proposal = {
        "date": today_str,
        "user_requested_rest": user_requested_rest,
        "condition_score": condition_score,
        "schedule_blocks": [b.model_dump() for b in draft.schedule_blocks],
        "changed_items": [c.model_dump() for c in draft.changed_items],
        "postponed_tasks": [p.model_dump() for p in draft.postponed_tasks],
        "unresolvable_warnings": draft.unresolvable_warnings,
        "remaining_violations": internal_violations,
    }

    return {"draft_proposal": draft_proposal}