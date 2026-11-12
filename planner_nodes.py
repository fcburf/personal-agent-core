from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState
from config import llm_small, llm_large


# =========================================================
# 1. 구조화 출력(Structured Output) 스키마 정의
# =========================================================
class ClassifiedTask(BaseModel):
    task_id: str = Field(description="작업 식별자 또는 제목")
    title: str = Field(description="일정/작업 제목")
    category: Literal["fixed", "important", "flexible"] = Field(
        description="fixed(고정 일정), important(오늘·내일 마감 또는 고집중 핵심 작업), flexible(여유 있는 유연 작업)"
    )
    is_movable: bool = Field(
        default=False,
        description="고정 일정(fixed)인 경우에만 의미 있음. 사용자가 '조정 가능'으로 명시했으면 True, 기본값은 False(조정 불가)"
    )
    is_collaborative: bool = Field(
        default=False,
        description="회의·팀플 등 다른 사람과 함께하는 협업 업무 여부 (True면 날짜 연기 절대 불가)"
    )
    days_until_deadline: Optional[int] = Field(
        default=None,
        description="마감까지 남은 일수 (오늘=0, 내일=1, 3일 이상=3 등. 마감 없으면 None)"
    )
    deadline_time: Optional[str] = Field(
        default=None,
        description="오늘 마감 시각 (HH:MM 24시간 형식, 예: '18:00', 없으면 None)"
    )
    duration_minutes: int = Field(description="예상 소요 시간(분)")
    location: Optional[str] = Field(default=None, description="장소 정보 (없으면 None)")
    original_slot: Optional[str] = Field(default=None, description="기존 배정 시간 (예: '14:00-15:00')")


class TaskClassificationResult(BaseModel):
    user_requested_rest: bool = Field(
        description="사용자가 메시지로 직접 '오늘 쉬고 싶다', '일정을 미루고 싶다'고 요청했는지 여부 (C1/E1 규칙용)"
    )
    classified_tasks: List[ClassifiedTask]


class ScheduledBlock(BaseModel):
    title: str = Field(description="일정 제목")
    category: Literal["fixed", "important", "flexible"]
    start_time: str = Field(description="시작 시각 (HH:MM 24시간 형식, 예: '09:30')")
    end_time: str = Field(description="종료 시각 (HH:MM 24시간 형식, 예: '11:00')")
    location: Optional[str] = Field(default=None, description="장소")
    deadline_time: Optional[str] = Field(default=None, description="오늘 마감 시각 (HH:MM)")
    action: Literal["keep", "create", "move", "propose_move", "shorten", "postpone"] = Field(
        description=(
            "keep(고정 유지), create(할 일 신규 배치), move(일반/유연 작업 시간 이동), "
            "propose_move(조정 가능 고정 일정의 이동 제안 - E3), shorten(최소 분량 축소 - E2), postpone(1~2일 연기 - E1)"
        )
    )
    reason: str = Field(description="입력 데이터(에너지 점수, 마감, 이동시간 등)에 근거한 배치 이유 (B1/B2/B3 준수)")


class ChangedItemSummary(BaseModel):
    title: str = Field(description="변경된 항목 제목")
    change_type: Literal["create", "move", "propose_move", "shorten", "postpone"]
    before: str = Field(description="변경 전 상태 (예: '미배치' 또는 '14:00-15:00')")
    after: str = Field(description="변경 후 상태 (예: '09:30-11:00' 또는 '내일(11/14)로 연기, 새 마감: 11/16')")
    is_proposal_only: bool = Field(default=False, description="조정 가능 고정 일정(E3) 변경 제안인 경우 True")
    reason: str = Field(description="데이터 근거 기반 변경 이유 (B3 필수)")


class SchedulePlanDraft(BaseModel):
    blocks: List[ScheduledBlock] = Field(description="오늘 하루 전체 시간표 블록 (시간순 정렬, 식사/휴식 포함)")
    changed_items: List[ChangedItemSummary] = Field(
        description="바뀐 항목별 '무엇을 → 어디로 왜' 요약 목록 (Critic B3 및 캘린더 반영용)"
    )
    postponed_tasks: List[Dict[str, str]] = Field(
        default_factory=list,
        description="E1 규칙으로 연기한 작업 목록: {'title': ..., 'new_date': ..., 'new_deadline': ..., 'reason': ...}"
    )
    summary_reason: str = Field(description="전체 재배치 핵심 근거 요약")
    alternatives: List[str] = Field(description="사용자가 거절할 경우 제시할 대안 1~2가지")
    unresolvable_warnings: List[str] = Field(
        default_factory=list,
        description="조정 불가 고정 일정 때문에 마감을 맞출 수 없을 때 남기는 충돌 알림 (이동 제안 금지)"
    )


# =========================================================
# 2. [노드 1] 컨디션 분석 노드 (condition_analyzer_node)
# =========================================================
def condition_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    수면, 피로도, 활동량, 날씨 등 context_data에 실제 존재하는 데이터만 근거로
    시간대별 에너지 점수(0~100)를 산출합니다. (B1/C3 위반 방지: 없는 데이터는 누락으로 명시)
    """
    print("[컨디션 분석] 실제 생체·환경 데이터 기반 시간대별 에너지 점수 산출 중...")
    context = state.get("context_data") or {}

    sleep_score = context.get("sleep_score")
    sleep_hours = context.get("sleep_hours")
    fatigue_level = context.get("fatigue_level")
    weather = context.get("weather")
    active_hours = context.get("active_hours", "08:00-23:00")  # D4 활동 시간대

    available_facts = []
    missing_data = []

    base_score = 70
    if isinstance(sleep_score, (int, float)):
        base_score = int(sleep_score)
        available_facts.append(f"수면 점수 {sleep_score}점")
    elif isinstance(sleep_hours, (int, float)):
        base_score = min(95, max(35, int((sleep_hours / 8.0) * 85)))
        available_facts.append(f"수면 시간 {sleep_hours}시간")
    else:
        missing_data.append("수면 데이터 없음(기본 기준점 70점 적용)")

    if fatigue_level in ("high", 4, 5):
        base_score -= 15
        available_facts.append(f"피로도 높음({fatigue_level})")
    elif fatigue_level in ("low", 1, 2):
        base_score += 10
        available_facts.append(f"피로도 낮음({fatigue_level})")
    elif fatigue_level is None:
        missing_data.append("피로도 자기보고 없음")

    if weather:
        available_facts.append(f"날씨 '{weather}'")
        if "비" in str(weather) or "흐림" in str(weather):
            base_score -= 5
    else:
        missing_data.append("날씨 데이터 없음(날씨 언급 금지)")

    base_score = max(25, min(95, base_score))

    morning_score = min(95, base_score + 15)
    afternoon_score = max(25, base_score - 20)
    evening_score = max(30, base_score - 5)

    scores_map = {"morning": morning_score, "afternoon": afternoon_score, "evening": evening_score}
    peak_period = max(scores_map, key=scores_map.get)

    condition_result = {
        "morning": morning_score,
        "afternoon": afternoon_score,
        "evening": evening_score,
        "peak_period": peak_period,
        "active_hours": active_hours,
        "available_facts": available_facts,   # B1 방어: 플래너는 여기 적힌 사실만 근거로 사용 가능
        "missing_data": missing_data,         # 여기에 적힌 항목은 이유에 절대 언급 금지
    }
    print(f" -> 산출 완료: 오전={morning_score}, 오후={afternoon_score}, 저녁={evening_score}")
    return {"condition_score": condition_result}


# =========================================================
# 3. Critic 조건표 기반 Python 규칙 검증기 (Deterministic Guardrail)
# =========================================================
def _to_minutes(hhmm: str) -> Optional[int]:
    try:
        h, m = map(int, hhmm.strip().split(":"))
        return h * 60 + m
    except Exception:
        return None


def validate_schedule_rules(
    plan: SchedulePlanDraft,
    classified_tasks: List[ClassifiedTask],
    user_requested_rest: bool,
    active_hours: str = "08:00-23:00",
) -> List[str]:
    """
    Critic 판정 조건표(A1~A5, B3, C1, D3, D4, E1~E3)를 코드로 사전 검사합니다.
    """
    violations: List[str] = []
    parsed_intervals = []

    act_start_s, act_end_s = active_hours.split("-") if "-" in active_hours else ("08:00", "23:00")
    act_start_m = _to_minutes(act_start_s) or 480
    act_end_m = _to_minutes(act_end_s) or 1380

    # A1: 오늘 마감인 작업이 계획(또는 연기 목록)에서 빠졌는지 검사
    planned_titles = {b.title for b in plan.blocks}
    postponed_titles = {p.get("title") for p in plan.postponed_tasks}
    for task in classified_tasks:
        if task.days_until_deadline == 0 and task.title not in planned_titles:
            violations.append(f"[A1_마감_누락] 오늘 마감 작업 '{task.title}'이 계획에서 빠졌습니다.")

        # A5 & C1 & E1/E2: 연기 규칙 검사
        if task.title in postponed_titles:
            if not user_requested_rest:
                violations.append(
                    f"[C1_무단_날짜_연기] 사용자가 쉬겠다고 말하지 않았는데 '{task.title}'의 날짜를 미뤘습니다. "
                    "컨디션 데이터만으로는 오늘 내 순서와 분량만 조절하세요."
                )
            if task.is_collaborative or (task.days_until_deadline is not None and task.days_until_deadline <= 2):
                violations.append(
                    f"[A5_연기_규칙_위반] 협업 업무이거나 마감 1~2일 이내 긴박한 '{task.title}'은 연기할 수 없습니다. "
                    "오늘 최소 분량으로 축소(action='shorten')하세요."
                )

    for b in plan.blocks:
        start_m = _to_minutes(b.start_time)
        end_m = _to_minutes(b.end_time)
        if start_m is None or end_m is None or start_m >= end_m:
            violations.append(f"[시간형식오류] '{b.title}'의 시작({b.start_time})~종료({b.end_time})가 잘못되었습니다.")
            continue

        # A2: 마감 시각 이후 배치 검사
        if b.deadline_time:
            dl_m = _to_minutes(b.deadline_time)
            if dl_m is not None and end_m > dl_m:
                violations.append(
                    f"[A2_마감_시각_초과] '{b.title}' 종료({b.end_time})가 마감({b.deadline_time})을 넘겼습니다."
                )

        # D4: 활동 시간대 밖 배치 검사
        if start_m < act_start_m or end_m > act_end_m:
            violations.append(
                f"[D4_활동시간_위반] '{b.title}'({b.start_time}-{b.end_time})이 활동 시간({active_hours})을 벗어났습니다."
            )

        parsed_intervals.append((start_m, end_m, b))

    # A4(시간 겹침) 및 D3(다른 장소 간 이동 시간 부족) 검사
    parsed_intervals.sort(key=lambda x: x[0])
    for i in range(len(parsed_intervals) - 1):
        curr_s, curr_e, curr_b = parsed_intervals[i]
        next_s, next_e, next_b = parsed_intervals[i + 1]
        if curr_e > next_s:
            violations.append(
                f"[A4_시간_충돌] '{curr_b.title}'({curr_b.start_time}-{curr_b.end_time})과 "
                f"'{next_b.title}'({next_b.start_time}-{next_b.end_time}) 시간이 겹칩니다."
            )
        elif (
            curr_b.location and next_b.location
            and curr_b.location != next_b.location
            and (next_s - curr_e) < 30
        ):
            violations.append(
                f"[D3_이동시간_부족] 장소가 다른 '{curr_b.title}'({curr_b.location})과 "
                f"'{next_b.title}'({next_b.location}) 사이에 최소 30분 이동 시간을 확보하세요."
            )

    # A3-1 & A3-2 & E3: 고정 일정 보호 검사
    planned_map = {b.title: b for b in plan.blocks}
    for task in classified_tasks:
        if task.category == "fixed":
            if task.title not in planned_map:
                violations.append(f"[A3-1_고정일정_삭제금지] 고정 일정 '{task.title}'이 삭제되었습니다.")
                continue
            pb = planned_map[task.title]
            if not task.is_movable:
                if pb.action in ("move", "propose_move"):
                    violations.append(
                        f"[A3-2_조정불가_고정일정_이동제안금지] '{task.title}'은 조정 불가 고정 일정입니다. "
                        "이동 제안을 하지 말고 원래 시간(keep)을 유지한 채 충돌 알림만 남기세요."
                    )
                elif task.original_slot and "-" in task.original_slot:
                    orig_s, orig_e = [x.strip() for x in task.original_slot.split("-")]
                    if (pb.start_time, pb.end_time) != (orig_s, orig_e):
                        violations.append(
                            f"[A3-1_고정일정_이동금지] 고정 일정 '{task.title}'의 시간이 변경되었습니다."
                        )
            else:
                # E3: 조정 가능 고정 일정은 'move'가 아니라 반드시 'propose_move'(제안)로 표시
                if pb.action == "move":
                    violations.append(
                        f"[E3_제안표시_누락] 조정 가능 고정 일정 '{task.title}' 변경은 확정 이동('move')이 아니라 "
                        "이동 제안('propose_move')으로 표시해야 합니다."
                    )

    # B3: 바뀐 항목에 이유가 있는지 검사
    for item in plan.changed_items:
        if not item.reason or len(item.reason.strip()) < 5:
            violations.append(f"[B3_변경이유_누락] 바뀐 항목 '{item.title}'에 구체적인 변경 이유가 없습니다.")

    return violations


# =========================================================
# 4. [노드 2] 일정 플래너 노드 (planner_agent_node)
# =========================================================
CLASSIFY_PROMPT = """당신은 개인 비서 DAYWISE의 작업 분류기입니다.
캘린더 일정과 할 일(Tasks) 목록을 분석해 각 항목을 분류하고 속성을 추출하세요.

[분류 기준]
1. fixed (고정 일정): 수업, 팀 회의, 병원 예약 등 정해진 시각의 일정.
   - `is_movable`: 데이터에 '조정 가능(movable=True)'으로 명시된 경우만 True, 언급이 없으면 무조건 False(조정 불가).
2. important (중요 작업): 오늘·내일 마감(days_until_deadline <= 2)이거나 고집중 핵심 과제.
3. flexible (유연 작업): 마감이 여유롭거나(3일 이상) 없어 시간 이동·분량 축소가 가능한 작업.
4. `user_requested_rest`: 사용자가 대화에서 직접 "오늘 쉬고 싶다", "일정 내일로 미뤄줘"라고 명시한 경우만 True.
"""

PLANNER_PROMPT = """당신은 사용자의 컨디션과 일정을 연결하는 '일정 플래너 에이전트'입니다.
Critic 판정 조건표(A1~D6, E1~E3)를 100% 통과하도록 오늘 하루 계획을 세우세요.

[필수 준수 규칙]
1. **목표 및 고정 일정 (A1~A5, E3)**:
   - 오늘 마감 작업(`days_until_deadline == 0`)은 절대 누락하지 말고(A1), 마감 시각(`deadline_time`) 이전에 끝내세요(A2).
   - `is_movable=False`인 고정 일정은 절대 이동하거나 이동 제안조차 하지 말고 `action="keep"`으로 원래 시간을 지키세요(A3-1, A3-2).
   - 만약 조정 불가 고정 일정 때문에 마감을 못 맞춘다면, 고정 일정을 옮기지 말고 `unresolvable_warnings`에 충돌 알림만 적으세요.
   - `is_movable=True`인 고정 일정만 `action="propose_move"`로 이동 제안을 할 수 있습니다(E3).
2. **날짜 연기 제한 (C1, E1, E2)**:
   - `user_requested_rest=False`일 때는 컨디션이 나빠도 절대 날짜를 미루지 말고(C1), 오늘 안에서 순서와 분량만 조절하세요.
   - `user_requested_rest=True`라도 협업 업무(`is_collaborative=True`)나 긴박 업무(`days_until_deadline <= 2`)는 연기 불가이며, `action="shorten"`으로 오늘 최소 분량(어디까지 할지)만 배정하세요(E2).
   - 개인 업무이면서 마감 3일 이상 남은 경우만 `action="postpone"`(1~2일 연기) 가능하며, `postponed_tasks`에 새 날짜와 새 마감을 모두 적으세요(E1).
3. **근거 정확성 및 과잉 개인화 금지 (B1~B3, C2)**:
   - 반드시 `available_facts`에 있는 데이터만 근거로 쓰고, `missing_data`에 적힌 항목(예: 없는 날씨/수면)은 절대 지어내지 마세요(B1).
   - 에너지 점수가 높은 시간대에 중요 작업을 배치하고, 점수 수치와 반대로 설명하지 마세요(B2).
   - 구매 기록 등 일정과 무관한 기억은 절대 언급하지 마세요(C2).
   - 모든 변경 사항은 `changed_items`에 '변경 전 → 변경 후'와 데이터 근거 이유를 빠짐없이 적으세요(B3).
4. **실현 가능성 및 거절 이력 회피 (D1, D3, D4, D6)**:
   - 점심/저녁 식사 또는 휴식 시간을 확보하고(D1), 다른 장소 일정 사이에는 최소 30분 이동 시간을 두며(D3), `active_hours` 안에서만 배치하세요(D4).
   - `rejected_history`에 있는 거절된 제안은 절대 다시 내지 마세요(D6).
"""


def planner_agent_node(state: AgentState) -> Dict[str, Any]:
    print("[일정 플래너] Critic 조건표 준수 모드: 작업 분류 및 재배치 시작...")

    context = state.get("context_data") or {}
    condition = state.get("condition_score") or {
        "morning": 75, "afternoon": 55, "evening": 60,
        "active_hours": "08:00-23:00", "available_facts": [], "missing_data": ["상세 컨디션 데이터 없음"]
    }
    rejected_history = state.get("rejected_history") or []
    user_decision = state.get("user_decision") or {}

    external_feedback = []
    if state.get("critic_verdict") == "rework" and state.get("critic_feedback"):
        external_feedback.append(f"[Critic 반려 피드백 - 반드시 수정]: {state['critic_feedback']}")
    if user_decision.get("status") == "modify" and user_decision.get("feedback"):
        external_feedback.append(f"[사용자 수정 요청]: {user_decision['feedback']}")

    # 1단계: 작업 분류 및 사용자 휴식 요청 여부 판별 (llm_small)
    raw_schedule_input = {
        "calendar_events": context.get("calendar_events") or context.get("calendar", []),
        "tasks": context.get("tasks", []),
        "user_message": state["messages"][-1].content if state.get("messages") else "오늘 일정 브리핑해줘",
    }

    classifier = llm_small.with_structured_output(TaskClassificationResult)
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFY_PROMPT),
        ("human", "다음 일정/할 일과 사용자 메시지를 분석해 분류하세요:\n{raw_data}"),
    ])
    classified_res: TaskClassificationResult = (classify_prompt | classifier).invoke(
        {"raw_data": str(raw_schedule_input)}
    )

    # 2단계 & 3단계: 재배치 및 Critic 조건표 사전 검증 루프 (최대 2회 재시도)
    planner_llm = llm_large.with_structured_output(SchedulePlanDraft)
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_PROMPT),
        ("human", (
            "1) 사용자 직접 휴식/연기 요청 여부(user_requested_rest): {user_requested_rest}\n\n"
            "2) 분류된 작업 목록:\n{tasks}\n\n"
            "3) 시간대별 컨디션 및 사용 가능한 근거(available_facts / missing_data):\n{condition}\n\n"
            "4) 과거 거절·수정 이력 (D6 - 동일 제안 금지):\n{rejected_history}\n\n"
            "5) 수정 피드백:\n{feedback}"
        )),
    ])

    current_feedback = "\n".join(external_feedback) if external_feedback else "없음"
    best_plan: Optional[SchedulePlanDraft] = None
    loop_violations: List[str] = []

    for attempt in range(3):
        plan_draft: SchedulePlanDraft = (plan_prompt | planner_llm).invoke({
            "user_requested_rest": classified_res.user_requested_rest,
            "tasks": str([t.model_dump() for t in classified_res.classified_tasks]),
            "condition": str(condition),
            "rejected_history": str(rejected_history),
            "feedback": current_feedback,
        })

        loop_violations = validate_schedule_rules(
            plan=plan_draft,
            classified_tasks=classified_res.classified_tasks,
            user_requested_rest=classified_res.user_requested_rest,
            active_hours=condition.get("active_hours", "08:00-23:00"),
        )
        best_plan = plan_draft

        if not loop_violations:
            print(f" -> [자체 Critic 검사 통과] {attempt + 1}회차에 위반 0건으로 확정")
            break

        print(f" -> [자체 Critic 검사 위반 감지] {attempt + 1}회차: {loop_violations}")
        current_feedback += "\n[자체 규칙 위반 - 즉시 수정할 것]: " + "; ".join(loop_violations)

    # 4단계: Critic 및 action_executor가 요구하는 규격으로 draft_proposal 조립
    calendar_write_payload = [
        {
            "action": b.action,
            "title": b.title,
            "start_time": b.start_time,
            "end_time": b.end_time,
            "location": b.location,
            "category": b.category,
            "description": f"[DAYWISE] {b.reason}",
        }
        for b in best_plan.blocks
        if b.action in ("create", "move", "propose_move", "shorten")
    ]

    draft_proposal = {
        "agent": "planner",
        "classified_tasks": [t.model_dump() for t in classified_res.classified_tasks],
        "daily_schedule": [b.model_dump() for b in best_plan.blocks],
        "changed_items": [c.model_dump() for c in best_plan.changed_items],  # B3: 무엇을 → 어디로 왜
        "postponed_tasks": best_plan.postponed_tasks,                        # E1: 미룬 작업과 새 마감
        "changes": calendar_write_payload,                                   # 승인 후 캘린더 반영 및 D6 추적용
        "reason": best_plan.summary_reason,
        "alternatives": best_plan.alternatives,
        "warnings": best_plan.unresolvable_warnings + loop_violations,
        "used_evidence": {
            "condition_score": condition,
            "available_facts": condition.get("available_facts", []),
        },
    }

    return {"draft_proposal": draft_proposal}
