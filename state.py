from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
import operator
from langchain_core.messages import BaseMessage

class CriticResult(TypedDict, total=False):
    is_valid: bool                 # 통과 여부 (True/False)
    violated_rules: List[str]      # 위반 규칙 목록 (Critic 조건표 ID 형식, 예: ["A1_마감_누락", "C2_무관한_기억_사용"])
    severity: str                  # 심각도 ("low" | "medium" | "high" | "critical")
    feedback: str                  # 구체적 개선 가이드 문장
    target_agent: Optional[str]    # 재작업을 지시할 에이전트 ("planner" | "purchase_advisor" | "recall_interview")

class RejectedEntry(TypedDict, total=False):
    status: str                    # 'rejected' | 'modify'
    changes: Any                   # 거절·수정 대상이 된 변경 내용 (draft_proposal의 changes)
    feedback: Optional[str]        # 사용자 피드백
    decided_at: Optional[str]      # 결정 시각 (ISO 8601)

class UserDecision(TypedDict, total=False):
    status: str                    # 승인 상태 ('approved' | 'rejected' | 'modify')
    feedback: Optional[str]        # 거절 또는 대안 요청 시 사용자 의견/피드백
    decided_at: Optional[str]      # 사용자 결정 시각 (ISO 8601 타임스탬프)

class AgentState(TypedDict):
    # LangGraph 권장 패턴: 대화 메시지 누적
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 1. Supervisor 판별 결과 (예: 'morning_briefing', 'purchase_consultation')
    request_type: str
    
    # 2. 맥락 수집 에이전트가 채워넣을 외부 실제 데이터 (캘린더, 날씨, 생체 등)
    context_data: Dict[str, Any]
    
    # 3. 컨디션 분석 결과 (시간대별 에너지 점수 등)
    condition_score: Dict[str, Any]
    
    # 4. 각 역할 에이전트(일정, 구매)가 생성한 제안 초안 (Dict 구조로 유연성 확보)
    draft_proposal: Dict[str, Any]
    
    # 5. Critic(검증) 에이전트 피드백 및 재작업 루프 관리
    is_valid: bool
    critic_feedback: str
    critic_result: CriticResult  # 구조화된 Critic 결과 (위반 규칙, 심각도, 대상 에이전트 등)
    rework_count: int
    
    # 6. 커뮤니케이터가 최종적으로 다듬은 출력 메시지
    final_response: str
    
    # 7. 사용자 승인 결과 (Streamlit 화면에서 사용자의 승인/거절/수정 의견 기록)
    user_decision: UserDecision

    # 8. Critic 최종 판정 (커뮤니케이터가 안내 문구를 고르는 기준)
    #    "rework"         : 반려, 재작업 기회 남음 (그래프 내부용, 커뮤니케이터까지 가지 않음)
    #    "pass"           : 통과 (medium·low 경고는 critic_result.violated_rules에 남아 있을 수 있음)
    #    "warn"           : 재작업 2회 후에도 high 잔존 → 경고 표시를 붙여 전달
    #    "keep_original"  : 재작업 2회 후에도 critical 잔존 → 제안을 내보내지 않고 "원래 일정 유지" 안내
    critic_verdict: Literal["rework", "pass", "warn", "keep_original"]

    # 10. 커뮤니케이터 응답 뒤 사용자 승인이 필요한지 (커뮤니케이터 노드가 기록, 라우터가 읽음)
    #     False: 일반 대화 / 원래 일정 유지 / 거절 후 안내 → 승인 단계 없이 종료
    awaiting_approval: bool

    # 9. 사용자가 거절·수정한 제안 이력 (D6: 거절한 제안 재제출 금지 검사용, 누적)
    #    각 항목: {"status": "rejected"|"modify", "changes": ..., "feedback": str, "decided_at": str}
    rejected_history: Annotated[List[RejectedEntry], operator.add]

# ==========================================
# 공용 도우미: 노드들이 State에서 값을 꺼낼 때 같은 방식으로 읽도록 통일
# ==========================================
def get_last_user_message(state) -> str:
    """messages에서 가장 최근 사용자(Human) 메시지 텍스트를 꺼낸다.
    사용자 메시지의 원본은 messages 하나뿐이다 (user_message 같은 별도 칸을 두지 않음).
    수정 요청(modify) 피드백도 user_approval 노드가 messages에 추가하므로 여기서 함께 읽힌다."""
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", None) == "human":
            content = msg.content
            if isinstance(content, list):  # 멀티모달 메시지: 텍스트 조각만 이어 붙임
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                )
            return str(content)
    return ""
