from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage

class CriticResult(TypedDict, total=False):
    is_valid: bool                 # 통과 여부 (True/False)
    violated_rules: List[str]      # 위반 규칙 목록 (예: ["마감_기한_누락", "과잉_개인화", "근거_부족"])
    severity: str                  # 심각도 ("low" | "medium" | "high" | "critical")
    feedback: str                  # 구체적 개선 가이드 문장
    target_agent: Optional[str]    # 재작업을 지시할 에이전트 ("planner" | "purchase_advisor" | "recall_interview")

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
    critic_result: Dict[str, Any]  # 구조화된 Critic 결과 (위반 규칙, 심각도, 대상 에이전트 등)
    rework_count: int
    
    # 6. 커뮤니케이터가 최종적으로 다듬은 출력 메시지
    final_response: str
    
    # 7. 사용자 승인 결과 (Streamlit 화면에서 사용자의 승인/거절/수정 의견 기록)
    user_decision: Dict[str, Any]