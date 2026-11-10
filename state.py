from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage

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
    rework_count: int
    
    # 6. 커뮤니케이터가 최종적으로 다듬은 출력 메시지
    final_response: str