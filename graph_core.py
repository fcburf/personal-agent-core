from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from config import llm_small, llm_large

# ==========================================
# [1] 에이전트 노드 (더미 함수)
# ==========================================
def supervisor_node(state: AgentState):
    print("[Supervisor] llm_small을 활용해 요청 종류 판별 중...")
    return {"request_type": "morning_briefing"}

def context_collector_node(state: AgentState):
    print("[맥락 수집] 외부 API 병렬 호출 중...")
    return {"context_data": {"calendar": "회의 2건", "weather": "비", "sleep_score": 60}}

def condition_analyzer_node(state: AgentState):
    print("[컨디션 분석] DB 패턴 및 생체 데이터로 에너지 점수 산출 중...")
    return {"condition_score": {"morning": 80, "afternoon": 40}}

def planner_agent_node(state: AgentState):
    print("[일정 플래너] llm_large를 활용해 일정 재배치 초안 작성 중...")
    # draft_proposal이 Dict[str, Any]로 변경됨
    return {"draft_proposal": {"action": "reschedule", "target": "오후 회의", "reason": "오후 에너지 부족"}}

def purchase_advisor_node(state: AgentState):
    print("[구매 어드바이저] 구매 추천/보류 초안 작성 중...")
    return {"draft_proposal": {"action": "hold", "target": "기계식 키보드", "reason": "이번 주 예산 초과"}}

def recall_interview_node(state: AgentState):
    print("[회상 인터뷰] 과거 구매 만족도 인터뷰 진행 중...")
    return {"draft_proposal": {"interview_item": "블루투스 이어폰", "questions": ["음질은 어땠나요?"]}}

def critic_node(state: AgentState):
    print("[Critic 검증] llm_large로 과잉 개인화 및 논리 검사 중...")
    current_count = state.get("rework_count", 0)
    
    # 실무 패턴: 대형 LLM의 구조화된 출력(Structured Output)을 모사
    # 첫 번째는 반려(False), 두 번째에 통과(True)하도록 더미 로직 구성
    if current_count < 1:
        return {
            "is_valid": False,
            "critic_feedback": "일정 마감 기한이 누락되었습니다. 다시 배치하세요.", 
            "rework_count": current_count + 1
        }
    return {
        "is_valid": True,
        "critic_feedback": "논리와 근거가 타당합니다.", 
        "rework_count": current_count
    }

def communicator_node(state: AgentState):
    print("[커뮤니케이터] 결론 -> 이유 -> 대안 순서로 메시지 포매팅 중...")
    return {"final_response": "제안: 회의를 오전으로 옮기세요. (이유: 오후 집중력 저하 예상)"}

def user_approval_node(state: AgentState):
    print("[사용자 승인] Streamlit UI에서 승인 대기 중 (Interrupt)...")
    return {}

# ==========================================
# [2] 조건부 라우팅 함수 (Edge)
# ==========================================
def route_after_context(state: AgentState) -> str:
    """Supervisor가 판별한 request_type에 따라 담당 에이전트로 이동"""
    req_type = state.get("request_type")
    if req_type == "morning_briefing":
        return "planner"
    elif req_type == "purchase_consultation":
        return "purchase_advisor"
    elif req_type == "retrospective":
        return "recall_interview"
    return "communicator"

def route_after_critic(state: AgentState) -> str:
    """Critic의 판단(is_valid)에 따라 통과면 소통으로, 실패면 원래 에이전트로 재작업 (최대 2회)"""
    is_valid = state.get("is_valid", False)
    rework_count = state.get("rework_count", 0)
    
    # 구조화된 출력의 boolean 플래그로 깔끔하게 검사
    if is_valid or rework_count >= 2:
        return "communicator"
    
    # 실패 시 원래 호출했던 에이전트로 반환
    req_type = state.get("request_type")
    if req_type == "morning_briefing":
        return "planner"
    elif req_type == "purchase_consultation":
        return "purchase_advisor"
    elif req_type == "retrospective":
        return "recall_interview"
    
    return "communicator"

# ==========================================
# [3] 워크플로우(그래프) 조립 및 컴파일
# ==========================================
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("context_collector", context_collector_node)
workflow.add_node("condition_analyzer", condition_analyzer_node)
workflow.add_node("planner", planner_agent_node)
workflow.add_node("purchase_advisor", purchase_advisor_node)
workflow.add_node("recall_interview", recall_interview_node)
workflow.add_node("critic", critic_node)
workflow.add_node("communicator", communicator_node)
workflow.add_node("user_approval", user_approval_node)

# 정적 엣지 연결
workflow.set_entry_point("supervisor")
workflow.add_edge("supervisor", "context_collector")
workflow.add_edge("context_collector", "condition_analyzer")

# 조건부 라우팅
workflow.add_conditional_edges("condition_analyzer", route_after_context, {
    "planner": "planner",
    "purchase_advisor": "purchase_advisor",
    "recall_interview": "recall_interview",
    "communicator": "communicator"
})

workflow.add_edge("planner", "critic")
workflow.add_edge("purchase_advisor", "critic")
workflow.add_edge("recall_interview", "critic")

workflow.add_conditional_edges("critic", route_after_critic, {
    "communicator": "communicator",
    "planner": "planner",
    "purchase_advisor": "purchase_advisor",
    "recall_interview": "recall_interview"
})

workflow.add_edge("communicator", "user_approval")
workflow.add_edge("user_approval", END)

# 체크포인터 및 인터럽트 설정
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["user_approval"]
)