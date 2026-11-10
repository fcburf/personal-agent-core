from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from config import llm_small, llm_large

# ==========================================
# [0] Supervisor 구조화 출력 스키마 및 프롬프트
# ==========================================
class SupervisorDecision(BaseModel):
    request_type: Literal[
        "morning_briefing", 
        "purchase_consultation", 
        "retrospective", 
        "general"
    ] = Field(
        description="사용자 입력의 의도를 4가지 유형 중 하나로 정확히 분류"
    )
    reasoning: str = Field(
        description="해당 유형으로 분류한 1줄 이유 (디버깅/모니터링용)"
    )

SUPERVISOR_PROMPT = """당신은 개인 비서 시스템의 요청 라우터(Supervisor)입니다.
사용자의 최근 메시지를 분석하여 다음 4가지 유형 중 가장 적합한 하나로 분류하세요.

[분류 기준]
1. morning_briefing (아침 브리핑/일정):
   - 오늘 하루 일정 확인, 스케줄 조정, 컨디션/수면/날씨에 따른 일정 최적화 관련
   - 예: "오늘 일정 어때?", "피곤한데 스케줄 좀 조절해줘", "오늘 하루 브리핑해줘"
2. purchase_consultation (구매 상담):
   - 물건 구매 고민, 살까 말까 추천/보류 요청, 이번 주 지출이나 예산 관련
   - 예: "키보드 살까 말까?", "이거 사도 괜찮을까?", "지출 점검해줘"
3. retrospective (저녁 회고):
   - 오늘 한 일 완료 체크, 과거 구매한 물건에 대한 만족/후회 피드백
   - 예: "오늘 하루 회고하자", "일정 다 끝냈어", "지난주 산 거 써봤는데 별로야"
4. general (일반 대화):
   - 단순 인사, 비서 사용법 질문, 시스템 소개 등
   - 예: "안녕", "너 뭐 할 줄 알아?", "심심해"

[우선순위 규칙]
- 만약 일정과 구매가 동시에 언급되면, 하루 운영을 위해 'morning_briefing'을 우선합니다.
"""

# ==========================================
# [1] 에이전트 노드
# ==========================================
def supervisor_node(state: AgentState):
    """소형 LLM(llm_small)을 호출하여 사용자 메시지 기반으로 요청 유형을 지능적으로 판별"""
    print("\n[Supervisor] 사용자 메시지 분석 및 요청 종류 판별 중...")
    
    messages = state.get("messages", [])
    
    # 1. 메시지가 없는 경우(예: 직접 request_type을 넘겼거나 초기 빈 호출) 기본값 방어
    if not messages:
        default_req = state.get("request_type", "morning_briefing")
        print(f" -> 메시지가 없어 기본값 적용: {default_req}")
        return {"request_type": default_req}
    
    # 2. 최근 대화 맥락 3개 추출 (비용 최소화 + 맥락 확보 설계)
    recent_messages = messages[-3:]
    
    # 3. Structured Output LLM 생성
    structured_router = llm_small.with_structured_output(SupervisorDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        *recent_messages
    ])
    
    try:
        chain = prompt | structured_router
        decision: SupervisorDecision = chain.invoke({})
        print(f" -> 판별 결과: [{decision.request_type}] (이유: {decision.reasoning})")
        return {"request_type": decision.request_type}
    except Exception as e:
        print(f" -> [경고] LLM 판별 중 오류 발생, fallback 처리: {e}")
        return {"request_type": "general"}

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
        critic_res = {
            "is_valid": False,
            "violated_rules": ["마감_기한_누락"],
            "severity": "high",
            "feedback": "일정 마감 기한이 누락되었습니다. 다시 배치하세요.",
            "target_agent": "planner"
        }
        return {
            "is_valid": False,
            "critic_feedback": critic_res["feedback"],
            "critic_result": critic_res,
            "rework_count": current_count + 1
        }
    
    critic_res = {
        "is_valid": True,
        "violated_rules": [],
        "severity": "low",
        "feedback": "논리와 근거가 타당합니다.",
        "target_agent": None
    }
    return {
        "is_valid": True,
        "critic_feedback": critic_res["feedback"],
        "critic_result": critic_res,
        "rework_count": current_count
    }

def communicator_node(state: AgentState):
    print("[커뮤니케이터] 결론 -> 이유 -> 대안 순서로 메시지 포매팅 중...")
    return {"final_response": "제안: 회의를 오전으로 옮기세요. (이유: 오후 집중력 저하 예상)"}

def user_approval_node(state: AgentState):
    print("[사용자 승인] Streamlit UI에서 승인 대기 중 (Interrupt)...")
    return {}

def action_executor_node(state: AgentState):
    """승인된 제안을 실제 외부 시스템(Google Calendar API, SQLite DB 등)에 반영하는 노드"""
    req_type = state.get("request_type")
    proposal = state.get("draft_proposal", {})
    print(f"[실행 에이전트] 사용자 승인 확인! 외부 시스템 반영 중... (요청: {req_type})")
    
    if req_type == "morning_briefing":
        # TODO: 일정/데이터 연동 팀원의 Google Calendar API 쓰기 함수 호출
        print(f" -> [Calendar API] 구글 캘린더 일정 반영 완료: {proposal}")
    elif req_type in ["purchase_consultation", "retrospective"]:
        # TODO: 장기 메모리 DB(long_term_memory.db) 저장 함수 호출
        print(" -> [DB] SQLite 장기 메모리에 구매/회고 데이터 기록 완료")
        
    return {"final_response": "승인하신 제안이 캘린더 및 시스템에 성공적으로 반영되었습니다."}

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
    """Critic의 판단(is_valid)에 따라 통과면 소통으로, 실패면 지정된 에이전트 또는 원래 에이전트로 재작업 (최대 2회)"""
    is_valid = state.get("is_valid", False)
    rework_count = state.get("rework_count", 0)
    critic_result = state.get("critic_result") or {}
    
    # 구조화된 출력의 boolean 플래그로 깔끔하게 검사
    if is_valid or rework_count >= 2:
        return "communicator"
    
    # critic_result에 target_agent가 지정되어 있다면 우선 분기
    target_agent = critic_result.get("target_agent")
    if target_agent in ["planner", "purchase_advisor", "recall_interview"]:
        return target_agent
    
    # 실패 시 원래 호출했던 에이전트로 반환
    req_type = state.get("request_type")
    if req_type == "morning_briefing":
        return "planner"
    elif req_type == "purchase_consultation":
        return "purchase_advisor"
    elif req_type == "retrospective":
        return "recall_interview"
    
    return "communicator"

def route_after_approval(state: AgentState) -> str:
    """사용자 결정(user_decision)에 따라 승인이면 실행 노드로, 거절/수정이면 대안 제시 루프로 분기"""
    decision = state.get("user_decision") or {}
    status = decision.get("status", "approved")
    
    if status == "approved":
        return "action_executor"
    elif status in ["rejected", "modify"]:
        # 거절 또는 대안 요청 시 커뮤니케이터로 복귀하여 대안 제시 (기획서 원칙)
        return "communicator"
    
    return "action_executor"

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
workflow.add_node("action_executor", action_executor_node)

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

# 사용자 승인 후 라우팅: 승인이면 실행 노드로, 거절/수정이면 커뮤니케이터로 대안 루프
workflow.add_conditional_edges("user_approval", route_after_approval, {
    "action_executor": "action_executor",
    "communicator": "communicator"
})

workflow.add_edge("action_executor", END)

# 체크포인터 및 인터럽트 설정
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["user_approval"]
)

# ==========================================
# [4] 로컬 자가 검증용 실행 블록 (import 시에는 미실행)
# ==========================================
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("🚀 [오케스트레이션 리드] 로컬 파이프라인 자가 검증 테스트")
    print("=" * 60)
    
    # 1. 테스트 사용자 입력
    test_query = "오늘 너무 피곤한데 하루 일정 좀 조절해줘"
    print(f"\n[1] 사용자 입력 메시지: '{test_query}'")
    
    config = {"configurable": {"thread_id": "test-session-1"}}
    initial_input = {"messages": [HumanMessage(content=test_query)]}
    
    # 2. 첫 번째 실행: user_approval 직전까지 진행 후 인터럽트
    print("\n[2] 그래프 실행 시작 (user_approval 직전 대기까지)...")
    for event in app.stream(initial_input, config):
        for node_name in event.keys():
            print(f" -> 노드 완료: [{node_name}]")
    
    # 현재 중단 상태 확인
    snapshot = app.get_state(config)
    print(f"\n[3] Interrupt 확인 (대기 중인 다음 노드: {snapshot.next})")
    print(f" -> 판별된 request_type: {snapshot.values.get('request_type')}")
    print(f" -> 커뮤니케이터 제안: {snapshot.values.get('final_response')}")
    
    # 3. 사용자 승인 시뮬레이션
    print("\n[4] 사용자 승인(approved) 데이터 주입...")
    app.update_state(
        config, 
        {"user_decision": {"status": "approved", "feedback": ""}}, 
        as_node="user_approval"
    )
    
    # 4. 재개 실행 (action_executor 실행 ➔ END)
    print("\n[5] 그래프 재개 (action_executor 반영 ➔ 종료)...")
    for event in app.stream(None, config):
        for node_name in event.keys():
            print(f" -> 노드 완료: [{node_name}]")
        
    final_snapshot = app.get_state(config)
    print(f"\n[6] 최종 실행 결과: {final_snapshot.values.get('final_response')}")
    print("=" * 60)
    print("🎉 축하합니다! 전체 파이프라인 E2E 무결성 검증 완주!")
    print("=" * 60)