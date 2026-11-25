from datetime import datetime, timedelta
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from config import llm_small, llm_large
from planner_nodes import condition_analyzer_node, planner_agent_node

# ==========================================
# [0] Supervisor 구조화 출력 스키마 및 프롬프트
# ==========================================
class SupervisorDecision(BaseModel):
    request_type: Literal[
        "trip_planning",
        "morning_briefing",
        "purchase_consultation",
        "retrospective",
        "general"
    ] = Field(
        description="사용자 입력의 의도를 5가지 유형 중 하나로 정확히 분류"
    )
    reasoning: str = Field(
        description="해당 유형으로 분류한 1줄 이유 (디버깅/모니터링용)"
    )

SUPERVISOR_PROMPT = """당신은 개인 비서 시스템의 요청 라우터(Supervisor)입니다.
사용자의 최근 메시지를 분석하여 다음 5가지 유형 중 가장 적합한 하나로 분류하세요.

[분류 기준]
0. trip_planning (여행 전 일정 정리):
   - 여행 전에 입력한 가고 싶은 곳·먹고 싶은 것을 바탕으로 여행 전체 일정을 짜 달라는 요청
   - 예: "오사카 여행 일정 짜줘", "여행 계획 정리해줘", "3일 동안 어디 갈지 정해줘"
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
- 여행 '전체' 일정을 짜 달라면 trip_planning, '오늘' 일정이면 morning_briefing입니다.
"""

# ==========================================
# [1] 에이전트 노드
# ==========================================
# 새 요청이 시작될 때마다 지난 요청의 흔적을 지우는 칸들.
# (같은 thread_id에서 이전 요청의 user_decision="rejected"나 rework_count=3이 남아 있으면
#  새 요청이 '거절 후'로 오인되거나 재작업 기회 없이 끝나버림)
NEW_REQUEST_RESET = {"user_decision": {}, "rework_count": 0, "is_valid": False}

def supervisor_node(state: AgentState):
    """소형 LLM(llm_small)을 호출하여 사용자 메시지 기반으로 요청 유형을 지능적으로 판별"""
    print("\n[Supervisor] 사용자 메시지 분석 및 요청 종류 판별 중...")
    
    messages = state.get("messages", [])
    
    # 1. 메시지가 없는 경우(예: 직접 request_type을 넘겼거나 초기 빈 호출) 기본값 방어
    if not messages:
        default_req = state.get("request_type", "morning_briefing")
        print(f" -> 메시지가 없어 기본값 적용: {default_req}")
        return {"request_type": default_req, **NEW_REQUEST_RESET}
    
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
        return {"request_type": decision.request_type, **NEW_REQUEST_RESET}
    except Exception as e:
        print(f" -> [경고] LLM 판별 중 오류 발생, fallback 처리: {e}")
        return {"request_type": "general", **NEW_REQUEST_RESET}

def build_dummy_context(today=None) -> dict:
    """[형식 예시] 맥락 수집 노드가 반환할 context_data 모양 (여행 시나리오, 11/26 확정).
    실제 데모 데이터(오사카 일정·컨디션·날씨)는 지인 님, 구매 이력·예산·상품은 예정 님이 이 형식으로 채움.
    키 이름·필드는 플래너(채린)·구매 에이전트(서경)·Critic(원형)이 그대로 읽으므로 바꾸면 README와 함께 공지."""
    today = today or datetime.now().date()
    d = lambda days: (today + timedelta(days=days)).strftime("%Y-%m-%d")
    return {
        "today": d(0),                       # 기준 날짜 YYYY-MM-DD
        "active_hours": "08:00-23:00",       # 사용자 활동 시간대 (D4)

        # 여행 정보 (여행 전 입력 폼 → trips 테이블) — 지인
        "trip": {
            "trip_id": "trip_osaka",
            "destination": "일본 오사카",
            "start_date": d(-1), "end_date": d(1),          # 오늘 = 여행 2일차
            "accommodation": "난바역 근처 호텔",
            "budget": 60000, "currency": "JPY",              # 여행 예산 (구매 상담·PA1)
            "wishlist": {
                "places": ["도톤보리", "오사카성", "유니버설 스튜디오 재팬"],
                "foods": ["타코야키", "오코노미야키"],
                "activities": ["쿠로몬 시장 구경"],
            },
        },

        # 오늘 일정만 (플래너 A3·D3 검사 대상). walking_level = 많음/보통/적음 — 지인
        "calendar_events": [
            {"id": "evt_d2_1", "title": "유니버설 스튜디오 재팬 (입장 예약)", "start_time": "09:00", "end_time": "17:00",
             "is_fixed": True, "is_adjustable": False, "has_attendees": False,
             "location": "유니버설 스튜디오 재팬", "walking_level": "많음"},
            {"id": "evt_d2_2", "title": "도톤보리 저녁", "start_time": "19:00", "end_time": "20:30",
             "is_fixed": True, "is_adjustable": True, "has_attendees": False,
             "location": "도톤보리", "walking_level": "보통"},
        ],
        "tasks": [],                         # 여행 중에는 보통 비어 있음 (형식은 기존과 동일)

        # 여행 전체 날짜별 일정 (구매 상담이 '남은 일정'을 읽는 곳) — 지인
        "trip_schedule": [
            {"date": d(-1), "events": [
                {"id": "evt_d1_1", "title": "오사카성", "start_time": "13:00", "end_time": "16:00", "walking_level": "많음"}]},
            {"date": d(0), "events": [
                {"id": "evt_d2_1", "title": "유니버설 스튜디오 재팬", "start_time": "09:00", "end_time": "17:00", "walking_level": "많음"},
                {"id": "evt_d2_2", "title": "도톤보리 저녁", "start_time": "19:00", "end_time": "20:30", "walking_level": "보통"}]},
            {"date": d(1), "events": [
                {"id": "evt_d3_1", "title": "쿠로몬 시장", "start_time": "10:00", "end_time": "13:00", "walking_level": "많음"}]},
        ],

        # 최근 3일 수면·피로도 + 오늘 걸음 수 (컨디션 분석, 구매 수량 판단) — 지인
        "biometrics": {
            "recent": [
                {"date": d(-2), "sleep_type": "score", "sleep_score": 78, "sleep_hours": 7.0, "fatigue_level": 2},
                {"date": d(-1), "sleep_type": "manual", "sleep_score": None, "sleep_hours": 5.0, "fatigue_level": 4},
                {"date": d(0),  "sleep_type": "manual", "sleep_score": None, "sleep_hours": 4.0, "fatigue_level": 4},
            ],
            "steps_today": 21000,
        },

        # 날짜별 날씨 (없으면 키 생략 → B1) — 지인
        "weather": [
            {"date": d(0), "condition": "맑음", "temp_min_c": 8, "temp_max_c": 15},
            {"date": d(1), "condition": "비", "temp_min_c": 9, "temp_max_c": 13},
        ],

        # 구매·지출 (예정 님 spending_builder). 여행 중 사용액 포함
        "spending": {
            "records": [
                {"item_name": "타코야키", "category": "식품", "price": 800, "currency": "JPY",
                 "purchase_date": d(-1), "data_source": "manual"},
            ],
            "weekly_total": 84000,
            "monthly_total": 213000,
            "monthly_budget": None,          # 미설정이면 None → PA1 검사 생략
            "trip_spent": 12800,             # 여행 예산 대비 사용액 (trip.currency 기준)
            "by_category": {"식품": 800},
            "data_quality": "ok",
        },

        # 구매 상담 비교 상품 (실서비스는 구매 에이전트가 Google 검색으로 채움, 개발용 더미) — 예정·서경
        "products": [
            {"id": "A", "name": "휴족시간 18매입", "price": 932, "currency": "JPY", "unit_count": 18,
             "source": "dummy", "source_url": None},
            {"id": "B", "name": "샤론파스 40매입", "price": 1180, "currency": "JPY", "unit_count": 40,
             "source": "dummy", "source_url": None},
        ],
    }

def context_collector_node(state: AgentState):
    print("[맥락 수집] 외부 API 병렬 호출 중... (현재 더미 데이터)")
    return {"context_data": build_dummy_context()}

def purchase_advisor_node(state: AgentState):
    """[임시 더미] TODO(서경): 실제 구매 어드바이저로 교체. 출력 형식은 state.PurchaseProposal"""
    print("[구매 어드바이저] 구매 추천/보류 초안 작성 중... (현재 더미)")
    # 오사카 데모 예시: 남은 일정 중 도보 '많음'인 날 수로 수량 판단, 여행 예산 내에서 저렴한 쪽
    return {"draft_proposal": {
        "decision": "recommend",
        "item": "다리 피로 회복 패치",
        "category": "건강·위생",
        "purchase_context": "유니버설 스튜디오에서 약 2만 보를 걸어 다리 통증",
        "criteria": ["남은 일정 도보량", "여행 예산"],
        "candidates": [
            {"id": "A", "name": "휴족시간 18매입", "price": 932, "currency": "JPY"},
            {"id": "B", "name": "샤론파스 40매입", "price": 1180, "currency": "JPY"},
        ],
        "recommended": {"name": "휴족시간 18매입", "price": 932, "currency": "JPY", "quantity": 1},
        "quantity_basis": "남은 일정 중 도보 '많음' 1일 + 오늘 밤 → 18매 1상자로 충분",
        "impulse_signals": [],
        "evidence": ["trip_schedule: 내일 쿠로몬 시장(도보 많음)", "trip.budget 60,000엔 중 12,800엔 사용"],
        "revisit_at": None,
    }}

def recall_interview_node(state: AgentState):
    """[임시 더미] TODO(서경): 실제 회상 인터뷰로 교체. 출력 형식은 state.RecallResult"""
    print("[회상 인터뷰] 과거 구매 만족도 인터뷰 진행 중... (현재 더미)")
    return {"draft_proposal": {
        "purchase_id": 1,
        "recall_stage": "1_week",
        "questions": ["블루투스 이어폰, 써보니 어떠셨나요?"],
        "satisfaction": None,
        "reason_tags": [],
        "comment": None,
        "skipped": False,
    }}

MAX_REWORK = 2  # 플래너 등 역할 에이전트가 다시 도는 최대 횟수 (Critic 검사는 최대 3번)
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

def decide_critic_verdict(severity: str, rework_count: int) -> tuple[bool, str]:
    """Critic 조건표의 '심각도별 처리' 규칙을 코드로 고정한 함수 (LLM이 아니라 코드가 판정).

    - rework_count: 이번 검사까지 포함한 누적 반려 횟수
    - 반환: (is_valid, critic_verdict)
        high 미만 → (True, "pass")
        high 이상 & 재작업 기회 남음 → (False, "rework")
        재작업 기회 소진 & critical → (False, "keep_original")
        재작업 기회 소진 & high     → (False, "warn")
    """
    if SEVERITY_RANK.get(severity, 0) < SEVERITY_RANK["high"]:
        return True, "pass"
    if rework_count <= MAX_REWORK:
        return False, "rework"
    return False, ("keep_original" if severity == "critical" else "warn")

def critic_node(state: AgentState):
    print("[Critic 검증] llm_large로 과잉 개인화 및 논리 검사 중...")
    prev_count = state.get("rework_count", 0)

    # TODO(원형): 대형 LLM 구조화 출력으로 교체. 입력: draft_proposal, context_data,
    #             condition_score, rejected_history(D6 검사용). 출력: violated_rules, severity, feedback, target_agent
    # 더미 로직: 첫 검사는 high로 반려, 이후 통과
    if prev_count < 1:
        critic_res = {
            "violated_rules": ["A1_마감_누락"],
            "severity": "high",
            "feedback": "일정 마감 기한이 누락되었습니다. 다시 배치하세요.",
            "target_agent": AGENT_BY_REQUEST.get(state.get("request_type")),
        }
    else:
        critic_res = {
            "violated_rules": [],
            "severity": "low",
            "feedback": "논리와 근거가 타당합니다.",
            "target_agent": None,
        }

    # 반려일 때만 카운트 증가 → 판정은 decide_critic_verdict가 일괄 결정
    rejected = SEVERITY_RANK.get(critic_res["severity"], 0) >= SEVERITY_RANK["high"]
    new_count = prev_count + 1 if rejected else prev_count
    is_valid, verdict = decide_critic_verdict(critic_res["severity"], new_count)
    critic_res["is_valid"] = is_valid
    print(f" -> 판정: {verdict} (severity={critic_res['severity']}, 누적 반려={new_count})")

    return {
        "is_valid": is_valid,
        "critic_feedback": critic_res["feedback"],
        "critic_result": critic_res,
        "critic_verdict": verdict,
        "rework_count": new_count,
    }

def needs_approval(state: AgentState) -> bool:
    """[오케스트레이션] 이번 커뮤니케이터 응답 뒤에 사용자 승인이 필요한가?
    승인할 제안이 없는 경우: 일반 대화 / 원래 일정 유지(keep_original) / 거절 후 안내"""
    if state.get("request_type") == "general":
        return False
    if state.get("critic_verdict") == "keep_original":
        return False
    if is_after_rejection(state):
        return False
    return True

def communicator_node(state: AgentState):
    """커뮤니케이터 = 문구(원형 님) + 승인 필요 여부 기록(리드).
    awaiting_approval은 반드시 이 노드가 기록한다. 라우터가 user_decision을 직접 보면,
    UI가 update_state로 결정을 써넣는 순간 라우터가 다시 계산되어 user_approval 노드가 건너뛰어진다."""
    result = _communicator_message(state)
    result["awaiting_approval"] = needs_approval(state)
    return result

def _communicator_message(state: AgentState):
    """TODO(원형): 안내 문구 작성. critic_verdict 값에 따라 문구 분기.
    - "pass"          : 결론 → 이유 → 대안 (critic_result.violated_rules의 medium·low 경고 표시)
    - "warn"          : 위와 같되 경고 표시를 붙여 전달
    - "keep_original" : 제안을 내보내지 않고 "원래 일정 유지" + 충돌 이유 안내 (승인 단계 없이 종료)
    - request_type == "general" : 일반 대화 응답 (승인 단계 없이 종료)
    - 거절 후(user_decision.status == "rejected") : "원래 일정 유지 + 무엇을 바꾸고 싶은지" 질문 (승인 단계 없이 종료)
      (커뮤니케이터는 draft_proposal을 바꾸지 못하므로, 거절된 계획을 다시 승인받지 않는다)
    """
    verdict = state.get("critic_verdict")
    print(f"[커뮤니케이터] 결론 -> 이유 -> 대안 순서로 메시지 포매팅 중... (verdict={verdict})")
    if state.get("request_type") == "general":
        return {"final_response": "안녕하세요! 일정 브리핑, 구매 상담, 저녁 회고를 도와드릴 수 있어요."}
    if is_after_rejection(state):
        return {"final_response": "알겠습니다. 오늘은 원래 일정을 그대로 유지할게요. 바꾸고 싶은 부분이 있으면 말씀해 주세요."}
    if verdict == "keep_original":
        return {"final_response": "검토 결과 마감·고정 일정을 지킬 수 있는 안전한 재배치안을 찾지 못해 원래 일정을 유지합니다."}
    prefix = "⚠️ [확인 필요] " if verdict == "warn" else ""
    return {"final_response": prefix + "제안: 회의를 오전으로 옮기세요. (이유: 오후 집중력 저하 예상)"}

def user_approval_node(state: AgentState):
    """interrupt_before로 멈췄다가, UI가 user_decision을 넣고 재개하면 실행되는 노드.
    거절·수정이면 rejected_history에 쌓고(D6), 새 계획을 위해 Critic 카운터를 초기화한다."""
    decision = state.get("user_decision") or {}
    status = decision.get("status", "approved")
    print(f"[사용자 승인] 사용자 결정 수신: {status}")

    if status not in ("rejected", "modify"):
        return {}

    proposal = state.get("draft_proposal") or {}
    entry = {
        "status": status,
        "changes": extract_changes(proposal),
        "feedback": decision.get("feedback", ""),
        "decided_at": decision.get("decided_at") or datetime.now().isoformat(timespec="seconds"),
    }
    # 구매 상담 전용: 거절 이유 태그(화면에서 선택) + 구매하려던 이유·상황 (같은 추천 판단, 다음 상담 참고)
    if decision.get("reason_tags"):
        entry["reject_reason_tags"] = decision["reason_tags"]
    if proposal.get("purchase_context"):
        entry["purchase_context"] = proposal["purchase_context"]
    update = {"rejected_history": [entry]}  # operator.add 리듀서 → 기존 이력 뒤에 추가
    if status == "modify":
        # 사용자 요청으로 새 계획을 짜는 것이므로 재작업 기회를 다시 2회 부여
        # (critic_verdict는 Critic이 재검사하며 덮어쓰므로 초기화하지 않음)
        update.update({"rework_count": 0, "is_valid": False})
        # 수정 요청 문장("3시 일정은 내일로")을 사용자 메시지로 남겨, 플래너가 get_last_user_message로 읽게 함
        if decision.get("feedback"):
            update["messages"] = [HumanMessage(content=decision["feedback"])]
    return update

def extract_changes(proposal: dict):
    """거절·수정 이력(D6·PD2 검사용)에 남길 '같은 추천인지 비교할 핵심'만 추린다.
    - 아침 브리핑(PlannerProposal): changed_items + postponed_tasks (원본 일정 id로 비교)
    - 여행 전체 일정(TripPlanProposal): 날짜별 블록 id·제목·시간
    - 구매 상담(PurchaseProposal): 품목·카테고리·결정·실제 권한 제품(이름·가격·통화·수량)
    - 그 외: changes 칸, 없으면 제안 전체"""
    if "changed_items" in proposal or "postponed_tasks" in proposal:
        return {
            "changed_items": proposal.get("changed_items", []),
            "postponed_tasks": proposal.get("postponed_tasks", []),
        }
    if "days" in proposal:
        return {"days": [
            {"date": day.get("date"),
             "blocks": [{k: b.get(k) for k in ("id", "title", "start_time", "end_time")}
                        for b in day.get("schedule_blocks", [])]}
            for day in proposal.get("days", [])
        ]}
    if "decision" in proposal and "item" in proposal:
        return {k: proposal.get(k) for k in ("decision", "item", "category", "recommended")}
    return proposal.get("changes", proposal)

def action_executor_node(state: AgentState):
    """승인된 제안을 실제 외부 시스템(Google Calendar API, SQLite DB 등)에 반영하는 노드"""
    req_type = state.get("request_type")
    proposal = state.get("draft_proposal", {})
    print(f"[실행 에이전트] 사용자 승인 확인! 외부 시스템 반영 중... (요청: {req_type})")
    
    if req_type in ("morning_briefing", "trip_planning"):
        # TODO(리드, 11/29): 지인 님 calendar_service.add_event / move_event 호출
        #   trip_planning: 여행 전체 일정(days)을 날짜별로 캘린더에 기록
        #   morning_briefing: 오늘 바뀐 일정만 반영
        # → 이후 구매 상담 때 맥락 수집이 캘린더에서 남은 일정을 다시 읽음 (에이전트 간 정보 전달)
        print(f" -> [Calendar API] 구글 캘린더 일정 반영 완료: {proposal}")
    elif req_type in ["purchase_consultation", "retrospective"]:
        # TODO: 장기 메모리 DB(long_term_memory.db) 저장 함수 호출
        print(" -> [DB] SQLite 장기 메모리에 구매/회고 데이터 기록 완료")
        
    return {"final_response": "승인하신 제안이 캘린더 및 시스템에 성공적으로 반영되었습니다."}

# ==========================================
# [2] 조건부 라우팅 함수 (Edge)
# ==========================================
def route_after_supervisor(state: AgentState) -> str:
    """Supervisor가 판별한 request_type에 따라 일반 대화는 바로 커뮤니케이터로, 업무는 맥락 수집으로 분기"""
    req_type = state.get("request_type")
    if req_type == "general":
        return "communicator"
    return "context_collector"

def route_after_context(state: AgentState) -> str:
    """Supervisor가 판별한 request_type에 따라 담당 에이전트로 이동 (AGENT_BY_REQUEST 표 하나로 관리)"""
    return AGENT_BY_REQUEST.get(state.get("request_type"), "communicator")

AGENT_BY_REQUEST = {
    "trip_planning": "planner",          # 여행 전 전체 일정 정리 (플래너 여행 전체 모드)
    "morning_briefing": "planner",
    "purchase_consultation": "purchase_advisor",
    "retrospective": "recall_interview",
}

def route_after_critic(state: AgentState) -> str:
    """판정은 critic_node가 critic_verdict로 이미 내렸으므로, 라우터는 그 값만 보고 길을 고른다.
    - "rework" → 현재 시나리오의 담당 에이전트로 재작업
    - 그 외("pass" / "warn" / "keep_original") → 커뮤니케이터 (문구는 커뮤니케이터가 verdict로 분기)
    시나리오마다 역할 에이전트는 하나뿐이므로, Critic(LLM)이 다른 시나리오의 에이전트를
    target_agent로 잘못 지정해도 따르지 않는다 (예: 구매 상담 반려인데 "planner").
    """
    if state.get("critic_verdict") != "rework":
        return "communicator"

    owner = AGENT_BY_REQUEST.get(state.get("request_type"))
    target_agent = (state.get("critic_result") or {}).get("target_agent")
    if owner and target_agent and target_agent != owner:
        print(f" -> [경고] Critic target_agent={target_agent}가 시나리오 담당({owner})과 달라 {owner}로 보냄")
    return owner or "communicator"

def is_after_rejection(state: AgentState) -> bool:
    return (state.get("user_decision") or {}).get("status") == "rejected"

def route_after_communicator(state: AgentState) -> str:
    """커뮤니케이터가 기록한 awaiting_approval만 보고 분기 (user_decision을 직접 보지 않음)"""
    return "user_approval" if state.get("awaiting_approval") else "end"

def route_after_approval(state: AgentState) -> str:
    """사용자 결정(user_decision)에 따라 분기:
    - 승인(approved): action_executor 노드로 이동하여 캘린더/DB에 반영
    - 수정 요청(modify): 사용자의 수정 피드백을 반영하기 위해 플래너(또는 해당 에이전트)로 복귀
    - 거절(rejected): 커뮤니케이터로 복귀하여 대안 텍스트 제시
    """
    decision = state.get("user_decision") or {}
    status = decision.get("status", "approved")

    if status == "approved":
        return "action_executor"
    elif status == "modify":
        # 수정 요청 시 담당 에이전트가 다시 계획을 짜고 Critic 재검증 (D6 규칙 준수)
        return AGENT_BY_REQUEST.get(state.get("request_type"), "planner")
    elif status == "rejected":
        # 단순 거절 시 커뮤니케이터로 대안 루프
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

# 엔트리 포인트 및 Supervisor 조건부 라우팅 (일반 대화 general은 context_collector 생략)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_after_supervisor, {
    "context_collector": "context_collector",
    "communicator": "communicator"
})

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

# 커뮤니케이터 후: 승인할 제안이 있을 때만 승인 단계로
workflow.add_conditional_edges("communicator", route_after_communicator, {
    "user_approval": "user_approval",
    "end": END
})

# 사용자 승인 후 라우팅: 승인이면 실행 노드로, 수정 요청이면 담당 에이전트로 재작성, 거절이면 커뮤니케이터 대안 루프
workflow.add_conditional_edges("user_approval", route_after_approval, {
    "action_executor": "action_executor",
    "planner": "planner",
    "purchase_advisor": "purchase_advisor",
    "recall_interview": "recall_interview",
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
    # as_node를 주지 않아야 재개 시 user_approval 노드가 실제로 실행되어 거절/수정 이력이 기록됨
    app.update_state(
        config,
        {"user_decision": {"status": "approved", "feedback": ""}}
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