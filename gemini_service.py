
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from config import llm_large


class WorkStep(BaseModel):
    name: str = Field(description="세부 작업 이름")
    minutes: int = Field(ge=1, description="예상 소요 시간(분)")


class TaskEstimate(BaseModel):
    needs_clarification: bool = Field(
        description="작업 시간 추정에 추가 정보가 필요한지"
    )
    questions: List[str] = Field(
        default_factory=list,
        description="추가 질문, 최대 2개"
    )
    breakdown: List[WorkStep] = Field(
        default_factory=list,
        description="세부 작업별 예상 시간"
    )
    estimated_minutes: Optional[int] = Field(
        default=None,
        description="총 예상 작업 시간(분)"
    )
    reason: str = Field(
        description="추정 근거와 불확실성"
    )


def estimate_task_time(
    title: str,
    description: str = "",
    additional_answers=None,
    allow_questions: bool = True,
):
    """Gemini로 작업 시간을 추정한다. DB/캘린더는 변경하지 않는다."""

    if not title.strip():
        raise ValueError("작업 제목이 필요합니다.")

    answers = additional_answers or []

    prompt = f"""
당신은 개인 일정 관리 에이전트의 작업 시간 추정 전문가입니다.

[작업 제목]
{title}

[작업 설명]
{description or "추가 설명 없음"}

[추가 질문과 사용자 답변]
{answers}

[규칙]
1. 작업의 실제 제작 범위와 난이도를 고려하세요.
2. 정보가 너무 부족하고 추가 질문이 허용된다면
   구체적이고 답하기 쉬운 질문을 최대 2개 하세요.
3. 추가 질문이 허용되지 않으면 현재 정보로 추정하세요.
   부족한 정보는 합리적인 가정으로 처리하고 그 가정을 밝히세요.
4. 세부 작업을 나누고 각각 분 단위로 추정하세요.
5. 총 예상 시간은 세부 작업 시간의 합계로 계산합니다.
6. 수면 점수나 피로도 때문에 기본 작업 시간을 임의로
   늘리거나 줄이지 마세요.
7. 시간 추정은 보장된 결과가 아닙니다.
8. 질문이 필요한 경우 breakdown은 빈 목록으로 반환하세요.
9. 질문이 필요하지 않으면 breakdown에 최소 1개 항목을 넣으세요.
10. 사용자의 실제 숙련도나 작업 환경을 알 수 없다면
    임의로 단정하지 말고 불확실성을 설명하세요.

[추가 질문 허용 여부]
{allow_questions}
"""

    llm = llm_large.with_structured_output(TaskEstimate)
    result = llm.invoke([HumanMessage(content=prompt)])

    if result.needs_clarification and allow_questions:
        questions = [
            q.strip() for q in result.questions
            if isinstance(q, str) and q.strip()
        ][:2]

        if questions:
            return {
                "needs_clarification": True,
                "questions": questions,
                "estimated_minutes": None,
                "breakdown": [],
                "reason": result.reason,
            }

    if not result.breakdown:
        raise ValueError(
            "Gemini가 세부 작업 시간을 반환하지 않았습니다."
        )

    breakdown = [
        {"name": step.name, "minutes": step.minutes}
        for step in result.breakdown
    ]

    total = sum(step["minutes"] for step in breakdown)

    if total <= 0:
        raise ValueError("예상 작업 시간이 올바르지 않습니다.")

    return {
        "needs_clarification": False,
        "questions": [],
        "estimated_minutes": total,
        "breakdown": breakdown,
        "reason": result.reason,
    }
