
from datetime import datetime, date, timedelta

from task_service import get_tasks, init_tasks_db
from condition_service import get_condition


def analyze_condition(condition):
    """컨디션에 따른 작업량 비율 계산"""

    if not condition:
        return {
            "level": "unknown",
            "capacity": 1.0,
            "message": "컨디션 기록이 없어 기본 작업량을 적용합니다."
        }

    fatigue = condition["fatigue_level"]
    sleep_type = condition["sleep_type"]
    sleep_minutes = condition["sleep_minutes"]
    sleep_score = condition["sleep_score"]

    # 밤샘 또는 극심한 피로
    if sleep_type == "all_nighter" or fatigue == 5:
        return {
            "level": "very_bad",
            "capacity": 0.3,
            "message": "휴식이 우선입니다. 필수 작업만 진행하세요."
        }

    # 수면 부족 여부
    poor_sleep = (
        (sleep_minutes is not None and sleep_minutes < 360)
        or (sleep_score is not None and sleep_score < 60)
    )

    if fatigue == 4 or poor_sleep:
        return {
            "level": "bad",
            "capacity": 0.6,
            "message": "컨디션이 좋지 않아 작업량 조정을 권장합니다."
        }

    if fatigue == 3:
        return {
            "level": "normal",
            "capacity": 0.8,
            "message": "무리하지 않는 범위에서 작업을 진행하세요."
        }

    return {
        "level": "good",
        "capacity": 1.0,
        "message": "컨디션이 양호합니다. 계획대로 진행해도 좋습니다."
    }


def calculate_task_priority(task, now=None):
    """마감까지 남은 시간에 따라 우선순위 계산"""

    now = now or datetime.now()

    deadline = datetime.strptime(
        task["deadline"], "%Y-%m-%d %H:%M"
    )

    remaining = deadline - now
    remaining_hours = remaining.total_seconds() / 3600

    if remaining_hours < 0:
        return 100, "마감 초과"

    if remaining_hours <= 24:
        return 90, "24시간 이내 마감"

    if remaining_hours <= 72:
        return 70, "3일 이내 마감"

    if remaining_hours <= 168:
        return 50, "7일 이내 마감"

    return 20, "마감 여유"


def recommend_schedule_adjustment(
    base_minutes=240,
    today=None,
    now=None,
):
    """컨디션과 마감일을 바탕으로 작업 일정 추천"""

    today = today or date.today()
    now = now or datetime.now()

    init_tasks_db()

    # 오늘의 컨디션 확인
    condition = get_condition(today.isoformat())
    analysis = analyze_condition(condition)

    capacity = analysis["capacity"]
    available_minutes = int(base_minutes * capacity)

    # 완료되지 않은 할 일 조회
    tasks = get_tasks()

    ranked_tasks = []

    for task in tasks:

        start_date = date.fromisoformat(
            task["start_date"]
        )

        # 미래에 시작하는 작업도 후보에 포함
        is_future_task = start_date > today

        priority, reason = calculate_task_priority(
            task, now
        )

        estimated_minutes = task["estimated_minutes"]

        # 예상 소요 시간이 없으면 60분으로 계산
        if estimated_minutes is None:
            estimated_minutes = 60

        ranked_tasks.append({
            **task,
            "priority": priority,
            "reason": reason,
            "estimated_minutes": estimated_minutes,
            "is_future_task": is_future_task,
        })

    # 우선순위가 높고 마감이 빠른 순서로 정렬
    ranked_tasks.sort(
        key=lambda task: (
            -task["priority"],
            task["deadline"],
        )
    )

    recommended = []
    postponed = []

    used_minutes = 0

    for task in ranked_tasks:

        duration = task["estimated_minutes"]

        # 1. 미래에 시작하는 작업
        # 오늘 작업량에 포함하지 않고 시작일부터 배정
        if task["is_future_task"]:
            postponed.append({
                **task,
                "suggested_date": task["start_date"],
            })
            continue

        # 2. 마감 임박 또는 초과 작업
        urgent = task["priority"] >= 90

        if urgent:
            recommended.append(task)
            used_minutes += duration
            continue

        # 3. 오늘 권장 작업량 이내라면 추천
        if used_minutes + duration <= available_minutes:
            recommended.append(task)
            used_minutes += duration

        # 4. 권장 작업량 초과 시 연기 후보
        else:
            postponed.append({
                **task,
                "suggested_date": (
                    today + timedelta(days=1)
                ).isoformat()
            })

    return {
        "condition": analysis,
        "base_minutes": base_minutes,
        "available_minutes": available_minutes,
        "used_minutes": used_minutes,
        "recommended": recommended,
        "postponed": postponed,
    }
