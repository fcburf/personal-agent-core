
from datetime import datetime, timedelta

from task_service import (
    add_task,
    get_task,
    set_calendar_event_id,
    init_tasks_db,
)
from calendar_service import add_event


def add_task_with_calendar(
    title,
    start_date,
    deadline,
    estimated_minutes=None
):
    """할 일을 등록하고 Google Calendar에 마감 일정 생성"""

    init_tasks_db()

    # 1. 할 일을 로컬 DB에 저장
    task_id = add_task(
        title=title,
        start_date=start_date,
        deadline=deadline,
        estimated_minutes=estimated_minutes,
    )

    # 2. 마감 시각을 캘린더 일정으로 변환
    due = datetime.strptime(
        deadline, "%Y-%m-%d %H:%M"
    )

    # 마감 표시용 15분 일정
    event_start = due
    event_end = due + timedelta(minutes=15)

    try:
        # 3. Google Calendar에 일정 생성
        event = add_event(
            title=f"[마감] {title}",
            start=event_start.strftime("%Y-%m-%d %H:%M"),
            end=event_end.strftime("%Y-%m-%d %H:%M"),
        )

    except Exception:
        # 캘린더 연결에 실패해도 로컬 할 일은 유지
        print("캘린더 등록 실패: 할 일은 DB에 저장되었습니다.")
        raise

    # 4. 생성된 캘린더 이벤트 ID 저장
    set_calendar_event_id(task_id, event["id"])

    return get_task(task_id)
