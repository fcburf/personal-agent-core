
from datetime import datetime
from googleapiclient.errors import HttpError

from calendar_service import (
    get_calendar_service,
    parse_local_datetime,
)
from task_service import (
    get_task,
    get_task_schedule,
    save_task_schedule,
)

TIMEZONE = "Asia/Seoul"


def apply_task_schedule(task_id, start, end):
    """승인된 작업 예정 시간을 캘린더에 반영"""

    task = get_task(task_id)

    if not task:
        raise ValueError("해당 할 일을 찾을 수 없습니다.")

    if task["completed"]:
        raise ValueError("이미 완료한 작업입니다.")

    start_dt = parse_local_datetime(start)
    end_dt = parse_local_datetime(end)
    deadline_dt = parse_local_datetime(task["deadline"])

    if end_dt <= start_dt:
        raise ValueError("종료 시간은 시작 시간보다 늦어야 합니다.")

    if end_dt > deadline_dt:
        raise ValueError("작업 종료 시간이 마감일을 넘습니다.")

    if start_dt < datetime.now(start_dt.tzinfo):
        raise ValueError("과거 시간으로 작업을 배치할 수 없습니다.")

    schedule = get_task_schedule(task_id)
    service = get_calendar_service()

    event_body = {
        "summary": f"[작업] {task['title']}",
        "description": (
            f"개인 에이전트 작업 ID: {task_id}\n"
            f"마감: {task['deadline']}\n"
            "컨디션 기반 작업 일정"
        ),
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": TIMEZONE,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": TIMEZONE,
        },
        "extendedProperties": {
            "private": {
                "personal_agent_task_id": str(task_id),
                "personal_agent_type": "work_block",
            }
        },
    }

    event_id = (
        schedule["calendar_event_id"]
        if schedule else None
    )

    if event_id:
        # 에이전트가 생성한 작업 일정인지 확인
        existing = service.events().get(
            calendarId="primary",
            eventId=event_id,
        ).execute()

        properties = existing.get(
            "extendedProperties", {}
        ).get("private", {})

        if (
            properties.get("personal_agent_task_id")
            != str(task_id)
            or properties.get("personal_agent_type")
            != "work_block"
        ):
            raise ValueError(
                "에이전트가 관리하는 작업 일정이 아닙니다."
            )

        # 기존 이벤트의 다른 정보는 유지
        existing.update({
            "summary": event_body["summary"],
            "description": event_body["description"],
            "start": event_body["start"],
            "end": event_body["end"],
        })

        event = service.events().update(
            calendarId="primary",
            eventId=event_id,
            body=existing,
        ).execute()

    else:
        event = service.events().insert(
            calendarId="primary",
            body=event_body,
        ).execute()

    save_task_schedule(
        task_id,
        start,
        end,
        event["id"],
    )

    return event

def delete_completed_task_calendar_event(task_id):
    """Delete only the verified agent-managed Calendar event for a completed task.

    Returns: deleted, already_missing, no_schedule, or a failure via exception.
    Local schedule linkage is cleared only after confirmed deletion/absence.
    """
    from task_service import (
        get_task, get_task_schedule, delete_task_schedule,
        set_calendar_event_id,
    )

    task = get_task(task_id)
    if not task:
        raise ValueError(f"작업 ID {task_id}가 존재하지 않습니다.")
    if not task["completed"]:
        raise ValueError("완료되지 않은 작업의 일정은 이 함수로 삭제할 수 없습니다.")

    schedule = get_task_schedule(task_id)
    event_id = (schedule or {}).get("calendar_event_id") or task.get("calendar_event_id")
    if not event_id:
        return "no_schedule"

    service = get_calendar_service()
    try:
        event = service.events().get(
            calendarId="primary", eventId=event_id
        ).execute()
    except HttpError as error:
        if error.resp.status in (404, 410):
            delete_task_schedule(task_id)
            set_calendar_event_id(task_id, None)
            return "already_missing"
        raise

    if event.get("status") == "cancelled":
        delete_task_schedule(task_id)
        set_calendar_event_id(task_id, None)
        return "already_missing"

    properties = event.get("extendedProperties", {}).get("private", {})
    if (properties.get("personal_agent_task_id") != str(task_id)
            or properties.get("personal_agent_type") != "work_block"):
        raise ValueError("연결된 이벤트가 해당 작업의 에이전트 일정이 아닙니다. 삭제하지 않았습니다.")

    try:
        service.events().delete(
            calendarId="primary", eventId=event_id
        ).execute()
    except HttpError as error:
        if error.resp.status not in (404, 410):
            raise
        status = "already_missing"
    else:
        status = "deleted"

    delete_task_schedule(task_id)
    set_calendar_event_id(task_id, None)
    return status

