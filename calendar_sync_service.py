
from datetime import datetime
from googleapiclient.errors import HttpError

from calendar_service import (
    get_calendar_service,
    parse_local_datetime,
)
from task_service import (
    get_all_task_schedules,
    delete_task_schedule,
    save_task_schedule,
    pause_task_schedule,
)

CALENDAR_ID = "primary"


def get_calendar_event(service, event_id):
    """Google Calendar에서 특정 이벤트 조회"""

    try:
        return service.events().get(
            calendarId=CALENDAR_ID,
            eventId=event_id,
        ).execute()

    except HttpError as error:
        # 404 또는 410: 이벤트가 삭제되었거나 더 이상 존재하지 않음
        if error.resp.status in (404, 410):
            return None

        # 인증 오류, 네트워크 문제 등은 삭제로 판단하지 않음
        raise


def get_event_time(event, field):
    """Google Calendar 이벤트의 시작 또는 종료 시간 변환"""

    value = event.get(field, {})

    # 시간이 지정된 이벤트만 작업 일정으로 처리
    date_time = value.get("dateTime")

    if not date_time:
        return None

    return datetime.fromisoformat(
        date_time.replace("Z", "+00:00")
    )


def sync_calendar_schedules():
    """Google Calendar 변경 사항을 로컬 작업 DB에 반영"""

    schedules = get_all_task_schedules()

    result = {
        "checked": 0,
        "deleted": [],
        "updated": [],
        "unchanged": [],
        "errors": [],
    }

    if not schedules:
        return result

    service = get_calendar_service()

    for schedule in schedules:
        task_id = schedule["task_id"]
        title = schedule["title"]
        event_id = schedule["calendar_event_id"]

        result["checked"] += 1

        if not event_id:
            result["errors"].append({
                "title": title,
                "reason": "연결된 캘린더 이벤트 ID가 없습니다.",
            })
            continue

        try:
            event = get_calendar_event(service, event_id)

            # 캘린더에서 삭제된 경우
            if event is None or event.get("status") == "cancelled":
                pause_task_schedule(task_id)
                delete_task_schedule(task_id)

                result["deleted"].append(title)
                continue

            # 에이전트가 생성한 이벤트인지 확인
            properties = event.get(
                "extendedProperties", {}
            ).get("private", {})

            if (
                properties.get("personal_agent_task_id") != str(task_id)
                or properties.get("personal_agent_type") != "work_block"
            ):
                result["errors"].append({
                    "title": title,
                    "reason": "에이전트가 관리하는 이벤트가 아닙니다.",
                })
                continue

            start = get_event_time(event, "start")
            end = get_event_time(event, "end")

            if start is None or end is None or end <= start:
                result["errors"].append({
                    "title": title,
                    "reason": "이벤트 시간이 올바르지 않습니다.",
                })
                continue

            old_start = parse_local_datetime(
                schedule["scheduled_start"]
            )
            old_end = parse_local_datetime(
                schedule["scheduled_end"]
            )

            # 시간대까지 고려해 실제 시각 비교
            if start != old_start or end != old_end:
                new_start = start.astimezone(
                    old_start.tzinfo
                ).strftime("%Y-%m-%d %H:%M")

                new_end = end.astimezone(
                    old_end.tzinfo
                ).strftime("%Y-%m-%d %H:%M")

                save_task_schedule(
                    task_id,
                    new_start,
                    new_end,
                    event_id,
                )

                result["updated"].append({
                    "title": title,
                    "start": new_start,
                    "end": new_end,
                })

            else:
                result["unchanged"].append(title)

        except Exception as error:
            result["errors"].append({
                "title": title,
                "reason": str(error),
            })

    return result
