
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

from schedule_adjustment_service import recommend_schedule_adjustment
from schedule_apply_service import apply_task_schedule
from task_service import get_task_schedule, is_task_schedule_paused, get_paused_task_schedules
from calendar_service import get_events_between
from calendar_sync_service import sync_calendar_schedules

KST = ZoneInfo("Asia/Seoul")

WORK_START = time(9, 0)
WORK_END = time(22, 0)
SLOT_STEP = timedelta(minutes=15)


def format_minutes(minutes):
    hours, mins = divmod(minutes, 60)
    return f"{hours}시간 {mins}분"


def to_datetime(value):
    return datetime.strptime(
        value, "%Y-%m-%d %H:%M"
    ).replace(tzinfo=KST)


def round_up_to_slot(dt):
    """시간을 다음 15분 단위로 올림"""
    dt = dt.replace(second=0, microsecond=0)

    remainder = dt.minute % 15

    if remainder:
        dt += timedelta(minutes=15 - remainder)

    return dt


def overlaps(start, end, busy_start, busy_end):
    """두 시간 구간이 겹치는지 확인"""
    return start < busy_end and end > busy_start


def find_available_slot(
    target_date,
    duration_minutes,
    deadline,
    busy_intervals,
):
    """기존 일정과 겹치지 않는 작업 시간 탐색"""

    start = datetime.combine(
        target_date, WORK_START, tzinfo=KST
    )
    day_end = datetime.combine(
        target_date, WORK_END, tzinfo=KST
    )

    now = datetime.now(KST)

    if target_date == now.date():
        start = max(start, round_up_to_slot(now))

    duration = timedelta(minutes=duration_minutes)

    while start + duration <= day_end:
        end = start + duration

        if end > deadline:
            return None

        conflict = next(
            (
                (busy_start, busy_end)
                for busy_start, busy_end in busy_intervals
                if overlaps(
                    start, end, busy_start, busy_end
                )
            ),
            None,
        )

        if conflict is None:
            return start, end

        # 충돌 일정이 끝난 다음 시간부터 다시 탐색
        start = round_up_to_slot(
            max(start + SLOT_STEP, conflict[1])
        )

    return None


def build_proposals(result, excluded_task_ids=None):
    """컨디션과 기존 캘린더를 고려한 작업 배정"""

    excluded_task_ids = set(excluded_task_ids or [])

    today = datetime.now(KST).date()
    available_minutes = result["available_minutes"]

    proposals = []
    occupied_ids = set()
    candidates = []

    for task in result["recommended"]:
        task_id = task["id"]
        occupied_ids.add(task_id)

        # 사용자가 재배정을 거부한 작업은 제외
        if task_id in excluded_task_ids:
            continue

        if is_task_schedule_paused(task_id) or get_task_schedule(task_id):
            continue

        candidates.append({
            "task": task,
            "target_date": today,
            "type": "recommended",
        })

    for task in result["postponed"]:
        task_id = task["id"]

        if task_id in occupied_ids:
            continue

        if task_id in excluded_task_ids:
            continue

        if is_task_schedule_paused(task_id) or get_task_schedule(task_id):
            continue

        candidates.append({
            "task": task,
            "target_date": date.fromisoformat(
                task["suggested_date"]
            ),
            "type": "postponed",
        })

    if not candidates:
        return []

    # 조회 기간은 후보 작업의 가장 늦은 마감일까지
    latest_deadline = max(
        to_datetime(item["task"]["deadline"])
        for item in candidates
    )

    now = datetime.now(KST)

    if latest_deadline <= now:
        return []

    calendar_events = get_events_between(
        now,
        latest_deadline + timedelta(seconds=1),
    )

    busy_intervals = [
        (event["start"], event["end"])
        for event in calendar_events
    ]

    today_used_minutes = 0

    for item in candidates:
        task = item["task"]
        duration = task.get("estimated_minutes") or 60
        deadline = to_datetime(task["deadline"])

        # 작업 시작 날짜 이전에는 배정하지 않음
        task_start_date = date.fromisoformat(
            task["start_date"]
        )

        target_date = max(
            item["target_date"],
            task_start_date,
            today,
        )

        # 권장 작업량을 넘는 작업은 다음 날부터 탐색
        if (
            target_date == today
            and today_used_minutes + duration
            > available_minutes
        ):
            target_date += timedelta(days=1)

        while target_date <= deadline.date():
            slot = find_available_slot(
                target_date,
                duration,
                deadline,
                busy_intervals,
            )

            if slot:
                start, end = slot

                proposals.append({
                    "task": task,
                    "start": start.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "end": end.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "type": item["type"],
                })

                # 이번 실행에서 배정한 시간도 점유 처리
                busy_intervals.append((start, end))

                if target_date == today:
                    today_used_minutes += duration

                break

            target_date += timedelta(days=1)

    return proposals


def main():
    print("=" * 55)
    print("       개인 에이전트 자동 일정 조정")
    print("=" * 55)

    # -----------------------------------------
    # 1. Google Calendar 동기화
    # -----------------------------------------

    print("\n[Google Calendar 동기화]")
    print("캘린더의 변경 사항을 확인합니다.")

    try:
        sync_result = sync_calendar_schedules()

    except Exception as error:
        print(f"\n동기화 실패: {error}")
        print("안전을 위해 자동 일정 조정을 중단합니다.")
        return

    print("\n[동기화 결과]")
    print(f"확인한 작업: {sync_result['checked']}개")
    print(f"삭제 감지: {len(sync_result['deleted'])}개")
    print(f"시간 변경: {len(sync_result['updated'])}개")
    print(f"변경 없음: {len(sync_result['unchanged'])}개")
    print(f"오류: {len(sync_result['errors'])}개")

    # 일부 일정 조회에 실패하면 자동 조정 중단
    if sync_result["errors"]:
        print("\n[동기화 오류]")

        for item in sync_result["errors"]:
            print(
                f"- {item['title']}: "
                f"{item['reason']}"
            )

        print(
            "\n일부 작업의 동기화에 실패했습니다."
        )
        print(
            "안전을 위해 자동 일정 조정을 중단합니다."
        )
        return

    # -----------------------------------------
    # 2. 삭제된 작업은 영구 보류 상태로 기록됨
    # -----------------------------------------
    excluded_task_ids = set()

    if sync_result["deleted"]:
        print("\n[캘린더에서 삭제된 작업 - 자동 재등록 보류]")
        for title in sync_result["deleted"]:
            print(f"- {title}")
        print("삭제된 일정은 사용자 요청 전까지 다시 배정하지 않습니다.")

    paused = [item for item in get_paused_task_schedules() if not item["completed"]]
    if paused:
        print(f"\n현재 자동 재등록 보류 중인 미완료 작업: {len(paused)}개")

    # -----------------------------------------
    # 3. 컨디션 기반 일정 분석
    # -----------------------------------------

    print("\n[컨디션 기반 일정 분석]")

    try:
        result = recommend_schedule_adjustment()

    except Exception as error:
        print(f"\n일정 분석 실패: {error}")
        return

    print("\n[오늘의 컨디션]")
    print(result["condition"]["message"])

    print("\n[오늘의 작업량]")
    print(
        "권장 작업량:",
        format_minutes(result["available_minutes"])
    )

    # -----------------------------------------
    # 4. Google Calendar 일정 충돌 확인
    # -----------------------------------------

    print(
        "\nGoogle Calendar의 기존 일정을 확인합니다."
    )

    try:
        proposals = build_proposals(
            result,
            excluded_task_ids=excluded_task_ids,
        )

    except Exception as error:
        print(f"\n일정 추천 실패: {error}")
        print(
            "캘린더 조회 또는 일정 계산을 확인해 주세요."
        )
        return

    if not proposals:
        print("\n배정 가능한 새로운 작업이 없습니다.")
        return

    # -----------------------------------------
    # 5. 자동 일정 조정 제안
    # -----------------------------------------

    print("\n[자동 일정 조정 제안]")

    for index, item in enumerate(proposals, 1):
        task = item["task"]

        label = (
            "우선 진행"
            if item["type"] == "recommended"
            else "연기 추천"
        )

        print(f"\n{index}. {task['title']}")
        print(f"   구분: {label}")
        print(f"   시작: {item['start']}")
        print(f"   종료: {item['end']}")
        print(f"   마감: {task['deadline']}")

    print(
        "\n※ 기존 Google Calendar 일정과의 "
        "충돌을 검사했습니다."
    )

    # -----------------------------------------
    # 6. 사용자 승인
    # -----------------------------------------

    answer = input(
        "\n추천 일정을 Google Calendar에 반영할까요? "
        "(y/n): "
    ).strip().lower()

    if answer != "y":
        print("\n일정 변경을 취소했습니다.")
        return

    # -----------------------------------------
    # 7. Google Calendar 일정 반영
    # -----------------------------------------

    print("\n[일정 반영 결과]")

    success_count = 0
    fail_count = 0

    for item in proposals:
        task = item["task"]

        try:
            event = apply_task_schedule(
                task["id"],
                item["start"],
                item["end"],
            )

            print(
                f"성공: {task['title']} "
                f"({event['id']})"
            )

            success_count += 1

        except Exception as error:
            print(
                f"실패: {task['title']} "
                f"- {error}"
            )

            fail_count += 1

    print("\n[최종 결과]")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")

    print("\n일정 반영 작업이 종료되었습니다.")


if __name__ == "__main__":
    main()
