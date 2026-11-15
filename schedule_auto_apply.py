
from datetime import datetime, date, timedelta

from schedule_adjustment_service import (
    recommend_schedule_adjustment,
)
from schedule_apply_service import apply_task_schedule
from task_service import get_task_schedule


def format_minutes(minutes):
    hours, mins = divmod(minutes, 60)
    return f"{hours}시간 {mins}분"


def find_start_time(target_date, duration_minutes):
    """테스트용 작업 시작 시간 계산"""

    now = datetime.now()
    start = datetime.combine(
        target_date,
        datetime.strptime("14:00", "%H:%M").time(),
    )

    if target_date == now.date() and start <= now:
        start = now.replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(hours=1)

    end = start + timedelta(minutes=duration_minutes)

    # 작업이 자정을 넘으면 다음 날로 연기
    if end.date() != target_date:
        return None

    return start, end


def build_proposals(result):
    """추천 결과를 실제 작업 시간 후보로 변환"""

    proposals = []
    occupied = set()

    # 오늘 진행할 작업
    for task in result["recommended"]:
        occupied.add(task["id"])

        # 이미 작업 시간이 정해진 작업은 중복 배정 방지
        if get_task_schedule(task["id"]):
            continue

        proposals.append({
            "task": task,
            "target_date": date.today(),
            "type": "recommended",
        })

    # 연기할 작업
    for task in result["postponed"]:
        if task["id"] in occupied:
            continue

        proposals.append({
            "task": task,
            "target_date": date.fromisoformat(
                task["suggested_date"]
            ),
            "type": "postponed",
        })

    # 같은 날의 작업이 겹치지 않도록 순서대로 배치
    day_end_times = {}
    scheduled = []

    for proposal in proposals:
        task = proposal["task"]
        duration = task["estimated_minutes"]
        target_date = proposal["target_date"]

        deadline = datetime.strptime(
            task["deadline"], "%Y-%m-%d %H:%M"
        )

        # 추천일에 배치할 수 없다면 마감일까지 탐색
        while target_date <= deadline.date():
            slot = find_start_time(target_date, duration)

            if slot is None:
                target_date += timedelta(days=1)
                continue

            start, end = slot

            # 이번 실행에서 먼저 배정한 작업과 겹치지 않게 함
            previous_end = day_end_times.get(target_date)

            if previous_end and start < previous_end:
                start = previous_end
                end = start + timedelta(minutes=duration)

            if (
                end <= deadline
                and end.date() == target_date
            ):
                day_end_times[target_date] = end

                scheduled.append({
                    "task": task,
                    "start": start.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "end": end.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "type": proposal["type"],
                })
                break

            target_date += timedelta(days=1)

    return scheduled


def main():
    print("=" * 55)
    print("       개인 에이전트 자동 일정 조정")
    print("=" * 55)

    result = recommend_schedule_adjustment()

    print("\n[오늘의 컨디션]")
    print(result["condition"]["message"])

    print("\n[오늘의 작업량]")
    print(
        "권장 작업량:",
        format_minutes(result["available_minutes"])
    )

    proposals = build_proposals(result)

    if not proposals:
        print("\n새롭게 배정할 작업이 없습니다.")
        return

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
        "충돌 여부는 아직 검사하지 않습니다."
    )

    answer = input(
        "\n추천 일정을 Google Calendar에 반영할까요? "
        "(y/n): "
    ).strip().lower()

    if answer != "y":
        print("\n일정 변경을 취소했습니다.")
        return

    print("\n[일정 반영 결과]")

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

        except Exception as error:
            print(
                f"실패: {task['title']} "
                f"- {error}"
            )

    print("\n일정 반영 작업이 종료되었습니다.")


if __name__ == "__main__":
    main()
