
from datetime import date, datetime

from task_service import (
    init_tasks_db,
    add_task,
    get_tasks,
    get_task,
    get_task_schedule,
    complete_task,
)

from schedule_adjustment_service import (
    recommend_schedule_adjustment,
)

from gemini_service import estimate_task_time


def format_minutes(minutes):
    if minutes is None:
        return "미입력"

    hours, mins = divmod(int(minutes), 60)

    if hours and mins:
        return f"{hours}시간 {mins}분"
    if hours:
        return f"{hours}시간"
    return f"{mins}분"


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()

        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False

        print("y 또는 n을 입력해 주세요.")


def input_positive_minutes(prompt, allow_empty=False):
    while True:
        value = input(prompt).strip()

        if not value and allow_empty:
            return None

        try:
            minutes = int(value)

            if minutes <= 0:
                raise ValueError

            return minutes

        except ValueError:
            print("1 이상의 정수를 입력해 주세요.")


def show_tasks():
    tasks = get_tasks()

    print("\n" + "=" * 50)
    print("           현재 남은 할 일")
    print("=" * 50)

    if not tasks:
        print("등록된 할 일이 없습니다.")
        return

    for task in tasks:
        print(f"\n[{task['id']}] {task['title']}")
        print(f"  시작 날짜 : {task['start_date']}")
        print(f"  마감일    : {task['deadline']}")
        print(
            "  예상 소요 :",
            format_minutes(task["estimated_minutes"])
        )

    print("=" * 50)


def input_date(prompt, date_format, default=None):
    while True:
        value = input(prompt).strip()

        if not value and default is not None:
            return default

        try:
            datetime.strptime(value, date_format)
            return value

        except ValueError:
            print("날짜 형식이 올바르지 않습니다.")


def estimate_with_ai(title, description):
    """Gemini 추가 질문 및 작업 시간 추정"""

    print("\n[Gemini AI 작업 시간 분석]")
    print("작업 범위와 예상 소요 시간을 분석합니다.")
    print("API 호출에 시간이 걸릴 수 있습니다.")

    answers = []

    try:
        result = estimate_task_time(
            title=title,
            description=description,
            additional_answers=answers,
            allow_questions=True,
        )

        if result["needs_clarification"]:
            print("\n정확한 추정을 위해 추가 정보가 필요합니다.")

            for question in result["questions"]:
                print(f"\nAI 질문: {question}")
                answer = input("답변: ").strip()

                answers.append({
                    "question": question,
                    "answer": answer or "잘 모르겠습니다.",
                })

            print("\n답변을 반영해 작업 시간을 추정합니다.")

            result = estimate_task_time(
                title=title,
                description=description,
                additional_answers=answers,
                allow_questions=False,
            )

        if result["needs_clarification"]:
            raise ValueError("추가 질문 단계가 완료되지 않았습니다.")

        print("\n[AI 예상 작업 시간]")

        for step in result["breakdown"]:
            print(
                f"  - {step['name']}: "
                f"{format_minutes(step['minutes'])}"
            )

        minutes = result["estimated_minutes"]

        print("\n총 예상 소요:", format_minutes(minutes))
        print("추정 근거:", result["reason"])
        print("\n※ AI 추정치이므로 실제 작업 시간과 다를 수 있습니다.")

        while True:
            print("\n1. AI 예상 시간 사용")
            print("2. 예상 시간 직접 수정")
            print("3. 예상 시간 없이 등록")

            choice = input("선택: ").strip()

            if choice == "1":
                return minutes

            if choice == "2":
                return input_positive_minutes(
                    "수정할 예상 시간 (분): "
                )

            if choice == "3":
                return None

            print("1~3 중 선택해 주세요.")

    except Exception as error:
        print("\nGemini 작업 시간 추정에 실패했습니다.")
        print("오류:", error)
        print("직접 입력 방식으로 전환합니다.")

        return input_positive_minutes(
            "예상 소요 시간 (분, Enter=미입력): ",
            allow_empty=True,
        )


def recommend_new_task(task_id):
    task = get_task(task_id)

    if not task:
        print("등록된 할 일을 찾지 못했습니다.")
        return

    print("\n[컨디션 기반 일정 분석]")

    result = recommend_schedule_adjustment()

    print(result["condition"]["message"])
    print(
        "오늘의 권장 작업량:",
        format_minutes(result["available_minutes"])
    )

    if get_task_schedule(task_id):
        print("이미 작업 시간이 배정된 할 일입니다.")
        return

    recommended_ids = {
        item["id"] for item in result["recommended"]
    }

    postponed_ids = {
        item["id"] for item in result["postponed"]
    }

    if task_id in recommended_ids:
        print("분석 결과: 오늘 우선 진행을 추천합니다.")

    elif task_id in postponed_ids:
        print("분석 결과: 작업 연기를 추천합니다.")

    else:
        print(
            "현재 추천 대상에 포함되지 않았습니다. "
            "시작 날짜나 작업 조건을 확인해 주세요."
        )

    print(
        "\nGoogle Calendar의 기존 일정과 "
        "충돌하지 않는 작업 시간을 찾으려면 "
        "자동 일정 조정을 실행하세요."
    )

    if not ask_yes_no(
        "지금 자동 일정 조정을 실행할까요? (y/n): "
    ):
        print("할 일은 저장되었으며 일정 반영은 보류했습니다.")
        return

    from schedule_auto_test import main as auto_schedule_main
    auto_schedule_main()


def register_task():
    print("\n[새 할 일 등록]")

    while True:
        title = input("할 일 제목: ").strip()

        if title:
            break

        print("제목을 입력해 주세요.")

    description = input(
        "작업 설명 (범위, 완성 기준 등): "
    ).strip()

    today = date.today().isoformat()

    while True:
        start_date = input_date(
            f"시작 날짜 (YYYY-MM-DD, Enter=오늘 {today}): ",
            "%Y-%m-%d",
            default=today,
        )

        deadline = input_date(
            "마감일 (YYYY-MM-DD HH:MM): ",
            "%Y-%m-%d %H:%M",
        )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(deadline, "%Y-%m-%d %H:%M")

        if end <= datetime.now():
            print("마감일은 현재 시각 이후여야 합니다.")
            continue

        if start > end:
            print("시작 날짜가 마감일보다 늦습니다.")
            continue

        break

    print("\n[예상 작업 시간 설정]")
    print("1. Gemini AI가 추정")
    print("2. 직접 입력")
    print("3. 미입력")

    while True:
        choice = input("선택: ").strip()

        if choice == "1":
            estimated_minutes = estimate_with_ai(
                title, description
            )
            break

        if choice == "2":
            estimated_minutes = input_positive_minutes(
                "예상 소요 시간 (분): "
            )
            break

        if choice == "3":
            estimated_minutes = None
            break

        print("1~3 중 선택해 주세요.")

    print("\n[등록 내용 확인]")
    print(f"제목       : {title}")
    print(f"작업 설명  : {description or '없음'}")
    print(f"시작 날짜  : {start_date}")
    print(f"마감일     : {deadline}")
    print(
        f"예상 소요  : {format_minutes(estimated_minutes)}"
    )

    if not ask_yes_no("\n등록할까요? (y/n): "):
        print("등록을 취소했습니다.")
        return

    try:
        task_id = add_task(
            title=title,
            start_date=start_date,
            deadline=deadline,
            estimated_minutes=estimated_minutes,
        )

        print(f"\n할 일 등록 성공! ID: {task_id}")

        if description:
            print(
                "참고: 작업 설명은 AI 추정에 사용되며 "
                "현재 DB에는 별도로 저장되지 않습니다."
            )

    except ValueError as error:
        print(f"\n등록 실패: {error}")
        return

    try:
        recommend_new_task(task_id)

    except Exception as error:
        print(f"\n일정 추천 중 오류: {error}")
        print("할 일 자체는 정상적으로 저장되었습니다.")



def finish_task():
    """여러 할 일 완료 처리 및 선택적 캘린더 일정 정리"""
    show_tasks()
    value = input(
        "\n완료할 할 일 ID (여러 개는 쉼표로 구분, 예: 2, 3, 5): "
    ).strip()
    if not value:
        print("완료할 할 일을 입력하지 않았습니다.")
        return
    try:
        task_ids = [int(part.strip()) for part in value.split(",")]
        if any(task_id <= 0 for task_id in task_ids):
            raise ValueError
    except ValueError:
        print("올바른 ID를 입력해 주세요. 예: 2, 3, 5")
        return
    task_ids = list(dict.fromkeys(task_ids))
    valid_tasks = []
    invalid_ids = []
    for task_id in task_ids:
        task = get_task(task_id)
        if task and not task["completed"]:
            valid_tasks.append(task)
        else:
            invalid_ids.append(task_id)
    if not valid_tasks:
        print("완료 처리할 수 있는 할 일이 없습니다.")
        return
    print("\n[완료 처리 대상]")
    for task in valid_tasks:
        print(f"  [{task['id']}] {task['title']}")
    if invalid_ids:
        print("\n존재하지 않거나 이미 완료된 ID:", ", ".join(map(str, invalid_ids)))
    if not ask_yes_no("\n위 할 일을 모두 완료 처리할까요? (y/n): "):
        print("완료 처리를 취소했습니다.")
        return

    succeeded = []
    failed = []
    for task in valid_tasks:
        if complete_task(task["id"]):
            succeeded.append(task)
        else:
            failed.append(task["id"])
    print(f"\n완료 처리 성공: {len(succeeded)}개")
    if failed:
        print("완료 처리 실패 ID:", failed)
    if not succeeded:
        return

    # No Calendar API call until the user explicitly opts in.
    if not ask_yes_no("\n연결된 Google Calendar 일정도 삭제할까요? (y/n): "):
        print("캘린더 일정은 그대로 유지합니다.")
        return

    from schedule_apply_service import delete_completed_task_calendar_event
    counts = {"deleted": 0, "already_missing": 0, "no_schedule": 0}
    errors = []
    for task in succeeded:
        try:
            status = delete_completed_task_calendar_event(task["id"])
            counts[status] += 1
            print(f"  [{task['id']}] {task['title']}: {status}")
        except Exception as error:
            errors.append((task["id"], str(error)))
            print(f"  [{task['id']}] 캘린더 정리 실패: {error}")
    print("\n[캘린더 정리 결과]")
    print(f"삭제 성공: {counts['deleted']}개")
    print(f"이미 삭제됨: {counts['already_missing']}개")
    print(f"연결된 일정 없음: {counts['no_schedule']}개")
    print(f"정리 실패: {len(errors)}개")
    if errors:
        print("정리 실패한 일정의 연결 기록은 유지했습니다. 재시도 전에 확인해 주세요.")


def main():
    init_tasks_db()

    while True:
        print("\n" + "=" * 50)
        print("           개인 에이전트 할 일 관리")
        print("=" * 50)
        print("1. 새 할 일 등록")
        print("2. 남은 할 일 조회")
        print("3. 할 일 완료 처리")
        print("4. 자동 일정 조정")
        print("0. 종료")

        choice = input("\n메뉴 선택: ").strip()

        if choice == "1":
            register_task()

        elif choice == "2":
            show_tasks()

        elif choice == "3":
            finish_task()

        elif choice == "4":
            from schedule_auto_test import main as auto_schedule_main
            auto_schedule_main()

        elif choice == "0":
            print("\n프로그램을 종료합니다.")
            break

        else:
            print("올바른 메뉴 번호를 입력해 주세요.")


if __name__ == "__main__":
    main()
