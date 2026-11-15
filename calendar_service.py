
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

KST = ZoneInfo("Asia/Seoul")

# 일정 읽기와 수정 권한
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def get_calendar_service():
    """Google Calendar에 인증하고 연결한다."""
    creds = None

    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(TOKEN_FILE), SCOPES
            )
            # 기존 읽기 전용 토큰은 다시 인증
            if not creds.has_scopes(SCOPES):
                creds = None
        except (ValueError, KeyError):
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(
            creds.to_json(), encoding="utf-8"
        )

    return build("calendar", "v3", credentials=creds)


def parse_local_datetime(value):
    """YYYY-MM-DD HH:MM 문자열을 한국 시간으로 변환."""
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=KST)


def get_events(days=30):
    """앞으로 지정된 일수만큼 캘린더 일정 조회."""
    service = get_calendar_service()

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)

    events = []
    page_token = None

    while True:
        response = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=250,
            pageToken=page_token,
        ).execute()

        for event in response.get("items", []):
            start = event["start"]

            if "dateTime" in start:
                dt = datetime.fromisoformat(
                    start["dateTime"].replace("Z", "+00:00")
                )
                start_text = dt.astimezone(KST).strftime(
                    "%Y-%m-%d %H:%M"
                )
                all_day = False
            else:
                start_text = start["date"]
                all_day = True

            events.append({
                "id": event["id"],
                "title": event.get("summary", "제목 없음"),
                "start": start_text,
                "all_day": all_day,
                "status": event.get("status", "confirmed"),
            })

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return events


def add_event(title, start, end):
    """시작·종료 시간을 지정해 새 일정 생성."""
    start_dt = parse_local_datetime(start)
    end_dt = parse_local_datetime(end)

    if end_dt <= start_dt:
        raise ValueError("종료 시간은 시작 시간보다 늦어야 합니다.")

    service = get_calendar_service()

    event_body = {
        "summary": title,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": "Asia/Seoul",
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Asia/Seoul",
        },
    }

    return service.events().insert(
        calendarId="primary",
        body=event_body,
    ).execute()


def move_event(event_id, new_start, new_end):
    """기존 일정의 시작·종료 시간을 변경."""
    start_dt = parse_local_datetime(new_start)
    end_dt = parse_local_datetime(new_end)

    if end_dt <= start_dt:
        raise ValueError("종료 시간은 시작 시간보다 늦어야 합니다.")

    service = get_calendar_service()

    # 기존 이벤트 정보를 가져와 다른 필드 보존
    event = service.events().get(
        calendarId="primary",
        eventId=event_id,
    ).execute()

    event["start"] = {
        "dateTime": start_dt.isoformat(),
        "timeZone": "Asia/Seoul",
    }
    event["end"] = {
        "dateTime": end_dt.isoformat(),
        "timeZone": "Asia/Seoul",
    }

    return service.events().update(
        calendarId="primary",
        eventId=event_id,
        body=event,
    ).execute()


def get_events_between(start, end):
    """지정된 시간 범위와 겹치는 Google Calendar 일정 조회"""
    service = get_calendar_service()

    events = []
    page_token = None

    while True:
        response = service.events().list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=250,
            pageToken=page_token,
        ).execute()

        for event in response.get("items", []):
            if event.get("status") == "cancelled":
                continue

            start_data = event["start"]
            end_data = event["end"]

            if "dateTime" in start_data:
                event_start = datetime.fromisoformat(
                    start_data["dateTime"].replace("Z", "+00:00")
                ).astimezone(KST)

                event_end = datetime.fromisoformat(
                    end_data["dateTime"].replace("Z", "+00:00")
                ).astimezone(KST)
            else:
                # 종일 일정도 해당 날짜 전체를 차지하도록 처리
                event_start = datetime.combine(
                    date.fromisoformat(start_data["date"]),
                    datetime.min.time(),
                    tzinfo=KST,
                )
                event_end = datetime.combine(
                    date.fromisoformat(end_data["date"]),
                    datetime.min.time(),
                    tzinfo=KST,
                )

            events.append({
                "id": event["id"],
                "title": event.get("summary", "제목 없음"),
                "start": event_start,
                "end": event_end,
            })

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return events
