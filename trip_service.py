import json
from db_init import get_connection
from datetime import date
from uuid import uuid4


def create_trip(
    destination: str,
    start_date: str,
    end_date: str,
    accommodation: str = "",
    budget: float = 0,
    currency: str = "KRW",
    places: list[str] | None = None,
    foods: list[str] | None = None,
    activities: list[str] | None = None,
) -> dict:
    """여행 기본 정보를 공통 context_data 형식으로 생성한다."""

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if end < start:
        raise ValueError("여행 종료일은 시작일보다 빠를 수 없습니다.")

    if not destination.strip():
        raise ValueError("여행 목적지를 입력해야 합니다.")

    if budget < 0:
        raise ValueError("여행 예산은 음수일 수 없습니다.")

    return {
        "trip_id": f"trip_{uuid4().hex[:12]}",
        "destination": destination.strip(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "accommodation": accommodation.strip(),
        "budget": budget,
        "currency": currency.upper(),
        "wishlist": {
            "places": places or [],
            "foods": foods or [],
            "activities": activities or [],
        },
    }


def save_trip(trip: dict) -> str:
    """여행 정보를 공용 SQLite DB에 저장한다."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO trips (
                trip_id, destination, start_date, end_date,
                accommodation, budget, currency, wishlist
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trip["trip_id"],
                trip["destination"],
                trip["start_date"],
                trip["end_date"],
                trip["accommodation"],
                trip["budget"],
                trip["currency"],
                json.dumps(trip["wishlist"], ensure_ascii=False),
            ),
        )

    return trip["trip_id"]


def get_trip(trip_id: str) -> dict | None:
    """여행 ID로 저장된 여행 정보를 조회한다."""
    with get_connection() as conn:
        conn.row_factory = __import__("sqlite3").Row

        row = conn.execute(
            """
            SELECT trip_id, destination, start_date, end_date,
                   accommodation, budget, currency, wishlist
            FROM trips
            WHERE trip_id = ?
            """,
            (trip_id,),
        ).fetchone()

    if row is None:
        return None

    trip = dict(row)
    trip["wishlist"] = json.loads(trip["wishlist"] or "{}")
    return trip


def get_active_trip(today: str | None = None) -> dict | None:
    """현재 진행 중이거나 가장 가까운 예정 여행을 조회한다."""
    if today is None:
        today = date.today().isoformat()

    with get_connection() as conn:
        conn.row_factory = __import__("sqlite3").Row

        row = conn.execute(
            """
            SELECT trip_id
            FROM trips
            WHERE end_date >= ?
            ORDER BY
                CASE
                    WHEN start_date <= ? THEN 0
                    ELSE 1
                END,
                start_date ASC
            LIMIT 1
            """,
            (today, today),
        ).fetchone()

    if row is None:
        return None

    return get_trip(row["trip_id"])


if __name__ == "__main__":
    demo_trip = create_trip(
        destination="일본 오사카·교토",
        start_date="2026-12-01",
        end_date="2026-12-03",
        accommodation="오사카 난바역 인근",
        budget=800000,
        currency="KRW",
        places=[
            "후시미 이나리 신사",
            "기요미즈데라",
            "유니버설 스튜디오 재팬",
            "도톤보리",
            "신사이바시",
        ],
        foods=["타코야키", "라멘", "말차 디저트"],
        activities=["교토 관광", "어트랙션 체험", "쇼핑"],
    )

    from pprint import pprint
    pprint(demo_trip, sort_dicts=False)
