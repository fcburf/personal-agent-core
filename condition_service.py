
import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "condition.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


def init_condition_db():
    """날짜별 피로도와 수면 데이터를 저장할 테이블 생성"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_date TEXT NOT NULL UNIQUE,
                fatigue_level INTEGER NOT NULL
                    CHECK(fatigue_level BETWEEN 1 AND 5),
                sleep_type TEXT NOT NULL
                    CHECK(sleep_type IN ('score', 'manual', 'all_nighter')),
                sleep_score INTEGER
                    CHECK(sleep_score BETWEEN 0 AND 100),
                bedtime TEXT,
                wake_time TEXT,
                sleep_minutes INTEGER,
                updated_at TEXT NOT NULL
            )
        """)


def calculate_sleep_minutes(bedtime, wake_time):
    """취침·기상 시각을 이용해 수면 시간 계산"""
    start = datetime.strptime(bedtime, "%H:%M")
    end = datetime.strptime(wake_time, "%H:%M")

    if end <= start:
        end += timedelta(days=1)

    minutes = int((end - start).total_seconds() / 60)

    if minutes > 16 * 60:
        raise ValueError("수면 시간이 16시간을 초과합니다. 입력을 확인해 주세요.")

    return minutes


def save_condition(
    fatigue_level,
    sleep_type,
    sleep_score=None,
    bedtime=None,
    wake_time=None,
    record_date=None,
):
    """오늘의 컨디션 저장. 같은 날짜에 재입력하면 수정."""
    init_condition_db()

    record_date = record_date or date.today().isoformat()
    date.fromisoformat(record_date)

    if not isinstance(fatigue_level, int) or not 1 <= fatigue_level <= 5:
        raise ValueError("피로도는 1~5 사이의 정수여야 합니다.")

    if sleep_type not in ("score", "manual", "all_nighter"):
        raise ValueError("올바른 수면 입력 방식을 선택해 주세요.")

    sleep_minutes = None

    if sleep_type == "score":
        if not isinstance(sleep_score, int) or not 0 <= sleep_score <= 100:
            raise ValueError("수면 점수는 0~100 사이의 정수여야 합니다.")
        bedtime = None
        wake_time = None

    elif sleep_type == "manual":
        if not bedtime or not wake_time:
            raise ValueError("취침 시간과 기상 시간을 입력해 주세요.")
        sleep_minutes = calculate_sleep_minutes(bedtime, wake_time)
        sleep_score = None

    else:
        sleep_score = 0
        sleep_minutes = 0
        bedtime = None
        wake_time = None

    updated_at = datetime.now().astimezone().isoformat()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO daily_conditions (
                record_date, fatigue_level, sleep_type,
                sleep_score, bedtime, wake_time,
                sleep_minutes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_date) DO UPDATE SET
                fatigue_level = excluded.fatigue_level,
                sleep_type = excluded.sleep_type,
                sleep_score = excluded.sleep_score,
                bedtime = excluded.bedtime,
                wake_time = excluded.wake_time,
                sleep_minutes = excluded.sleep_minutes,
                updated_at = excluded.updated_at
        """, (
            record_date, fatigue_level, sleep_type,
            sleep_score, bedtime, wake_time,
            sleep_minutes, updated_at
        ))

    return get_condition(record_date)


def get_condition(record_date=None):
    """특정 날짜의 컨디션 조회"""
    init_condition_db()
    record_date = record_date or date.today().isoformat()

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM daily_conditions WHERE record_date = ?",
            (record_date,)
        ).fetchone()

    return dict(row) if row else None


def get_recent_conditions(days=7):
    """최근 N일간의 컨디션 기록 조회"""
    init_condition_db()

    if days < 1:
        raise ValueError("조회 일수는 1 이상이어야 합니다.")

    start_date = (date.today() - timedelta(days=days - 1)).isoformat()

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM daily_conditions
            WHERE record_date >= ?
            ORDER BY record_date DESC
        """, (start_date,)).fetchall()

    return [dict(row) for row in rows]
