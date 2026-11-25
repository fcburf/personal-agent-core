"""장기 메모리 DB 초기화 (오케스트레이션 리드 관리)

모든 팀원 코드는 이 파일의 DB_PATH 하나(long_term_memory.db)를 함께 사용합니다.
테이블을 추가·변경할 때는 README의 '장기 메모리 DB' 절과 함께 수정하고 팀에 공지합니다.

| 테이블                | 쓰는 쪽 (작성)            | 읽는 쪽                         |
|----------------------|--------------------------|--------------------------------|
| trips                | 여행 전 입력 폼 (지인)       | 맥락 수집(trip)                    |
| purchase_history     | 구매·지출 데이터 (예정)     | 구매 에이전트(서경), 맥락 수집(spending) |
| purchase_feedback    | 회상 인터뷰 (서경)          | 구매 에이전트 기준 도출, Critic       |
| purchase_decisions   | 실행 에이전트 (구매 승인 후) | 구매 에이전트 재검토, Critic(PD2)     |
| user_patterns        | 회고·컨디션 분석            | 컨디션 분석, 구매 에이전트           |
| tasks                | 일정 연동 (지인)            | 맥락 수집(tasks)                  |
| task_work_schedules  | 일정 연동 (지인)            | 캘린더 동기화                      |
| daily_conditions     | 일정·생체 연동 (지인)        | 맥락 수집(biometrics)              |
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "long_term_memory.db"


def get_connection():
    """모든 모듈이 같은 DB 파일을 쓰도록 하는 공용 연결 함수"""
    return sqlite3.connect(DB_PATH)


SCHEMA = [
    # ---------- 여행 (지인) — 여행 전 입력 폼 ----------
    """
    CREATE TABLE IF NOT EXISTS trips (
        trip_id TEXT PRIMARY KEY,
        destination TEXT NOT NULL,
        start_date TEXT NOT NULL,           -- YYYY-MM-DD
        end_date TEXT NOT NULL,
        accommodation TEXT,
        budget INTEGER,
        currency TEXT DEFAULT 'KRW',
        wishlist TEXT,                      -- JSON 문자열 {places, foods, activities}
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # ---------- 구매 (예정 → 서경) ----------
    # 구매 이력: 데이터 연동이 저장 (Gmail / 캡처 / 수동 입력). 만족도는 purchase_feedback에 따로 저장
    """
    CREATE TABLE IF NOT EXISTS purchase_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name TEXT NOT NULL,
        category TEXT,
        price INTEGER CHECK(price > 0),
        purchase_date TEXT,                 -- YYYY-MM-DD
        year_missing INTEGER DEFAULT 0,     -- 캡처에 연도가 없어 기준일로 보정했으면 1
        data_source TEXT CHECK(data_source IN ('gmail', 'capture', 'manual')),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # 회상 인터뷰 결과: 구매 1주·1개월 뒤 각각 기록 가능 (한 구매에 여러 행)
    """
    CREATE TABLE IF NOT EXISTS purchase_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        purchase_id INTEGER NOT NULL REFERENCES purchase_history(id),
        recall_stage TEXT CHECK(recall_stage IN ('first_run', '1_week', '1_month')),
        satisfaction INTEGER CHECK(satisfaction BETWEEN 1 AND 5),  -- 건너뛰면 NULL
        reason_tags TEXT,                   -- JSON 배열 문자열 (예: '["음질", "가격 대비"]')
        comment TEXT,
        skipped INTEGER DEFAULT 0,          -- '나중에'로 넘긴 경우 1 (RD1: 같은 날 재질문 금지)
        recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # 구매 상담 결정: 승인된 추천/보류 기록 (보류는 재검토 시점 포함, PD1)
    """
    CREATE TABLE IF NOT EXISTS purchase_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id TEXT,                       -- 여행 중 상담이면 trips.trip_id
        item TEXT NOT NULL,
        category TEXT,
        decision TEXT CHECK(decision IN ('recommend', 'hold')),
        recommended TEXT,                   -- JSON {name, price, currency, quantity}
        user_response TEXT CHECK(user_response IN ('approved', 'rejected', 'modify')),
        reject_reason_tags TEXT,            -- JSON 배열 (가격/수량/이미 있음/필요 없음/브랜드)
        purchase_context TEXT,              -- 구매하려던 이유·상황 한 줄
        evidence TEXT,                      -- JSON 문자열 (검색 출처 등)
        revisit_at TEXT,                    -- 보류 재검토 날짜 YYYY-MM-DD
        decided_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # 확인된 개인 패턴 (C3·PC3·RA2: 한 번의 답으로 확정 금지 → confirmed_count로 관리)
    """
    CREATE TABLE IF NOT EXISTS user_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT,
        description TEXT,
        confidence_score REAL,
        confirmed_count INTEGER DEFAULT 1,
        last_updated TEXT
    )
    """,
    # ---------- 일정·생체 (지인) — 지인 님 task_service / condition_service 스키마와 동일 ----------
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        start_date TEXT NOT NULL,
        deadline TEXT NOT NULL,             -- YYYY-MM-DD HH:MM
        estimated_minutes INTEGER,
        completed INTEGER DEFAULT 0,
        calendar_event_id TEXT,
        is_collaborative INTEGER DEFAULT 0  -- [추가] 플래너 A5·E2 판단용 (협업 업무면 1)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_work_schedules (
        task_id INTEGER PRIMARY KEY,
        scheduled_start TEXT NOT NULL,
        scheduled_end TEXT NOT NULL,
        calendar_event_id TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_conditions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_date TEXT NOT NULL UNIQUE,
        fatigue_level INTEGER NOT NULL CHECK(fatigue_level BETWEEN 1 AND 5),
        sleep_type TEXT NOT NULL CHECK(sleep_type IN ('score', 'manual', 'all_nighter')),
        sleep_score INTEGER CHECK(sleep_score BETWEEN 0 AND 100),
        bedtime TEXT,
        wake_time TEXT,
        sleep_minutes INTEGER,
        updated_at TEXT NOT NULL
    )
    """,
]


def init_db(db_path=None):
    """테이블이 없으면 만들고, 있으면 그대로 둠 (여러 번 실행해도 안전)"""
    conn = sqlite3.connect(db_path or DB_PATH)
    try:
        for ddl in SCHEMA:
            conn.execute(ddl)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"장기 메모리 DB 테이블 생성 완료: {DB_PATH}")
