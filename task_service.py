import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "tasks.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


def init_tasks_db():
    """할 일 저장용 데이터베이스 생성"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                start_date TEXT NOT NULL,
                deadline TEXT NOT NULL,
                estimated_minutes INTEGER,
                completed INTEGER DEFAULT 0,
                calendar_event_id TEXT
            )
        """)


def add_task(title, start_date, deadline, estimated_minutes=None):
    """새로운 할 일 등록"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    due = datetime.strptime(deadline, "%Y-%m-%d %H:%M")

    if due < start:
        raise ValueError("마감일은 시작 날짜보다 늦어야 합니다.")

    if not title.strip():
        raise ValueError("할 일 제목을 입력해야 합니다.")

    if estimated_minutes is not None and estimated_minutes <= 0:
        raise ValueError("예상 소요 시간은 0보다 커야 합니다.")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO tasks
            (title, start_date, deadline, estimated_minutes)
            VALUES (?, ?, ?, ?)
        """, (title.strip(), start_date, deadline, estimated_minutes))

        return cursor.lastrowid


def get_tasks(include_completed=False):
    """할 일 목록 조회"""
    query = "SELECT * FROM tasks"

    if not include_completed:
        query += " WHERE completed = 0"

    query += " ORDER BY deadline ASC"

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query)]


def complete_task(task_id):
    """할 일 완료 처리"""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE tasks SET completed = 1 WHERE id = ?",
            (task_id,)
        )
        return cursor.rowcount > 0


def set_calendar_event_id(task_id, event_id):
    """할 일에 연결된 Google Calendar 이벤트 ID 저장"""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE tasks
            SET calendar_event_id = ?
            WHERE id = ?
            """,
            (event_id, task_id)
        )
        return cursor.rowcount > 0


def get_task(task_id):
    """특정 할 일 하나 조회"""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,)
        ).fetchone()

        return dict(row) if row else None


def init_task_schedule_db():
    """작업 예정 시간 저장 테이블 생성"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_work_schedules (
                task_id INTEGER PRIMARY KEY,
                scheduled_start TEXT NOT NULL,
                scheduled_end TEXT NOT NULL,
                calendar_event_id TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)


def get_task_schedule(task_id):
    """작업 예정 시간 조회"""
    init_task_schedule_db()

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM task_work_schedules
            WHERE task_id = ?
            """,
            (task_id,)
        ).fetchone()

    return dict(row) if row else None


def save_task_schedule(task_id, start, end, event_id):
    """작업 예정 시간과 캘린더 ID 저장"""
    init_task_schedule_db()

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO task_work_schedules
                (task_id, scheduled_start,
                 scheduled_end, calendar_event_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                scheduled_start = excluded.scheduled_start,
                scheduled_end = excluded.scheduled_end,
                calendar_event_id = excluded.calendar_event_id
        """, (task_id, start, end, event_id))

def get_all_task_schedules():
    """DB에 저장된 모든 작업 시간 배정 조회"""
    init_task_schedule_db()

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                s.task_id,
                s.scheduled_start,
                s.scheduled_end,
                s.calendar_event_id,
                t.title,
                t.completed
            FROM task_work_schedules AS s
            JOIN tasks AS t ON s.task_id = t.id
            ORDER BY s.scheduled_start
        """).fetchall()

    return [dict(row) for row in rows]


def delete_task_schedule(task_id):
    """작업 시간 배정 기록만 삭제하고 할 일은 유지"""
    init_task_schedule_db()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM task_work_schedules
            WHERE task_id = ?
            """,
            (task_id,)
        )

        return cursor.rowcount > 0

# --- Calendar deletion pause: persistent and independent of scheduled blocks ---
def init_task_pause_db():
    """Create a separate table without changing or deleting existing tasks."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_schedule_pauses (
                task_id INTEGER PRIMARY KEY,
                paused_at TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT 'calendar_deleted',
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)


def pause_task_schedule(task_id, reason='calendar_deleted'):
    """Prevent automatic rescheduling, but keep the task itself."""
    init_task_pause_db()
    with get_connection() as conn:
        if conn.execute('SELECT 1 FROM tasks WHERE id = ?', (task_id,)).fetchone() is None:
            raise ValueError(f'할 일 ID {task_id}가 존재하지 않습니다.')
        conn.execute("""
            INSERT INTO task_schedule_pauses (task_id, paused_at, reason)
            VALUES (?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                paused_at = excluded.paused_at,
                reason = excluded.reason
        """, (task_id, datetime.now().astimezone().isoformat(), reason))


def resume_task_schedule(task_id):
    """Allow a previously paused task to be proposed again."""
    init_task_pause_db()
    with get_connection() as conn:
        return conn.execute(
            'DELETE FROM task_schedule_pauses WHERE task_id = ?',
            (task_id,),
        ).rowcount > 0


def is_task_schedule_paused(task_id):
    init_task_pause_db()
    with get_connection() as conn:
        return conn.execute(
            'SELECT 1 FROM task_schedule_pauses WHERE task_id = ?',
            (task_id,),
        ).fetchone() is not None


def get_paused_task_schedules():
    init_task_pause_db()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT p.task_id, t.title, t.completed, p.paused_at, p.reason
            FROM task_schedule_pauses p
            JOIN tasks t ON t.id = p.task_id
            ORDER BY p.paused_at DESC
        """).fetchall()
        return [dict(row) for row in rows]
