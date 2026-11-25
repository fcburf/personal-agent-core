"""장기 메모리 DB 스키마 테스트 (임시 DB 파일 사용, 실제 long_term_memory.db는 건드리지 않음)"""
import json
import sqlite3
import tempfile
from pathlib import Path
from helpers import ROOT  # noqa: F401  (sys.path 설정)
import db_init
import task_service
import condition_service


def _fresh_db():
    path = Path(tempfile.mkdtemp()) / "test_ltm.db"
    db_init.init_db(path)
    return path


def test_all_tables_created_and_rerun_safe():
    path = _fresh_db()
    db_init.init_db(path)                                           # 두 번 실행해도 안전
    tables = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"trips", "purchase_history", "purchase_feedback", "purchase_decisions", "user_patterns",
            "tasks", "task_work_schedules", "daily_conditions"} <= tables


def test_jiin_functions_work_on_unified_db():
    path = _fresh_db()
    old_t, old_c = task_service.DB_FILE, condition_service.DB_FILE
    task_service.DB_FILE = condition_service.DB_FILE = path         # DB 경로만 바꾸면 기존 함수 그대로 동작
    try:
        tid = task_service.add_task("여행 짐 싸기", "2026-11-28", "2026-11-29 22:00", 60)
        t = task_service.get_task(tid)
        assert t["title"] == "여행 짐 싸기" and t["is_collaborative"] == 0
        condition_service.save_condition(fatigue_level=4, sleep_type="manual",
                                         bedtime="02:00", wake_time="06:00", record_date="2026-11-30")
        assert condition_service.get_condition("2026-11-30")["sleep_minutes"] == 240
    finally:
        task_service.DB_FILE, condition_service.DB_FILE = old_t, old_c


def test_trip_and_purchase_rejection_rows():
    con = sqlite3.connect(_fresh_db())
    con.execute("INSERT INTO trips(trip_id, destination, start_date, end_date, budget, currency, wishlist) "
                "VALUES ('trip_osaka','일본 오사카','2026-11-29','2026-12-01',60000,'JPY',?)",
                (json.dumps({"places": ["도톤보리"]}, ensure_ascii=False),))
    con.execute("INSERT INTO purchase_decisions(trip_id, item, category, decision, recommended, user_response, "
                "reject_reason_tags, purchase_context) VALUES ('trip_osaka','다리 피로 회복 패치','건강·위생',"
                "'recommend',?, 'rejected', ?, 'USJ에서 2만 보')",
                (json.dumps({"name": "휴족시간 18매입", "price": 932, "currency": "JPY", "quantity": 1}),
                 json.dumps(["가격"], ensure_ascii=False)))
    row = con.execute("SELECT user_response, reject_reason_tags FROM purchase_decisions").fetchone()
    assert row[0] == "rejected" and json.loads(row[1]) == ["가격"]


def test_two_recalls_per_purchase_and_constraints():
    con = sqlite3.connect(_fresh_db())
    pid = con.execute("INSERT INTO purchase_history(item_name, category, price, purchase_date, data_source) "
                      "VALUES ('이어폰','전자기기',39000,'2026-11-01','capture')").lastrowid
    for stage, score in (("1_week", 4), ("1_month", 2)):
        con.execute("INSERT INTO purchase_feedback(purchase_id, recall_stage, satisfaction) VALUES (?,?,?)",
                    (pid, stage, score))
    assert con.execute("SELECT COUNT(*) FROM purchase_feedback").fetchone()[0] == 2
    try:
        con.execute("INSERT INTO purchase_history(item_name, price, data_source) VALUES ('x', 100, 'kakao')")
        raise AssertionError("잘못된 data_source가 들어감")
    except sqlite3.IntegrityError:
        pass


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:
            failed += 1; print("FAIL", name, "->", repr(e)[:200])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
