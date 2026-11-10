import sqlite3

def init_db():
    """팀원들이 로컬 환경에서 장기 메모리 DB를 초기화할 수 있는 스크립트"""
    conn = sqlite3.connect("long_term_memory.db")
    cursor = conn.cursor()
    
    # 구매 이력 및 만족도 테이블 (회상 인터뷰 및 구매 어드바이저용)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS purchase_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            category TEXT,
            price INTEGER,
            purchase_date TEXT,
            satisfaction_score INTEGER CHECK(satisfaction_score BETWEEN 1 AND 5),
            reason_tag TEXT,
            data_source TEXT
        )
    """)
    
    # 확인된 개인 패턴 테이블 (컨디션 분석용)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            description TEXT,
            confidence_score REAL,
            last_updated TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("장기 메모리 SQLite DB(long_term_memory.db) 테이블 생성 완료!")

if __name__ == "__main__":
    init_db()