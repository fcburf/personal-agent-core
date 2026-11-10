# Personal Agent Core (`personal-agent-core`)

> **개인화 비서 에이전트 프로젝트**의 오케스트레이션 및 LangGraph 메인 코어 레포지토리입니다.
> 사용자의 컨디션, 일정, 소비 기록을 분석하여 근거 있는 하루 계획 및 구매 제안을 수행합니다.

---

## 📁 프로젝트 구조 (Directory Structure)

```text
personal-agent-core/
├── .env              # 환경 변수 (Git 추적 제외)
├── .env.example      # 환경 변수 템플릿
├── .gitignore        # Git 추적 제외 설정 (.env, DB 파일, 캐시 등)
├── README.md         # 프로젝트 가이드 문서
├── requirements.txt  # 의존성 패키지 목록
├── config.py         # LLM (Gemini Pro/Flash) 및 공통 환경 설정
├── state.py          # 노드 간 데이터 전달을 위한 공통 AgentState 스키마
├── graph_core.py     # 전체 에이전트 오케스트레이션 (Supervisor, Critic 루프, Interrupt)
└── db_init.py        # 장기 메모리(SQLite) 테이블 초기화 스크립트