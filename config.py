import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # .env 파일 자동 로드

# 모델 설정 (.env 값이 있으면 우선 적용)
# - 소형: 분류·추출 (Supervisor, 컨디션 분석)
# - 대형: 판단·검증 (플래너, Critic)
# ※ 2026-11 기준 팀 API 키로 쓸 수 있는 Pro 모델은 gemini-3.1-pro-preview 뿐
#   (gemini-3.8-pro는 존재하지 않음, gemini-2.5-pro는 신규 사용자 차단)
MODEL_SMALL = os.getenv("GEMINI_MODEL_SMALL", "gemini-3.8-flash")
MODEL_LARGE = os.getenv("GEMINI_MODEL_LARGE", "gemini-3.1-pro-preview")

# preview 모델은 v1beta API에서만 열려 있으므로 API 버전을 명시 (v1이면 404)
API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")

llm_small = ChatGoogleGenerativeAI(model=MODEL_SMALL, temperature=0.2, api_version=API_VERSION)
llm_large = ChatGoogleGenerativeAI(model=MODEL_LARGE, temperature=0.7, api_version=API_VERSION)
