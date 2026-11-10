import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # .env 파일 자동 로드

# 환경변수에서 로드하거나 최신 3.8 라인업을 기본값으로 사용
MODEL_SMALL = os.getenv("GEMINI_MODEL_SMALL", "gemini-3.8-flash")
MODEL_LARGE = os.getenv("GEMINI_MODEL_LARGE", "gemini-3.8-pro")

llm_small = ChatGoogleGenerativeAI(model=MODEL_SMALL, temperature=0.2)
llm_large = ChatGoogleGenerativeAI(model=MODEL_LARGE, temperature=0.7)