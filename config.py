import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # .env 파일 자동 로드

llm_small = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
llm_large = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)