"""내 OpenAI 계정에서 쓸 수 있는 모델 ID 목록 출력.

실행 (레포 루트, venv 켠 상태):
    python -m scripts.list_models

여기 뜬 gpt-5.x 중 하나를 골라 .env 의 LLM_MODEL 에 넣으면 된다.
"""
from dotenv import load_dotenv

load_dotenv()  # .env 의 OPENAI_API_KEY 를 환경변수로 로드

from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 자동 사용
ids = sorted(m.id for m in client.models.list().data)

print("=== 사용 가능한 모델(gpt 계열) ===")
for m in ids:
    if "gpt" in m:
        print(" -", m)

print("\n=== 전체 개수:", len(ids), "===")
