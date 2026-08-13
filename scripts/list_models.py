"""엔드포인트에서 쓸 수 있는 모델 ID 목록 출력.

실행 (레포 루트, venv 켠 상태):
    python -m scripts.list_models

.env 의 LLM_BASE_URL 이 있으면 그 엔드포인트(로컬/OpenAI 호환 서버)의 모델을 나열한다.
여기 뜬 것 중 하나를 .env 의 LLM_MODEL 에 넣으면 된다.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # .env 의 OPENAI_API_KEY / LLM_BASE_URL 로드

from openai import OpenAI

base_url = os.getenv("LLM_BASE_URL") or None
# LLM_BASE_URL 이 있으면 그 엔드포인트(예: 로컬/호환 서버)를, 없으면 OpenAI 본가를 조회.
client = OpenAI(base_url=base_url)

print(f"=== 엔드포인트: {base_url or 'https://api.openai.com/v1 (기본)'} ===")
try:
    ids = sorted(m.id for m in client.models.list().data)
except Exception as exc:
    print(f"모델 목록 조회 실패: {exc}")
    raise SystemExit(1)

for m in ids:
    print(" -", m)
print(f"\n=== 전체 {len(ids)}개 ===")
