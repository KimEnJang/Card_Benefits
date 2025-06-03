import json
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Chroma 클라이언트 초기화 (디스크 저장)
client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection(name="card-benefits")

# 2. 임베딩 모델 로딩
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 3. JSON 파일 로드
with open("hyundae_cards_merged_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 4. 데이터 벡터화 및 저장
for i, item in enumerate(data):
    document = item["content"]
    metadata = {
        "card_company": item["card_company"],
        "card_name": item["card_name"],
        "title": item["title"]
    }
    collection.add(
        documents=[document],
        embeddings=[model.encode(document)],
        ids=[f"card_{i}"],
        metadatas=[metadata]
    )

print("✅ 카드 혜택 데이터 벡터화 및 저장 완료 (디스크 저장)")
