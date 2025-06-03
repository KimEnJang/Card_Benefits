
import json
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 디스크 기반 ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_collection(name="card-benefits")

# 2. 임베딩 모델 로딩
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 3. KULLM 모델 및 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/kullm3-7b")
model = AutoModelForCausalLM.from_pretrained(
    "nlpai-lab/kullm3-7b",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

# 4. 챗봇 루프 시작
print("카드 혜택 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)")
while True:
    question = input("\n사용자 질문: ")
    if question.strip().lower() in ["exit", "quit", "종료"]:
        print("챗봇을 종료합니다.")
        break

    # 질문 임베딩 후 유사 문서 검색
    query_embedding = embedding_model.encode(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieved_docs = results["documents"][0]
    context = "\n".join(retrieved_docs)

    # 프롬프트 구성
    prompt = f"""### 질문:
{question}

### 카드 혜택 정보:
{context}

### 답변:"""

    # LLM 추론
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 출력
    print("\n🤖 챗봇 응답:")
    print(response.split("### 답변:")[-1].strip())
