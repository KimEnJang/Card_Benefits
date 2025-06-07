import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def run_chatbot():
    # 1. ChromaDB 로딩
    client = chromadb.PersistentClient(path="./chroma_storage")
    collection = client.get_or_create_collection(name="card-benefits")

    # 2. 임베딩 모델
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 3. 경량 LLM
    model_id = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("🧾 카드 혜택 Q&A 챗봇입니다. 종료하려면 'exit' 입력")

    # 4. 문서 및 임베딩 불러오기
    results_all = collection.get(include=["documents", "embeddings"])
    docs = results_all["documents"]
    embeddings = results_all["embeddings"]

    # 5. 검색 함수 정의
    def search_context(query, top_k=3):
        query_emb = embedding_model.encode(query).tolist()

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x**2 for x in a) ** 0.5
            norm_b = sum(x**2 for x in b) ** 0.5
            return dot / (norm_a * norm_b + 1e-8)

        scores = [cosine_sim(query_emb, emb) for emb in embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # title 키워드 필터링
        title_keywords = ["기본 혜택", "추가 혜택", "우대 서비스"]
        filtered = []
        for i in top_indices:
            doc = docs[i]
            if any(k in query and k in doc for k in title_keywords):
                filtered.append(doc)
            if len(filtered) >= top_k:
                break

        # fallback: 유사도 상위 top_k
        if not filtered:
            filtered = [docs[i] for i in top_indices[:top_k]]

        return "\n".join(filtered).strip()

    # 6. 챗봇 실행 루프
    while True:
        question = input("\n🙋 사용자 질문: ").strip()
        if question.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break

        context = search_context(question)

        # context 없으면 중단
        if not context or len(context) < 20:
            print("\n🤖 챗봇 응답: 죄송합니다. 해당 혜택에 대한 정보는 찾을 수 없습니다.")
            continue

        # 7. 프롬프트 구성
        prompt = f"""당신은 카드 혜택을 안내하는 상담 챗봇입니다.

[카드 혜택 정보]
{context}

[사용자 질문]
"{question}"

[답변 규칙]
- 반드시 위 카드 혜택 정보만을 기반으로 답변하세요.
- 질문과 관련된 혜택 제목(예: '기본 혜택')에 해당하는 정보만 사용하세요.
- 관련 없는 정보, 반복 문장, 외부 지식은 절대 포함하지 마세요.
- 명확하고 자연스럽게 한두 문장으로 답변을 마무리하세요.
- 정보가 없으면 "죄송합니다. 해당 혜택에 대한 정보는 확인되지 않았습니다."라고 답변하세요.

[답변]
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=2.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("[답변]")[-1].strip()

        print(f"\n🤖 챗봇 응답: {answer}")

if __name__ == "__main__":
    run_chatbot()
