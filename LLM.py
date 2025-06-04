import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def run_chatbot():
    # 1. ChromaDB 불러오기
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

    print("🧾 카드 혜택 Q&A 챗봇입니다. (종료하려면 'exit' 입력)")

    while True:
        question = input("\n🙋 사용자 질문: ").strip()
        if question.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break

        # 4. 유사 문서 검색
        query_embedding = embedding_model.encode(question, normalize_embeddings=True)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        retrieved_docs = results["documents"][0] if results["documents"] else []
        context = "\n".join(retrieved_docs).strip()

        # 🔐 5. context가 없으면 답변 금지
        if not context or len(context) < 20:
            print("\n🤖 챗봇 응답: 죄송합니다. 해당 혜택에 대한 정보는 찾을 수 없습니다.")
            continue

        # ✅ 6. 프롬프트 구성 (지시 강화)
        prompt = f"""
당신은 카드 혜택 정보를 안내하는 전문 챗봇입니다.
아래 제공된 카드 혜택 정보만을 근거로 답변해야 하며, 다른 지식이나 추측을 추가하지 마세요.
만약 아래 정보에 답이 없으면 "죄송합니다. 해당 혜택에 대한 정보는 찾을 수 없습니다."라고 답하세요.
의미 없는 반복 문장, 같은 문장 구조 반복, 모호한 정의는 피하세요.

[카드 혜택 정보]
{context}

[질문]
{question}

[답변]
"""

        # 7. 텍스트 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("[답변]")[-1].strip()

        print(f"\n🤖 챗봇 응답: {answer}")

if __name__ == "__main__":
    run_chatbot()
