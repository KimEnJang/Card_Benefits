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

    # 3. LLM 로딩 (KULLM 5.8B, 4bit + GPU 또는 CPU 환경 자동 할당)
    tokenizer = AutoTokenizer.from_pretrained("circulus/kullm-polyglot-5.8b-v2")
    model = AutoModelForCausalLM.from_pretrained(
        "circulus/kullm-polyglot-5.8b-v2",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )

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
        context = "\n".join(retrieved_docs) if retrieved_docs else ""

        # 5. 프롬프트 구성 (자연스럽고 친절한 말투)
        prompt = f"""당신은 카드 혜택 정보를 안내하는 챗봇입니다.
제공된 카드 혜택 정보만 사용해서 사용자 질문에 답변해 주세요.
정보가 없으면 "죄송합니다. 해당 혜택에 대한 정보는 찾을 수 없습니다."라고 말하세요.

카드 혜택 정보:
{context}

질문:
{question}

답변:"""

        # 6. 텍스트 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
        answer = response.split("답변:")[-1].strip()

        print(f"\n🤖 챗봇 응답: {answer}")

if __name__ == "__main__":
    run_chatbot()
