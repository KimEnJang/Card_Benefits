import json
import torch
import chromadb
from google.colab import files
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 임베딩 모델
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. LLM 불러오기
model_id = "beomi/KoAlpaca-Polyglot-5.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. ChromaDB 설정
client = chromadb.Client()
collection = client.get_or_create_collection(name="card-benefits")

# 4. JSON 업로드 및 벡터 저장
print("📁 카드 혜택 JSON 파일을 업로드하세요 (예: card_benefits.json)")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

with open(filename, "r", encoding="utf-8") as f:
    json_list = json.load(f)

def convert_to_sentence(card):
    return f"{card['card_company']}의 {card['card_name']} 카드는 '{card['title']}' 혜택으로 {card['content']}"

for i, card in enumerate(json_list):
    doc = convert_to_sentence(card)
    embedding = embedding_model.encode(doc).tolist()
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[f"card_{i}"]
    )

# 5. 검색 함수 (context 선택적으로 붙임)
def search_context(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n".join(results["documents"][0])

# 6. 프롬프트 구성 및 응답 생성
def generate_answer(query, context=None):
    if context:
        prompt = f"""### Instruction:
아래는 신용카드 혜택 정보입니다. 이 정보를 바탕으로 사용자의 질문에 친절히 응답하세요.

카드 혜택 정보:
{context}

질문: {query}

### Response:"""
    else:
        prompt = f"""### Instruction:
당신은 신용카드 혜택에 대해 잘 아는 상담원입니다. 아래 질문에 대해 자연스럽고 친절하게 답변하세요.

질문: {query}

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "### Response:" in decoded:
        return decoded.split("### Response:")[1].strip()
    return decoded.strip()

# 7. 챗봇 실행
def run_chatbot():
    print("💳 카드 혜택 Q&A 챗봇입니다. 종료하려면 'exit' 입력")
    while True:
        query = input("\n🙋 사용자 질문: ").strip()
        if query.lower() in ["exit", "quit", "종료"]:
            print("챗봇 종료합니다.")
            break

        try:
            context = search_context(query)
        except:
            context = None

        answer = generate_answer(query, context)
        print("\n🧠 답변:\n", answer)

# 8. 실행
run_chatbot()
