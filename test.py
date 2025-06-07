import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# 0. Hugging Face 로그인 (토큰 직접 입력)
login("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")  # 🔁 실제 Hugging Face 토큰으로 교체

# 1. 임베딩 모델 로드
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. LLM 로드
model_id = "beomi/KoAlpaca-Polyglot-5.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=True,
    device_map="cpu",
    torch_dtype=torch.float16
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. JSON 파일 경로 수동 지정
filename = "card_benefits.json"  # 동일 폴더에 파일 위치
with open(filename, "r", encoding="utf-8") as f:
    json_list = json.load(f)

# 4. JSON -> 문장 및 임베딩
docs = []
embeddings = []

def convert_to_sentence(card):
    return f"{card['card_company']}의 {card['card_name']} 카드는 '{card['title']}' 혜택으로 {card['content']}"

for card in json_list:
    doc = convert_to_sentence(card)
    docs.append(doc)
    emb = embedding_model.encode(doc).tolist()
    embeddings.append(emb)

# 5. 유사도 기반 검색
def search_context(query, top_k=3):
    query_emb = embedding_model.encode(query).tolist()

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    scores = [cosine_sim(query_emb, emb) for emb in embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return "\n".join(docs[i] for i in top_indices)

# 6. 자연스러운 응답 생성
def generate_answer(query, context):
    prompt = f"""안녕하세요! 아래는 카드 혜택에 대한 정보입니다:

{context}

이 정보를 참고하여 고객이 다음과 같은 질문을 했습니다:
"{query}"

위 카드 혜택 정보만을 바탕으로, 정확하고 친절하게 답변해 주세요.
추측하지 말고 모르면 "해당 정보를 찾을 수 없습니다."라고 답변하세요.

답변:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "답변:" in decoded:
        return decoded.split("답변:")[1].strip()
    return decoded.strip()

# 7. 챗봇 실행
def run_chatbot():
    print("💳 카드 혜택 Q&A 챗봇입니다. 종료하려면 'exit' 입력")
    while True:
        query = input("\n🙋 사용자 질문: ").strip()
        if query.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break
        try:
            context = search_context(query)
        except:
            context = ""
        answer = generate_answer(query, context)
        print("\n🧠 답변:\n", answer)

# 8. 실행
if __name__ == "__main__":
    run_chatbot()
