# test
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# .env 파일 로드
load_dotenv()

# Hugging Face API 토큰 체크
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")

# ✅ 내부 LLM 모델 (예: Llama2, KoGPT)
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# `pad_token`을 명시적으로 설정 (모델 로드 직후 한 번만 설정)
tokenizer.pad_token = tokenizer.eos_token

# None 또는 빈 문자열을 "정보 없음"으로 변환하는 함수
def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

# Step-Back 기능 추가된 텍스트 생성 함수
def generate_step_back_text(obj, query=None):
    """
    Step-Back 스타일로 텍스트를 생성하는 함수:
    - 입력 데이터를 기반으로 동적으로 FAQ 텍스트를 생성.
    - 사용자 질문(query)을 받아 질문을 더 추상적이고 일반적인 형태로 변환.
    """
    
    # 1. 사용자 질문을 추상화하여 Step-Back 처리
    step_back_question = (
        "이 서비스의 전반적인 정보는 무엇인가요?"
        if query is None
        else f"'{query}'와 관련된 정보를 종합하여 요약하세요."
    )
    
    # 2. FAQ 스타일 정보 생성
    faq_text = (
        f"Q: 이 서비스의 이름은 무엇인가요?\nA: {clean_metadata(obj.get('서비스명'))}\n\n"
        f"Q: 이 서비스의 ID는 무엇인가요?\nA: {clean_metadata(obj.get('서비스ID'))}\n\n"
        f"Q: 이 서비스는 어떤 기관에서 제공하나요?\nA: {clean_metadata(obj.get('부서명'))}\n\n"
        f"Q: 이 서비스는 어떤 분야에 속하나요?\nA: {clean_metadata(obj.get('서비스분야'))}\n\n"
        f"Q: 이 서비스의 목적은 무엇인가요?\nA: {clean_metadata(obj.get('서비스목적요약'))}\n\n"
        f"Q: 이 서비스를 받을 수 있는 대상은 누구인가요?\nA: {clean_metadata(obj.get('지원대상'))}\n\n"
        f"Q: 지원 내용은 무엇인가요?\nA: {clean_metadata(obj.get('지원내용'))}\n\n"
        f"Q: 이 서비스의 선정 기준은 무엇인가요?\nA: {clean_metadata(obj.get('선정기준'))}\n\n"
        f"Q: 이 서비스의 지원 유형은 무엇인가요?\nA: {clean_metadata(obj.get('지원유형'))}\n\n"
        f"Q: 언제까지 신청할 수 있나요?\nA: {clean_metadata(obj.get('신청기한'))}\n\n"
        f"Q: 신청 방법은 무엇인가요?\nA: {clean_metadata(obj.get('신청방법'))}\n\n"
        f"Q: 어디에서 신청할 수 있나요?\nA: {clean_metadata(obj.get('접수기관'))}\n\n"
    )
    
    # 3. 자연어 스타일 문장 생성 (종합)
    summary_text = (
        f"{clean_metadata(obj.get('서비스명'))} 서비스 (서비스ID: {clean_metadata(obj.get('서비스ID'))})는 "
        f"{clean_metadata(obj.get('부서명'))}에서 운영하는 {clean_metadata(obj.get('서비스분야'))} 분야의 서비스입니다. "
        f"이 서비스는 {clean_metadata(obj.get('서비스목적요약'))}을 목표로 합니다. "
        f"이 서비스를 받을 수 있는 대상은 {clean_metadata(obj.get('지원대상'))}입니다. "
        f"주요 지원 내용은 다음과 같습니다: {clean_metadata(obj.get('지원내용'))}. "
        f"이 서비스의 선정 기준은 다음과 같습니다: {clean_metadata(obj.get('선정기준'))}. "
        f"지원 유형은 {clean_metadata(obj.get('지원유형'))}입니다. "
        f"신청 기한은 {clean_metadata(obj.get('신청기한'))}까지이며, 신청 방법은 {clean_metadata(obj.get('신청방법'))}입니다. "
        f"관련 문의 및 신청은 {clean_metadata(obj.get('접수기관'))}에서 할 수 있습니다."
    )
    
    # 4. Step-Back 질문과 결합
    step_back_text = (
        f"### Step-Back Question\n"
        f"{step_back_question}\n\n"
        f"### FAQ 스타일 정보\n"
        f"{faq_text}\n\n"
        f"### 종합 요약\n"
        f"{summary_text}"
    )
    
    return step_back_text

# JSON 데이터 로드
json_file_path = r"/home/elicer/embedding2/FinalPJ/20250304.json"
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"JSON 파일 '{json_file_path}'을 찾을 수 없습니다.")
except json.JSONDecodeError:
    raise ValueError(f"JSON 파일 '{json_file_path}'의 형식이 잘못되었습니다.")

if not isinstance(json_data, list):
    raise ValueError("JSON 데이터가 리스트 형태가 아닙니다.")

# Document 객체로 변환
documents = []
for obj in json_data:
    text = generate_step_back_text(obj)
    metadata = {
        "서비스ID": clean_metadata(obj.get("서비스ID")),
        "서비스명": clean_metadata(obj.get("서비스명")),
        "서비스목적요약": clean_metadata(obj.get("서비스목적요약")),
        "신청기한": clean_metadata(obj.get("신청기한")),
        "지원내용": clean_metadata(obj.get("지원내용")),
        "서비스분야": clean_metadata(obj.get("서비스분야")),
        "선정기준": clean_metadata(obj.get("선정기준")),
        "신청방법": clean_metadata(obj.get("신청방법")),
        "부서명": clean_metadata(obj.get("부서명")),
        "접수기관": clean_metadata(obj.get("접수기관"))
    }
    documents.append(Document(page_content=text, metadata=metadata))

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",  # 기존과 동일한 모델 사용
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
# # FAISS 벡터스토어 생성 및 저장
persist_directory = r"/home/elicer/embedding2/FinalPJ"
# os.makedirs(persist_directory, exist_ok=True)

# try:
#     vectorstore = FAISS.from_documents(documents, embedding_model)
#     vectorstore.save_local(persist_directory)
#     print(f"총 {len(documents)}개의 문서가 FAISS 벡터DB에 저장됨.")
#     print(f"저장 경로: {persist_directory}")
# except Exception as e:
#     print(f"FAISS 벡터 DB 저장 중 오류 발생: {e}")

# FAISS 로드 및 리트리버 생성
try:
    vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    raise Exception(f"FAISS 벡터 DB 로드 중 오류 발생: {e}")

# ✅ 일반 RAG 검색
def retrieve_documents(query, top_k=20):
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs
#retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # ✅ retriever 설정


def extract_rank_score(ranked_text, document):
    """
    GPT가 생성한 랭킹 결과에서 해당 문서의 점수를 추출하는 함수
    """
    # 문서 내용의 일부를 기준으로 검색 (일치율 높이기 위해 일부 문구 사용)
    doc_snippet = document.page_content[:30]  # 첫 30자 정도 사용 (중복 방지)

    # 정규 표현식으로 점수 추출 (예: "문서 A - 95점")
    pattern = re.compile(rf"{re.escape(doc_snippet)}.*?(\d+)점", re.DOTALL)
    match = pattern.search(ranked_text)

    if match:
        return int(match.group(1))  # 점수 반환
    return 0  # 점수 없으면 0점

def rank_documents_with_llm(query, documents):
    prompt = "다음 사용자 질문에 대해 가장 관련 있는 문서를 순서대로 나열하고 각각의 관련성 점수를 1점에서 100점 사이로 매겨주세요.\n\n"
    for i, doc in enumerate(documents):
        prompt += f"{i+1}. {doc.page_content}\n"

    prompt += "\n각 문서의 순위와 점수를 다음과 같은 형식으로 출력해주세요: (순위). (문서 내용 일부) - (점수)점"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=200)

    ranked_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 결과에서 점수 기반으로 문서를 정렬 (가장 높은 점수를 받은 문서가 상위)
    ranked_docs = sorted(documents, key=lambda x: extract_rank_score(ranked_text, x), reverse=True)

    return ranked_docs[:15]  # 상위 15개 문서 반환


def step_back_rag(query, documents):
    context = "\n".join([doc.page_content for doc in documents])
    input_text = f"사용자 질문: {query}\n\n제공된 정보:\n{context}\n\n제공된 정보를 바탕으로 사용자 질문에 대한 정확하고 상세한 답변을 생성하세요."

    # 입력 텍스트를 토크나이즈하면서, attention_mask와 pad_token_id 설정
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True  # 자동으로 패딩 처리
    ).to("cuda")

    # attention_mask 명시적으로 설정 (패딩 토큰을 0으로 설정)
    inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()

    # pad_token_id를 명시적으로 설정
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=300,
        pad_token_id=tokenizer.pad_token_id  # 패딩 토큰 명시적 설정
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 테스트 쿼리 실행
query = "서울에 1인가구 저소득층에 맞는 서비스를 찾아줘."

print("🔍 1단계: 일반 RAG 검색")
retrieved_docs = retrieve_documents(query)

print("📊 2단계: RankGPT 대체 (LLM 랭킹)")
ranked_docs = rank_documents_with_llm(query, retrieved_docs)

print("🧠 3단계: Step-Back RAG 적용")
response = step_back_rag(query, ranked_docs)

print(response)