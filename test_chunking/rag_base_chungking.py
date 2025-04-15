# test
import os
import json
import traceback
import re
import string
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# .env 파일 로드
load_dotenv()

# API 토큰 체크
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# None 또는 빈 문자열을 "정보 없음"으로 변환하는 함수
def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

# FAQ 스타일과 자연어 문장 스타일을 결합한 텍스트 생성 함수 (새 필드 반영)
def generate_selected_text(obj):
    return (
        f"Q: 이 서비스의 이름은 무엇인가요?\nA: {clean_metadata(obj.get('benefit_summary'))}\n\n"
        f"Q: 이 서비스의 ID는 무엇인가요?\nA: {clean_metadata(obj.get('service_id'))}\n\n"
        f"Q: 이 서비스는 어떤 기관에서 제공하나요?\nA: {clean_metadata(obj.get('source'))}\n\n"
        f"Q: 이 서비스는 어떤 분야에 속하나요?\nA: {clean_metadata(obj.get('benefit_category'))}\n\n"
        f"Q: 이 서비스의 목적은 무엇인가요?\nA: {clean_metadata(obj.get('benefit_summary'))}\n\n"
        f"Q: 이 서비스를 받을 수 있는 대상은 누구인가요?\nA: {clean_metadata(obj.get('personal_summary'))} (최소 나이: {clean_metadata(obj.get('min_age'))}, 최대 나이: {clean_metadata(obj.get('max_age'))}, 성별: {clean_metadata(obj.get('gender'))})\n\n"
        f"Q: 지원 내용은 무엇인가요?\nA: {clean_metadata(obj.get('support_summary'))}\n\n"
        f"Q: 이 서비스의 세부 내용은 무엇인가요?\nA: {clean_metadata(obj.get('benefit_details'))}\n\n"
        f"Q: 이 서비스의 지원 유형은 무엇인가요?\nA: {clean_metadata(obj.get('support_type'))}\n\n"
        f"Q: 언제부터 언제까지 신청할 수 있나요?\nA: 시작일: {clean_metadata(obj.get('start_date'))}, 종료일: {clean_metadata(obj.get('end_date'))}\n\n"
        f"Q: 신청 방법은 무엇인가요?\nA: {clean_metadata(obj.get('application_method'))}\n\n"
        f"Q: 어디에서 신청할 수 있나요?\nA: {clean_metadata(obj.get('application_summary'))}\n\n"
        f"{clean_metadata(obj.get('benefit_summary'))} 서비스 (서비스ID: {clean_metadata(obj.get('service_id'))})는 "
        f"{clean_metadata(obj.get('source'))}에서 운영하는 {clean_metadata(obj.get('benefit_category'))} 분야의 서비스입니다. "
        f"이 서비스는 {clean_metadata(obj.get('benefit_summary'))}을 목표로 하며, 대상은 {clean_metadata(obj.get('personal_summary'))}입니다. "
        f"주요 지원 내용은 {clean_metadata(obj.get('support_summary'))}이며, 세부 내용은 {clean_metadata(obj.get('benefit_details'))}입니다. "
        f"지원 유형은 {clean_metadata(obj.get('support_type'))}이며, 신청 기간은 {clean_metadata(obj.get('start_date'))}부터 {clean_metadata(obj.get('end_date'))}까지입니다. "
        f"신청 방법은 {clean_metadata(obj.get('application_method'))}이며, 문의 및 신청은 {clean_metadata(obj.get('application_summary'))}에서 진행됩니다."
    )

# 텍스트 정규화 함수
def normalize_text(text):
    normalized = " ".join(text.strip().split()).lower()
    normalized = re.sub(f"[{re.escape(string.punctuation)}]", "", normalized)
    return normalized

# JSON 데이터 로드 및 서비스ID 기준 중복 제거
json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\meanchunking\to_rds_v1.json"  # 새 파일 경로로 변경 필요
with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

if not isinstance(json_data, list):
    raise ValueError("JSON 데이터가 리스트 형태가 아닙니다.")

documents_dict = {}
for obj in json_data:
    service_id = clean_metadata(obj.get("service_id"))
    if service_id not in documents_dict:
        text = generate_selected_text(obj)
        metadata = {
            "서비스ID": service_id,
            "서비스명": clean_metadata(obj.get("benefit_summary")),  # benefit_summary를 서비스명으로 사용
            "서비스목적요약": clean_metadata(obj.get("benefit_summary")),
            "신청기한": f"{clean_metadata(obj.get('start_date'))} ~ {clean_metadata(obj.get('end_date'))}",
            "지원내용": clean_metadata(obj.get("support_summary")),
            "서비스분야": clean_metadata(obj.get("benefit_category")),
            "세부내용": clean_metadata(obj.get("benefit_details")),  # benefit_details 추가
            "지원유형": clean_metadata(obj.get("support_type")),
            "부서명": clean_metadata(obj.get("source")),
            "접수기관": clean_metadata(obj.get("application_summary")),
            "지원대상": clean_metadata(obj.get("personal_summary")),
            "최소나이": clean_metadata(obj.get("min_age")),  # 새 필드 추가
            "최대나이": clean_metadata(obj.get("max_age")),  # 새 필드 추가
            "성별": clean_metadata(obj.get("gender")),       # 새 필드 추가
            "키워드": clean_metadata(obj.get("keywords"))    # 새 필드 추가
        }
        documents_dict[service_id] = Document(page_content=text, metadata=metadata)
documents = list(documents_dict.values())
print(f"총 {len(documents)}개의 고유 문서가 로드됨.")

# 의미 기반 청킹 (중복 제거 강화)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunked_documents = []
seen_service_ids = {}
for doc in documents:
    service_id = doc.metadata["서비스ID"]
    if service_id not in seen_service_ids:
        chunks = text_splitter.split_text(doc.page_content)
        chunked_documents.append(Document(page_content=chunks[0], metadata=doc.metadata))
        seen_service_ids[service_id] = True
print(f"청크 분할 및 중복 제거 후 총 {len(chunked_documents)}개의 문서(청크) 생성됨.")

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 벡터스토어 경로 설정
persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\meanchunking"
faiss_index_path = os.path.join(persist_directory, "index.faiss")

# FAISS 인덱스 생성 또는 로드
if os.path.exists(faiss_index_path):
    try:
        vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
        print("기존 FAISS 벡터스토어 로드 성공.")
        print(f"FAISS 인덱스 차원: {vectorstore.index.d}")
    except Exception as e:
        print(f"FAISS 로드 중 오류 발생: {e}")
        traceback.print_exc()
        raise
else:
    print("FAISS 인덱스가 존재하지 않음. 새로 생성합니다...")
    try:
        vectorstore = FAISS.from_documents(chunked_documents, embedding_model)
        vectorstore.save_local(persist_directory)
        print("FAISS 벡터스토어 생성 및 저장 성공.")
        print(f"FAISS 인덱스 차원: {vectorstore.index.d}")
    except Exception as e:
        print(f"FAISS 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise

# 리트리버 생성 (상위 15개 결과 가져와 중복 제거 후 10개 선택)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
print("리트리버 생성 완료 (상위 15개 결과 설정).")

# LLM (Cohere) 초기화
try:
    llm = ChatCohere()
    print("LLM 초기화 완료.")
except Exception as e:
    print(f"LLM 초기화 중 오류 발생: {e}")
    traceback.print_exc()
    raise

# 프롬프트 설정 (중복 통합 강화)
system_prompt = (
    "아래 컨텍스트에 포함된 상위 10개의 문서를 바탕으로 질문에 대한 답변을 작성하십시오. "
    "각 문서의 주요 정보를 번호를 붙여 간략히 요약하되, 동일한 서비스ID를 가진 문서는 반드시 하나로 통합하여 중복 없이 요약하십시오. "
    "최종 응답에서 중복된 내용이나 문장이 절대 반복되지 않도록 주의하며, 관련 없는 문서는 명시적으로 제외하십시오. "
    "컨텍스트: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 체인 생성
try:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("체인 생성 완료.")
except Exception as e:
    print(f"체인 생성 중 오류 발생: {e}")
    traceback.print_exc()
    raise

# LLM 응답 후처리 (중복 제거 강화)
def postprocess_response(response_text, threshold=0.9):
    sentences = re.split(r'(?<=[.!?])\s+', response_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return response_text
    
    service_id_pattern = r"(서비스ID:\s*[A-Za-z0-9]+)"
    seen_service_ids = set()
    filtered_sentences = []
    
    vectorizer = TfidfVectorizer().fit(sentences)
    tfidf_matrix = vectorizer.transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    for i, sentence in enumerate(sentences):
        service_id_match = re.search(service_id_pattern, sentence)
        service_id = service_id_match.group(0) if service_id_match else None
        
        if service_id and service_id in seen_service_ids:
            continue
        
        is_unique = all(sim_matrix[i, j] <= threshold for j in range(i + 1, len(sentences)))
        if is_unique:
            filtered_sentences.append(sentence)
            if service_id:
                seen_service_ids.add(service_id)
    
    return " ".join(filtered_sentences)

# 쿼리 리스트 정의
queries = [
    "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘.",
    "전라남도 목포시에 사는 48세 중년 남자에게 맞는 귀어 관련 창업에 대해 알려줘.",
    "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘."
]

# 쿼리 실행 (최소 10개 문서 보장)
for i, query in enumerate(queries, 1):
    print(f"\n=== 쿼리 {i}: {query} ===")
    try:
        retrieved_docs = retriever.invoke(query)
        print(f"검색된 문서 수: {len(retrieved_docs)}")
        
        # 서비스ID 기준 고유 문서 선택
        unique_docs = {}
        for doc in retrieved_docs:
            service_id = doc.metadata.get("서비스ID", "정보 없음")
            if service_id not in unique_docs:
                unique_docs[service_id] = doc
        top_docs = list(unique_docs.values())[:10]  # 상위 10개만 선택 (15개 중에서)
        
        print(f"중복 제거 후 상위 {len(top_docs)}개의 문서(서비스) 출력")
        for j, doc in enumerate(top_docs, 1):
            print(f"문서 {j}: {doc.metadata.get('서비스명', '정보 없음')} (서비스ID: {doc.metadata.get('서비스ID', '정보 없음')})")
        
        response = chain.invoke({"input": query})
        processed_answer = postprocess_response(response['answer'])
        print("응답:")
        print(processed_answer)
    except Exception as e:
        print(f"쿼리 실행 중 오류 발생: {e}")
        traceback.print_exc()