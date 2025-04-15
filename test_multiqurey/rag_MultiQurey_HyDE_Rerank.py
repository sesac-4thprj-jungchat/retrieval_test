# test
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
import traceback

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

# FAQ 스타일 텍스트 생성 함수
def generate_selected_text(obj):
    return (
        f"Q: 서비스 이름은 무엇인가요?\nA: {clean_metadata(obj.get('서비스명'))}\n\n"
        f"Q: 서비스 ID는 무엇인가요?\nA: {clean_metadata(obj.get('서비스ID'))}\n\n"
        f"Q: 제공 부서는 어디인가요?\nA: {clean_metadata(obj.get('부서명'))}\n\n"
        f"Q: 서비스 분야는 무엇인가요?\nA: {clean_metadata(obj.get('서비스분야'))}\n\n"
        f"Q: 서비스 목적은 무엇인가요?\nA: {clean_metadata(obj.get('서비스목적요약'))}\n\n"
        f"Q: 지원 내용은 무엇인가요?\nA: {clean_metadata(obj.get('지원내용'))}\n\n"
        f"Q: 선정 기준은 무엇인가요?\nA: {clean_metadata(obj.get('선정기준'))}\n\n"
        f"Q: 신청 기한은 언제까지인가요?\nA: {clean_metadata(obj.get('신청기한'))}\n\n"
        f"Q: 신청 방법은 무엇인가요?\nA: {clean_metadata(obj.get('신청방법'))}\n\n"
        f"Q: 접수 기관은 어디인가요?\nA: {clean_metadata(obj.get('접수기관'))}\n\n"
    )

# JSON 데이터 로드
json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\20250304.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

if not isinstance(json_data, list):
    raise ValueError("JSON 데이터가 리스트 형태가 아닙니다.")

# Document 객체로 변환
documents = []
for obj in json_data:
    text = generate_selected_text(obj)
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

print(f"총 {len(documents)}개의 문서가 로드됨.")

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 벡터스토어 경로
persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ"
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
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(persist_directory)
        print("FAISS 벡터스토어 생성 및 저장 성공.")
        print(f"FAISS 인덱스 차원: {vectorstore.index.d}")
    except Exception as e:
        print(f"FAISS 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise

# 기본 FAISS 리트리버 생성 (상위 10개로 설정)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("기본 FAISS 리트리버 생성 완료 (상위 10개 설정).")

# LLM (ChatCohere) 설정
print("ChatCohere LLM 초기화 시작...")
try:
    llm = ChatCohere(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="command",
        temperature=0.7,
        max_tokens=512
    )
    print("ChatCohere LLM 초기화 완료.")
except Exception as e:
    print(f"ChatCohere LLM 초기화 중 오류 발생: {e}")
    traceback.print_exc()
    raise

# MultiQueryRetriever를 위한 프롬프트 정의
multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="질문: {question}\n다양한 관점에서 3개의 검색 쿼리를 생성:\n1. [첫 번째 쿼리]\n2. [두 번째 쿼리]\n3. [세 번째 쿼리]"
)

# MultiQueryRetriever 설정 (parser 제거, 상위 30개 제한)
print("MultiQueryRetriever 설정 시작...")
try:
    multi_query_retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=base_retriever,
        prompt=multi_query_prompt
    )
    # MultiQueryRetriever의 invoke 메서드를 오버라이드하여 상위 30개만 반환
    original_invoke = multi_query_retriever.invoke
    def limited_invoke(query):
        docs = original_invoke(query)
        return docs[:30]  # 상위 30개로 제한
    multi_query_retriever.invoke = limited_invoke
    print("MultiQueryRetriever 생성 완료 (상위 30개 제한).")
except Exception as e:
    print(f"MultiQueryRetriever 설정 중 오류 발생: {e}")
    traceback.print_exc()
    raise

# HyDE 프롬프트 정의 (간결하게 수정)
hyde_template = """질문: {question}\n가상 문서 (간결히):"""
prompt_hyde = ChatPromptTemplate.from_template(hyde_template)

# HyDE 가상 문서 생성 체인
generate_hyde_docs = (
    prompt_hyde | llm | StrOutputParser()
)

# HyDE + MultiQueryRetriever 결합
def hyde_multi_query_retrieval(query):
    # HyDE로 가상 문서 생성
    hyde_doc = generate_hyde_docs.invoke({"question": query})
    print(f"HyDE 가상 문서 생성 완료:\n{hyde_doc[:200]}...")
    # MultiQueryRetriever로 검색 (상위 30개 제한 적용됨)
    retrieved_docs = multi_query_retriever.invoke(query)
    return retrieved_docs

# Re-Ranker 설정 (상위 10개로 재정렬)
try:
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )
    print("Re-Ranker 및 압축 리트리버 생성 완료 (상위 10개 설정).")
except Exception as e:
    print(f"Re-Ranker 설정 중 오류 발생: {e}")
    traceback.print_exc()
    raise

# RAG 프롬프트 설정 (간결히 수정)
system_prompt = (
    "주어진 컨텍스트에서 검색된 상위 10개 문서를 나열하여 질문에 한국어로 답변하세요. "
    "각 문서의 서비스명, 서비스ID, 지원내용을 간략히 요약해 번호 붙여 제공하세요. "
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
    print("HyDE, MultiQuery, CohereRerank이 적용된 체인 생성 완료.")
except Exception as e:
    print(f"체인 생성 중 오류 발생: {e}")
    traceback.print_exc()
    raise

# 쿼리 리스트 정의
queries = [
    "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘.",
    "전라남도 목포시에 사는 48세 중년 남자에게 맞는 귀어 관련 창업에 대해 알려줘.",
    "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘"
]

# 각 쿼리를 따로 실행
for i, query in enumerate(queries, 1):
    print(f"\n=== 쿼리 {i}: {query} ===")
    try:
        # HyDE + MultiQueryRetriever 검색
        retrieved_docs = hyde_multi_query_retrieval(query)
        print(f"검색된 문서 수: {len(retrieved_docs)}")
        for j, doc in enumerate(retrieved_docs, 1):
            print(f"문서 {j}: {doc.metadata['서비스명']} (서비스ID: {doc.metadata['서비스ID']})")

        # 체인 실행
        response = chain.invoke({"input": query})
        print("응답:")
        print(response['answer'])
    except Exception as e:
        print(f"쿼리 실행 중 오류 발생: {e}")
        traceback.print_exc()