import os
import json
from dotenv import load_dotenv
from langsmith import Client

# --- Langsmith를 이용한 Query Translation (rag-fusion-query-generation) ---
# 1. 환경 변수 로드: api.env 파일에서 LANGSMITH_API_KEY를 읽어옵니다.
load_dotenv('/Users/minjoo/Desktop/SeSac/final/api.env')
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY를 찾을 수 없습니다. api.env 파일을 확인하세요.")

# 2. Langsmith 클라이언트 초기화 및 프롬프트 가져오기
client = Client(api_key=LANGSMITH_API_KEY)
prompt_data = client.pull_prompt("langchain-ai/rag-fusion-query-generation", include_model=True)
print("불러온 prompt_data:", prompt_data)

# 3. 원래 쿼리 (예: '서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘.')
original_query = "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘. "

# 4. Query Translation 단계 (여기서는 prompt_data를 이용해 여러 검색 쿼리 생성 가능)
# 실제로 LLM 호출을 통해 변환할 수 있으나, 예제에서는 단순히 원래 쿼리를 그대로 사용합니다.
def translate_query(query, prompt_data):
    # prompt_data의 템플릿을 활용하여 쿼리를 변환하는 로직을 넣을 수 있습니다.
    # 예시: translated_query = prompt_data.format(original_query=query)
    # 여기서는 간단히 동일한 쿼리를 반환합니다.
    return query

translated_query = translate_query(original_query, prompt_data)
print("Translated Query:", translated_query)

# --- 임베딩/랭킹(re-rank) 방식 적용 ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever

# 5. JSON 파일 로드: 문서 데이터 (원본 JSON 파일)
json_path = "/Users/minjoo/Desktop/SeSac/final/20250304.json"
with open(json_path, "r", encoding="utf-8") as f:
    json_docs = json.load(f)
print(f"로드된 문서 개수: {len(json_docs)}")


# 6.
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={'device': 'cpu'},  # CUDA 대신 CPU 사용
    encode_kwargs={'normalize_embeddings': True}
)


# 7. FAISS 벡터스토어 로드 (이미 임베딩된 벡터가 저장된 디렉토리)
persist_directory = "/Users/minjoo/Desktop/SeSac/final"
vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 8. 문서 압축 및 re-rank를 위한 CohereRerank 설정
compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=15)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 9. re-rank된 문서 검색 (translated_query를 이용)
reranked_docs = compression_retriever.get_relevant_documents(translated_query)
print(f"\nRe-rank 후 최종 문서 개수: {len(reranked_docs)}")

def rag_fusion_final(query, docs):
    # 상위 N개의 문서를 선택 (여기서는 15개)
    top_docs = docs[:15]
    # 각 문서의 "서비스ID", "서비스명", "서비스목적요약", "지원내용"을 결합
    docs_text = "\n\n".join(
        f"서비스ID: {doc.metadata.get('서비스ID', 'N/A')}\n"
        f"서비스명: {doc.metadata.get('서비스명', 'N/A')}\n"
        f"{((doc.metadata.get('서비스목적요약') or '') + ' ' + (doc.metadata.get('지원내용') or '')).strip()}"
        for doc in top_docs
    )
    final_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nAnswer:"
    return final_prompt

# --- 최종 RAG Fusion 프롬프트 생성 ---
# def rag_fusion_final(query, docs):
#     # 상위 N개의 문서를 선택 (여기서는 15개)
#     top_docs = docs[:15]
#     # 각 문서의 "서비스ID", "서비스명", "서비스목적요약", "지원내용"을 결합
#     docs_text = "\n\n".join(
#         f"서비스ID: {doc.get('서비스ID', 'N/A')}\n"
#         f"서비스명: {doc.get('서비스명', 'N/A')}\n"
#         f"{((doc.get('서비스목적요약') or '') + ' ' + (doc.get('지원내용') or '')).strip()}"
#         for doc in top_docs
#     )
#     final_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nAnswer:"
#     return final_prompt

final_prompt = rag_fusion_final(translated_query, reranked_docs)
print("\n최종 프롬프트:")
print(final_prompt)