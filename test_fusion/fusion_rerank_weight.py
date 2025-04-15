import os
import json
import re
from dotenv import load_dotenv
from langsmith import Client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever

# --- 환경 변수 로드 ---
load_dotenv('/Users/minjoo/Desktop/SeSac/final/api.env')
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY를 찾을 수 없습니다. api.env 파일을 확인하세요.")

# --- Langsmith 클라이언트 초기화 및 프롬프트 가져오기 ---
client = Client(api_key=LANGSMITH_API_KEY)
prompt_data = client.pull_prompt("langchain-ai/rag-fusion-query-generation", include_model=True)
print("불러온 prompt_data:", prompt_data)

# --- 원래 쿼리 ---
original_query = "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘."

def translate_query(query, prompt_data):
    """Query Translation 수행 (RAG Fusion 방식)"""
    response = prompt_data.invoke({"original_query": query})
    output_text = response.messages[-1].content  # 마지막 메시지
    return output_text

translated_output = translate_query(original_query, prompt_data)
print("\n[Translated Output]")
print(translated_output)

def pick_first_query(output_text):
    """번역된 쿼리에서 첫 번째 문장을 선택"""
    match = re.search(r"1\)\s*(.+)", output_text)
    return match.group(1).strip() if match else output_text

translated_query = pick_first_query(translated_output)

# ✅ **Query에 "도봉" 추가 (검색 정확도 향상)**
translated_query = translated_query + " 도봉"

print("\n[Final Query with Region Keyword]")
print(translated_query)

# --- JSON 파일 로드 ---
json_path = "/Users/minjoo/Desktop/SeSac/final/20250304.json"
with open(json_path, "r", encoding="utf-8") as f:
    json_docs = json.load(f)
print(f"로드된 문서 개수: {len(json_docs)}")

# --- FAISS 임베딩 모델 설정 ---
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={'device': 'cpu'},  # Mac 환경에서 CPU 사용
    encode_kwargs={'normalize_embeddings': True}
)

# --- FAISS 벡터스토어 로드 ---
persist_directory = "/Users/minjoo/Desktop/SeSac/final"
vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# --- Re-rank 설정 ---
compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=15)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# --- Re-rank 수행 ---
reranked_docs = compression_retriever.get_relevant_documents(translated_query)
print(f"\nRe-rank 후 최종 문서 개수: {len(reranked_docs)}")

# --- ✅ 지역 가중치 적용 ---
REGION_KEYWORD = "서울"  # 도봉구 관련 문서 점수를 높이기 위한 키워드

def rerank_with_region_weight(docs, region_keyword, weight=5):
    """
    '소관기관명'에 특정 지역 키워드(예: '도봉')가 포함되면 가중치 적용.
    """
    ranked_docs = []
    
    for doc in docs:
        base_score = doc.metadata.get("score", 1) or 1  # 기본 점수, None 방지
        agency_name = doc.metadata.get("소관기관명", "") or ""  # None 방지
        region_score = weight if region_keyword in agency_name else 0
        total_score = base_score + region_score
        ranked_docs.append((total_score, doc))

    # 점수가 높은 순서대로 정렬
    ranked_docs.sort(key=lambda x: x[0], reverse=True)

    # 정렬된 결과 확인
    print("\n[Sorted Documents After Re-Ranking]")
    for score, doc in ranked_docs[:10]:  # 상위 10개만 확인
        print(f"Score: {score}, 소관기관명: {doc.metadata.get('소관기관명', '')}")

    return [doc for _, doc in ranked_docs]

# --- 지역 가중치 적용 ---
final_reranked_docs = rerank_with_region_weight(reranked_docs, REGION_KEYWORD, weight=5)

print(f"\n지역 가중치 적용 후 최종 문서 개수: {len(final_reranked_docs)}")

# --- ✅ 최종 RAG Fusion 프롬프트 생성 ---
def rag_fusion_final(query, docs):
    """ 최종 프롬프트 생성 (소관기관명 포함) """
    top_docs = docs[:15]  # 상위 15개 문서 선택
    
    docs_text = "\n\n".join(
        f"서비스ID: {doc.metadata.get('서비스ID', 'N/A')}\n"
        f"서비스명: {doc.metadata.get('서비스명', 'N/A')}\n"
        f"소관기관명: {doc.metadata.get('소관기관명', 'N/A')}\n"
        f"{((doc.metadata.get('서비스목적요약') or '') + ' ' + (doc.metadata.get('지원내용') or '')).strip()}"
        for doc in top_docs
    )
    
    final_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nAnswer:"
    return final_prompt

final_prompt = rag_fusion_final(translated_query, final_reranked_docs)
print("\n최종 프롬프트:")
print(final_prompt)
