from langsmith import Client

import os
import json
from dotenv import load_dotenv

# 임포트: FAISS 벡터스토어, HuggingFaceEmbeddings, Document, 리트리버, 압축(re-rank) 등
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere

# 1. 환경 변수 로드: api.env 파일에서 LANGSMITH_API_KEY를 읽어옵니다.

load_dotenv('/Users/minjoo/Desktop/SeSac/final/api.env')
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY를 찾을 수 없습니다. api.env 파일을 확인하세요.")

# 2. Langsmith 클라이언트 초기화 및 프롬프트 가져오기
client = Client(api_key=LANGSMITH_API_KEY)
prompt_data = client.pull_prompt("langchain-ai/rag-fusion-query-generation", include_model=True)
print("불러온 prompt_data:", prompt_data)

# 3. JSON 파일 로드: rag fusion에 사용할 데이터 (문서 목록 등)
json_path = "/Users/minjoo/Desktop/SeSac/final/20250304.json"
with open(json_path, "r", encoding="utf-8") as f:
    documents = json.load(f)
print(f"로드된 문서 개수: {len(documents)}")

# 4. 기존의 rank_documents 함수 (단순 랭킹 방식)
def rank_documents(query, docs):
    query_terms = set(query.lower().split())
    ranked_docs = []
    for doc in docs:
        # '서비스목적요약'과 '지원내용' 필드 값이 None일 경우 빈 문자열로 대체
        service_summary = doc.get("서비스목적요약") or ""
        support_content = doc.get("지원내용") or ""
        doc_text = (service_summary + " " + support_content).lower()
        score = sum(1 for term in query_terms if term in doc_text)
        ranked_docs.append((score, doc))
    ranked_docs.sort(key=lambda x: x[0], reverse=True)
    return ranked_docs

# 5. 재정렬(rerank) 함수: prompt_data에 rerank 메서드가 있다면 이를 사용하고,
# 없으면 기존 rank_documents 함수를 사용하도록 합니다.
def rerank_documents(query, docs, prompt_data):
    if hasattr(prompt_data, 'rerank'):
        # 예시: prompt_data.rerank(query, docs)
        return prompt_data.rerank(query, docs)
    else:
        return rank_documents(query, docs)

# 6. RAG Fusion 함수: 상위 문서를 결합하여 최종 프롬프트 생성
def rag_fusion(query, docs, prompt_data):
    # rerank_documents 함수를 사용해 문서를 재정렬
    ranked_docs = rerank_documents(query, docs, prompt_data)
    # 상위 3개 문서 선택 (필요에 따라 조정)
    top_docs = [doc for score, doc in ranked_docs[:15]]
    # 각 문서의 "서비스ID"와 "서비스명", 그리고 "서비스목적요약", "지원내용"을 결합하여 텍스트 생성
    docs_text = "\n\n".join(
        f"서비스ID: {doc.get('서비스ID', 'N/A')}\n"
        f"서비스명: {doc.get('서비스명', 'N/A')}\n"
        f"{((doc.get('서비스목적요약') or '') + ' ' + (doc.get('지원내용') or '')).strip()}"
        for doc in top_docs
    )
    final_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nAnswer:"
    return final_prompt

# 7. 사용자 쿼리 설정 (실제 사용할 query 문구로 변경)
query = "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘."
final_prompt = rag_fusion(query, documents, prompt_data)
print("\n최종 프롬프트:")
print(final_prompt)