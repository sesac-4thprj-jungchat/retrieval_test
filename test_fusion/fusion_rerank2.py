import os
import json
import torch
import traceback
from dotenv import load_dotenv

# LangChain 및 관련 라이브러리
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.retrievers import ReciprocalRankFusionRetriever

from llama_index.core.retrievers import RecursiveRetriever  # ✅ 최신 버전 호환


# LlamaIndex 관련 라이브러리
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.query_engine import RetrieverQueryEngine

# Transformers 및 CrossEncoder 임포트
from transformers import AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder

# .env 파일 로드
load_dotenv()
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# 1️⃣ 내부 LLM 설정 (Mistral-7B 사용)
model_name = "mistralai/Mistral-7B-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"
model = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_new_tokens=256, device=device)

# 2️⃣ 헬퍼 함수
def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

def generate_text(obj):
    return f"서비스명: {clean_metadata(obj.get('서비스명'))}\n서비스ID: {clean_metadata(obj.get('서비스ID'))}"

# 3️⃣ JSON 데이터 로드 및 Document 객체 생성
json_file_path = "/Users/minjoo/Desktop/SeSac/final/20250304.json"  # 실제 경로로 변경
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"JSON 파일 '{json_file_path}'을 찾을 수 없습니다.")

documents = [Document(page_content=generate_text(obj), metadata={k: clean_metadata(obj.get(k)) for k in obj}) for obj in json_data]
print(f"총 {len(documents)}개의 문서가 로드됨.")

# 4️⃣ FAISS 벡터스토어 생성/로드
embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko", model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
persist_directory = "faiss_db"

try:
    vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
    print("기존 FAISS 벡터스토어 로드 성공.")
except:
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(persist_directory)
    print(f"총 {len(documents)}개의 문서가 FAISS 벡터 DB에 저장됨.")

# 5️⃣ Cross-Encoder Re-rank 설정
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)

def cross_encoder_rerank(query, documents, top_n=10):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
    return sorted_docs[:top_n]

# 6️⃣ RAG Fusion + MultiQueryRetriever 설정
index = VectorStoreIndex.from_vector_store(vectorstore)
service_context = ServiceContext.from_defaults(llm=model)

base_retriever = index.as_retriever(search_kwargs={"k": 30})
multi_query_prompt = PromptTemplate(input_variables=["question"], template="질문: {question}\n3개의 검색 쿼리 생성:\n1. [첫 번째 쿼리]\n2. [두 번째 쿼리]\n3. [세 번째 쿼리]")

multi_query_retriever = MultiQueryRetriever.from_llm(llm=model, retriever=base_retriever, prompt=multi_query_prompt)

# ✅ 최신 SimilarityFusionRetriever 적용 (RAG Fusion)
rag_fusion_retriever = RecursiveRetriever(base_retriever=multi_query_retriever, search_kwargs={"k": 15})

# 7️⃣ Cohere Rerank 적용
cohere_compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
retriever_cohere_compressed = ContextualCompressionRetriever(base_compressor=cohere_compressor, base_retriever=rag_fusion_retriever)

# 8️⃣ 최종 RAG Query Engine 구성
query_engine = RetrieverQueryEngine.from_args(retriever=retriever_cohere_compressed, service_context=service_context)

# 9️⃣ LLM 응답 생성 체인
system_prompt = "검색된 문서에서, 서비스명과 서비스ID를 번호별로 출력하세요.\n출력 형식: [번호]. 서비스명 (서비스ID)"
rag_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{context}")])
combine_docs_chain = create_stuff_documents_chain(model, rag_prompt)
chain = create_retrieval_chain(retriever_cohere_compressed, combine_docs_chain)

print("🔹 MultiQuery + RAG Fusion (Reciprocal Rank Fusion) + Re-rank (CrossEncoder + Cohere) 설정 완료!")

# 🔟 테스트 쿼리 실행
if __name__ == "__main__":
    queries = [
        "서울 도봉구에 사는 34세 청년 남성에게 맞는 서비스를 찾아줘.",
        "전라남도 목포시에 사는 48세 중년 남성에게 맞는 귀어 관련 창업 정책을 알려줘.",
        "강원도 동해시에 사는 68세 노인 여성에게 맞는 정책을 찾아줘.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n=== 쿼리 {i}: {query} ===")
        try:
            response = query_engine.query(query)
            print("🔹 최종 응답:")
            print(response)
        except Exception as e:
            print(f"쿼리 실행 중 오류 발생: {e}")
            traceback.print_exc()