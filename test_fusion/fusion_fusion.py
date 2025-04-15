import os
import faiss
import json
import gc
import torch
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

import torch
import gc
import multiprocessing

# # 멀티프로세싱 안전하게 설정
# if __name__ == '__main__':
#     multiprocessing.set_start_method("spawn", force=True)

# ✅ .env 파일 로드
load_dotenv("api.env")
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("❌ HUGGINGFACE_HUB_TOKEN이 .env 파일에 없습니다.")

# ✅ FAISS 인덱스 경로
persist_directory = "/Users/minjoo/Desktop/SeSac/final/"
index_file = os.path.join(persist_directory, "index.faiss")

# ✅ JSON 데이터 파일 경로
json_path = "/Users/minjoo/Desktop/SeSac/final/20250304.json"

# ✅ FAISS 인덱스 존재 여부 확인
use_existing_faiss = os.path.exists(index_file)

# ✅ 임베딩 모델 설정 (가벼운 모델로 변경)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",  # 1024차원 모델
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)

# ✅ FAISS 인덱스 로드 또는 생성
if use_existing_faiss:
    print(f"✅ 기존 FAISS 인덱스 로드 중: {index_file}")
    vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)

    # ✅ FAISS 메모리 캐시 줄이기
    torch.set_grad_enabled(False)
    faiss.omp_set_num_threads(1)

    # ✅ FAISS 인덱스 최적화
    index = vectorstore.index
    print("✅ FAISS 인덱스 최적화 완료")

else:
    print("⚠️ 기존 FAISS 인덱스가 없어 새로 생성합니다.")

    # ✅ 새로운 FAISS 인덱스 생성
    index = faiss.IndexFlatL2(embedding_model.client.get_sentence_embedding_dimension())
    index = faiss.IndexIDMap(index)
    print("✅ 새로운 FAISS 인덱스 생성 완료")

# ✅ Retriever 설정
print("✅ Retriever 설정 시작")
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
print("✅ Retriever 설정 완료")

# ✅ RankLLM 설정 (RankGPT)
compressor = RankLLMRerank(
    top_n=2,
    model="gpt",
    gpt_model="gpt-3.5-turbo"  # 더 작은 모델로 교체
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
print("✅ RankLLM 설정 완료")

# ✅ RAG-Fusion 쿼리 생성
print("✅ RAG-Fusion 쿼리 생성 시작")
template = """Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

# generate_queries = (
#     prompt_rag_fusion 
#     | ChatOpenAI(temperature=0, max_tokens=512, streaming=True)
#     | StrOutputParser() 
#     | (lambda x: x.split("\n"))
# )
generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0, max_tokens=512, streaming=True)
    | StrOutputParser()
    | (lambda x: x.split("\n") if isinstance(x, str) else x)  # 여러 개의 쿼리가 생성될 가능성 있음
    | (lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)  # 🔹 리스트가 되면 첫 번째 값만 사용
)

print("✅ RAG-Fusion 쿼리 생성 완료")

# ✅ RAG-Fusion Reciprocal Rank Fusion
def reciprocal_rank_fusion(results: list[list], k=60):
    print("✅ RAG-Fusion: 시작")
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    return [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ✅ RAG-Fusion Retrieval Chain (병렬 검색 제거)
retrieval_chain_rag_fusion = (
    generate_queries
    | retriever  # 🔹 retriever.map() 대신 일반 retriever 사용
    | reciprocal_rank_fusion
)


# ✅ 최종 RAG-Fusion Query 실행
query = "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘."
print(f"🔍 질의 수행 중: {query}, 타입: {type(query)}") 

docs = retrieval_chain_rag_fusion.invoke({"question": query})
print(f"✅ {len(docs)}개의 문서 검색 완료!")

# ✅ LLM 응답 생성
template_rag = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt_rag = ChatPromptTemplate.from_template(template_rag)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": docs, "question": query}  # 직접 docs를 사용
    | prompt_rag
    | llm
    | StrOutputParser()
)

final_response = final_rag_chain.invoke({"docs": docs, "question": query})

print("\n🔹 최종 응답:\n")
print(final_response)

# ✅ 리소스 정리
gc.collect()  # 가비지 컬렉션 실행
print("✅ 리소스 정리 완료!")