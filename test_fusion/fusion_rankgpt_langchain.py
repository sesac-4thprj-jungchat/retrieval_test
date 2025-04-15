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
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

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

if use_existing_faiss:
    print(f"✅ 기존 FAISS 인덱스 로드 중: {index_file}")
else:
    print("⚠️ FAISS 인덱스가 존재하지 않습니다. 새로 생성합니다.")

# ✅ HuggingFace 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},  # 'torch_dtype' 제거
    encode_kwargs={'normalize_embeddings': True}
)

# ✅ FAISS 인덱스 로드 또는 생성
if use_existing_faiss:
    vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
else:
    # ✅ JSON 파일 로드
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 데이터 파일을 찾을 수 없습니다: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_docs = json.load(f)
    
    print(f"🔍 JSON 문서 로드 완료: 총 {len(json_docs)}개 문서")

    # ✅ 문서 변환 및 FAISS 생성
    documents = []
    for obj in json_docs:
        try:
            # ✅ 필수 키 검증 및 기본값 처리
            text = (
                f"서비스명: {obj.get('서비스명', '정보 없음')}\n"
                f"서비스ID: {obj.get('서비스ID', '정보 없음')}\n"
                f"서비스목적요약: {obj.get('서비스목적요약', '정보 없음')}\n"
                f"지원내용: {obj.get('지원내용', '정보 없음')}\n"
            )
            documents.append(Document(page_content=text))  # metadata 최소화하여 FAISS 저장 문제 방지
        
        except Exception as e:
            print(f"❌ 오류 발생: {e} → 해당 객체: {obj}")

    # ✅ 텍스트 분할 후 FAISS 인덱스 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embedding_model)
    vectorstore.save_local(persist_directory)
    print("✅ 새로운 FAISS 인덱스 저장 완료!")

# ✅ Retriever 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# ✅ RankLLM 설정 (RankGPT)
compressor = RankLLMRerank(
    top_n=2,  
    model="gpt",  
    gpt_model="gpt-4o"  
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

print("✅ RankLLM 설정 완료")

# ✅ LLM (GPT-4o) 설정
llm = ChatOpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

# ✅ 질의 실행
query = "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘. "
print(f"🔍 질의 수행 중: {query}")

retrieved_docs = retriever.invoke(query)
print(f"✅ {len(retrieved_docs)}개의 문서 검색 완료!")

# ✅ RAG-Fusion 최종 출력 형식 변환
def rag_fusion_final(query, docs):
    top_docs = docs[:15]
    docs_text = "\n\n".join(
        f"서비스ID: {doc.metadata.get('서비스ID', 'N/A')}\n"
        f"서비스명: {doc.metadata.get('서비스명', 'N/A')}\n"
        f"{((doc.metadata.get('서비스목적요약') or '') + ' ' + (doc.metadata.get('지원내용') or '')).strip()}"
        for doc in top_docs
    )
    final_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nAnswer:"
    return final_prompt

# ✅ 최종 실행 결과
final_output = rag_fusion_final(query, retrieved_docs)
print("\n🔹 RAG-Fusion 최종 출력:\n")
print(final_output)

# ✅ 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # GPU 캐시 정리
elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()  # Mac MPS 캐시 정리

gc.collect()  # 가비지 컬렉션 실행
print("✅ 리소스 정리 완료!")