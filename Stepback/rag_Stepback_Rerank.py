# test
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from typing import List, Dict, Any

# .env 파일 로드
load_dotenv()

# API 토큰 체크
if not os.getenv("HUGGINGFACE_HUB_TOKEN") or not os.getenv("COHERE_API_KEY"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN 또는 COHERE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

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

# JSON 데이터 로드 및 벡터 DB 설정 함수
def setup_vectordb(json_file_path, persist_directory):
    embedding_model = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    try:
        if os.path.exists(os.path.join(persist_directory, "index.faiss")):
            print("기존 벡터 DB를 로드합니다...")
            return FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
        
        print("새로운 벡터 DB를 생성합니다...")
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if not isinstance(json_data, list):
            raise ValueError("JSON 데이터가 리스트 형태가 아닙니다.")

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

        os.makedirs(persist_directory, exist_ok=True)  # 디렉토리 생성
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(persist_directory)
        print(f"총 {len(documents)}개의 문서가 FAISS 벡터DB에 저장됨.")
        return vectorstore
    except Exception as e:
        raise ValueError(f"벡터 DB 설정 중 오류 발생: {e}")

# Step-Back 질문 생성 함수
def generate_step_back_query(original_query: str, llm) -> str:
    step_back_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 구체적인 질문에서 핵심 개념을 추출하여 더 일반적이고 추상적인 질문을 만드는 전문가입니다. "
         "원래 질문의 맥락을 유지하면서 더 광범위한 관점에서 접근할 수 있는 질문을 만들어주세요."),
        ("human", 
         "다음 질문에서 핵심 개념을 추출하고 더 일반적인 질문으로 바꿔주세요. 반드시 질문 형태로 작성해주세요.\n\n"
         "원래 질문: {original_query}")
    ])
    response = llm.invoke(step_back_prompt.format(original_query=original_query))
    return response.content

# 문서 ID 추출 함수 (중복 제거용)
def get_doc_id(doc):
    return doc.metadata.get('서비스ID', doc.page_content[:100])

# 통합 RAG 파이프라인 클래스
class StepBackRAG:
    def __init__(self, vectorstore_path, json_data_path):
        # 벡터스토어 설정
        self.vectorstore = setup_vectordb(json_data_path, vectorstore_path)
        print(f"Vectorstore 초기화 완료: {self.vectorstore.index.ntotal} 문서 포함")
        
        # LLM 설정
        self.llm = ChatCohere(temperature=0.1)
        
        # 기본 검색기
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 50})
        
        # 리랭커
        self.compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=30)
        self.reranker = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )
        
        # 응답 생성 프롬프트
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 공공 서비스 안내 전문가입니다. 주어진 정보를 바탕으로 사용자의 질문에 친절하고 정확하게 답변해주세요.\n"
             "답변에는 서비스명, 지원내용, 지원대상, 신청방법, 신청기한 등의 중요 정보를 꼭 포함시켜주세요.\n"
             "정보가 없는 경우 없다고 솔직히 답변하세요.\n\n"
             "참고 정보: {context}"),
            ("human", "{query}")
        ])
        
        # 응답 생성 체인
        self.doc_chain = create_stuff_documents_chain(self.llm, self.response_prompt)
    
    def retrieve_with_step_back(self, query: str):
        original_docs = self.base_retriever.invoke(query)
        print(f"원본 쿼리로 검색된 문서 수: {len(original_docs)}")
        try:
            step_back_query = generate_step_back_query(query, self.llm)
            print(f"원본 질문: {query}")
            print(f"추상화 질문: {step_back_query}")
            step_back_docs = self.base_retriever.invoke(step_back_query)
            print(f"추상화 쿼리로 검색된 문서 수: {len(step_back_docs)}")
            
            doc_ids = set()
            combined_docs = []
            for doc in original_docs:
                doc_id = get_doc_id(doc)
                if doc_id not in doc_ids:
                    doc_ids.add(doc_id)
                    combined_docs.append(doc)
            for doc in step_back_docs:
                doc_id = get_doc_id(doc)
                if doc_id not in doc_ids:
                    doc_ids.add(doc_id)
                    combined_docs.append(doc)
        except Exception as e:
            print(f"추상화 질문 생성 중 오류 발생: {e}")
            combined_docs = original_docs
        
        reranked_docs = self.reranker.invoke(query)
        print("리랭킹된 상위 10개 문서:")
        for i, doc in enumerate(reranked_docs[:10], 1):
            score = doc.metadata.get("relevance_score", "N/A")
            print(f"{i}. {doc.metadata['서비스명']} (서비스ID: {doc.metadata['서비스ID']}, 점수: {score})")
        
        if len(reranked_docs) < 10:
            additional_docs = original_docs[:10 - len(reranked_docs)]
            reranked_docs.extend(additional_docs)
       
        return reranked_docs[:10]
    
    def answer_query(self, query: str) -> str:
        docs = self.retrieve_with_step_back(query)
        if not docs:
            return "죄송합니다. 질문에 관련된 서비스 정보를 찾을 수 없습니다."
        response = self.doc_chain.invoke({
            "query": query,
            "context": docs
        })
        return response

# 사용 예시
if __name__ == "__main__":
    json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\20250304.json"
    persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ"
    
    rag = StepBackRAG(persist_directory, json_file_path)
    test_query = "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘."
    response = rag.answer_query(test_query)
    
    print("\n===== 최종 응답 =====")
    print(response)