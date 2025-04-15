# test
import os
import json
import re
import torch
import cohere
import traceback
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from langchain.llms.base import LLM
from pydantic import PrivateAttr

# 커스텀 LLM 래퍼 정의
class CustomHuggingFacePipeline(LLM):
    _pipeline: any = PrivateAttr()

    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline

    def _call(self, prompt: str, stop=None) -> str:
        result = self._pipeline(prompt)
        return result[0]["generated_text"]

    @property
    def _identifying_params(self):
        return {"pipeline": self._pipeline}

    @property
    def _llm_type(self) -> str:
        return "custom_hf_pipeline"

# .env 파일 로드 및 API 토큰 체크
load_dotenv('api.env')
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# 헬퍼 함수 정의
def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

# FAQ + 자연어 문장 스타일 결합 함수
def generate_selected_text(obj):
    return (
        # FAQ 스타일
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
        # 자연어 문장 스타일
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

# 1. 내부 LLM (AutoGPTQForCausalLM) 설정
model_name = "TheBloke/openchat-3.5-1210-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    revision="main",
    use_safetensors=True,
    device=device,
    trust_remote_code=True,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
)
llm_gptq = CustomHuggingFacePipeline(pipeline=pipe)
print("AutoGPTQ 기반 LLM (커스텀 래퍼) 초기화 완료.")

# 2. JSON 데이터 로드 및 Document 객체 생성
json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\20250304.json"
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"JSON 파일 '{json_file_path}'을 찾을 수 없습니다.")
except json.JSONDecodeError:
    raise ValueError(f"JSON 파일 '{json_file_path}'의 형식이 잘못되었습니다.")

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
        "접수기관": clean_metadata(obj.get("접수기관")),
    }
    documents.append(Document(page_content=text, metadata=metadata))
print(f"총 {len(documents)}개의 문서가 로드됨.")

# 3. 임베딩 모델 및 FAISS 벡터스토어 생성/로드
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ"
faiss_index_path = os.path.join(persist_directory, "index.faiss")

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

# 4. 검색 및 리트리버 체인 구성
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("기본 FAISS 리트리버 생성 완료 (상위 10개 설정).")

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "질문: {question}\n"
        "다양한 관점에서 3개의 검색 쿼리를 생성:\n"
        "1. [첫 번째 쿼리]\n"
        "2. [두 번째 쿼리]\n"
        "3. [세 번째 쿼리]"
    ),
)
multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=llm_gptq, retriever=base_retriever, prompt=multi_query_prompt
)
print("MultiQueryRetriever 생성 완료.")

compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
retriever_compressed = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=multi_query_retriever
)
print("Re-Ranker 및 압축 리트리버 생성 완료 (상위 10개 설정).")

system_prompt = (
    "검색된 상위 10개 문서의 정보에서, 오직 서비스명과 서비스ID만을 번호별로 출력하세요.\n"
    "출력 형식은 반드시 다음과 같아야 합니다:\n"
    "[번호]. 서비스명 (서비스ID)\n"
    "다른 어떠한 내용도 포함하지 말고, 줄바꿈도 최소화할 것."
)
rag_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{context}")]
)
combine_docs_chain = create_stuff_documents_chain(llm_gptq, rag_prompt)
chain = create_retrieval_chain(retriever_compressed, combine_docs_chain)
print("MultiQuery, CohereRerank이 적용된 체인 생성 완료.")

# 5. 테스트 쿼리 실행
if __name__ == "__main__":
    queries = [
        "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘.",
        "전라남도 목포시에 사는 48세 중년 남자에게 맞는 귀어 관련 창업에 대해 알려줘.",
        "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n=== 쿼리 {i}: {query} ===")
        try:
            retrieved_docs_mq = multi_query_retriever.get_relevant_documents(query)
            print(f"검색된 문서 수: {len(retrieved_docs_mq)}")

            reranked_docs = retriever_compressed.get_relevant_documents(query)
            print("Rerank된 상위 10개 문서:")
            for j, doc in enumerate(reranked_docs, 1):
                print(f"문서 {j}: {doc.metadata['서비스명']} (서비스ID: {doc.metadata['서비스ID']})")

            response_chain = chain.invoke({"input": query})
            print("체인 응답:")
            print(response_chain["answer"])
        except Exception as e:
            print(f"쿼리 실행 중 오류 발생: {e}")
            traceback.print_exc()