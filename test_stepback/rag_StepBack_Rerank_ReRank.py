# test
import os
import json
import traceback
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import CrossEncoder
from langchain.llms.base import LLM
from pydantic import PrivateAttr, Field

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

class RerankingDataset(Dataset):
    def __init__(self, query, documents):
        self.query = query
        self.documents = documents
    def __len__(self):
        return len(self.documents)
    def __getitem__(self, idx):
        return self.query, self.documents[idx].page_content

load_dotenv()
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

model_name = "TheBloke/openchat-3.5-1210-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoGPTQForCausalLM.from_quantized(
    model_name, revision="main", use_safetensors=True, device=device, trust_remote_code=True
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
llm_gptq = CustomHuggingFacePipeline(pipeline=pipe)
print("AutoGPTQ 기반 LLM 초기화 완료.")

def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

def generate_step_back_text(obj, query=None):
    step_back_question = (
        "이 서비스의 전반적인 정보는 무엇인가요?"
        if query is None
        else f"'{query}'와 관련된 정보를 종합하여 요약하세요."
    )
    faq_text = (
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
    )
    summary_text = (
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
    step_back_text = (
        f"### Step-Back Question\n"
        f"{step_back_question}\n\n"
        f"### FAQ 스타일 정보\n"
        f"{faq_text}\n\n"
        f"### 종합 요약\n"
        f"{summary_text}"
    )
    return step_back_text

json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\20250304.json"
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"JSON 파일 '{json_file_path}'을 찾을 수 없습니다.")
documents = [Document(page_content=generate_step_back_text(obj), metadata={k: clean_metadata(obj.get(k)) for k in obj}) for obj in json_data]
print(f"총 {len(documents)}개의 문서가 로드됨.")

embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko", model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ"
try:
    vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
    print("기존 FAISS 벡터스토어 로드 성공.")
except:
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(persist_directory)
    print(f"총 {len(documents)}개의 문서가 FAISS 벡터 DB에 저장됨.")

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L2-v2', device=device)
def cross_encoder_rerank(query, documents, top_n=10, batch_size=32):
    dataset = RerankingDataset(query, documents)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    all_docs = []
    for batch_query, batch_docs in dataloader:
        pairs = [[query, doc] for doc in batch_docs]
        scores = cross_encoder.predict(pairs, batch_size=batch_size)
        all_scores.extend(scores)
        all_docs.extend(documents[len(all_scores)-len(scores):len(all_scores)])
    sorted_pairs = sorted(zip(all_scores, all_docs), reverse=True)
    return [doc for _, doc in sorted_pairs][:top_n]

class CrossEncoderCompressor(BaseDocumentCompressor):
    cross_encoder: CrossEncoder = Field(...)
    top_n: int = Field(...)
    class Config:
        arbitrary_types_allowed = True
    def __init__(self, cross_encoder: CrossEncoder, top_n: int):
        super().__init__(cross_encoder=cross_encoder, top_n=top_n)
    def compress_documents(self, documents, query, callbacks=None):
        return cross_encoder_rerank(query, documents, self.top_n)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

cross_encoder_compressor = CrossEncoderCompressor(cross_encoder=cross_encoder, top_n=20)
retriever_cross_compressed = ContextualCompressionRetriever(base_compressor=cross_encoder_compressor, base_retriever=base_retriever)

cohere_compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=15)
retriever_cohere_compressed = ContextualCompressionRetriever(base_compressor=cohere_compressor, base_retriever=retriever_cross_compressed)

final_compressor = CrossEncoderCompressor(cross_encoder=cross_encoder, top_n=10)
retriever_final = ContextualCompressionRetriever(base_compressor=final_compressor, base_retriever=retriever_cohere_compressed)

system_prompt = "검색된 상위 10개 문서의 정보에서, 오직 서비스명과 서비스ID만을 번호별로 출력하세요.\n출력 형식: [번호]. 서비스명 (서비스ID)"
rag_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{context}")])
combine_docs_chain = create_stuff_documents_chain(llm_gptq, rag_prompt)
chain = create_retrieval_chain(retriever_final, combine_docs_chain)
print("Cross-Encoder (20) + CohereRerank (15) + Cross-Encoder (10) 체인 생성 완료.")

if __name__ == "__main__":
    queries = [
        "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘.",
        "전라남도 목포시에 사는 48세 중년 남자에게 맞는 귀어 관련 창업에 대해 알려줘.",
        "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n=== 쿼리 {i}: {query} ===")
        try:
            retrieved_docs = base_retriever.invoke(query)
            print(f"기본 검색된 문서 수: {len(retrieved_docs)}")
            cross_docs = retriever_cross_compressed.invoke(query)
            print(f"1차 Cross-Encoder 후 문서 수: {len(cross_docs)}")
            cohere_docs = retriever_cohere_compressed.invoke(query)
            print(f"Cohere Rerank 후 문서 수: {len(cohere_docs)}")
            final_docs = retriever_final.invoke(query)
            print("2차 Cross-Encoder 후 상위 10개 문서:")
            for j, doc in enumerate(final_docs, 1):
                print(f"문서 {j}: {doc.metadata['서비스명']} (서비스ID: {doc.metadata['서비스ID']})")
            response_chain = chain.invoke({"input": query})
            print("체인 응답:")
            print(response_chain["answer"])
        except Exception as e:
            print(f"쿼리 실행 중 오류 발생: {e}")
            traceback.print_exc()