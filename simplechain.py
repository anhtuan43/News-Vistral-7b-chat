from langchain_community.llms import CTransformers
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.faiss import FAISS, VectorStore


# cau hinh
model_file = "models/ggml-vistral-7B-chat-q4_0.gguf"
vector_db_path = "vectorstores/db_faiss/"
#load LLM
def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "mistral",
        max_new_tokens = 1024,
        temperature = 0.1
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context","question"])
    return prompt

#Tao simple chain
def create_simple_chain(prompt, llm, db):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

def create_chat_chain(prompt, llm, db):
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit = 1024),
        return_source_documents = False,
        chain_type_kwargs = {'prompt':prompt}
    )
    return llm_chain

def read_vector_db():
    #Embedding
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True )
    return db

template= '''<s>[INST] <<SYS>>Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
<</SYS>>
{context}
{question} [/INST]
'''
#Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
db = read_vector_db()
prompt = create_prompt(template)
llm = load_llm(model_file)
llm_chain = create_chat_chain(prompt, llm, db)

question = "Google đã thực hiện những biện pháp gì để trấn an người dùng về việc bảo mật dữ liệu cá nhân?"
#response = llm_chain.invoke({"question": question})
response = llm_chain.invoke({"query": question}) 
print(response)



