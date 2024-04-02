# -*- coding: utf-8 -*-
"""RAG_original.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bvsacjkvZ_ukOWGCrm2Vq2sJVxrW684v
"""

# !pip install ctransformers transformers langchain langchain-community torch pypdf sentence-transformers gpt4all faiss-cpu

# !pip install -q bitsandbytes accelerate loralib datasets loralib
# !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

# !pip install -qq -U accelerate=='0.25.0' peft=='0.7.1' bitsandbytes=='0.41.3.post2' trl=='0.7.4' transformers=='4.36.1'

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install 'openai-1.14.1.tar.gz'

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS, VectorStore
from langchain_community.embeddings import GPT4AllEmbeddings

# #Khai bao bien
# file_path = "/content/drive/MyDrive/RAG/data/all_context.txt"

# vector_db_path = "/content/drive/MyDrive/RAG/vectorstores/db_faiss"

# pdf_data_path = "/content/drive/MyDrive/RAG/data/"

# # Mở file txt để đọc
# with open(file_path, 'r', encoding='utf-8') as f:
#     # Đọc nội dung từ file và lưu vào biến content
#     content = f.read()

# model_path =  "/content/drive/MyDrive/RAG/models/all-MiniLM-L6-v2-f16.gguf"

# # Đọc file model
# with open(model_path, 'rb') as file:
#     model_data = file.read()


# def create_db_from_text():
#     raw_text = content


#     text_splitter = CharacterTextSplitter(
#         separator='\n',
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len
#     )

#     chunks = text_splitter.split_text(raw_text)

#     #Embedding
#     embedding_model = GPT4AllEmbeddings(model_file =  model_data)

#     #Dua vao Faiss Vector DB
#     db = FAISS.from_texts(texts = chunks, embedding=embedding_model)
#     db.save_local(vector_db_path)
#     return db

# def create_db_from_file():
#     loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#     chunks = text_splitter.split_documents(documents)

#     #Embedding
#     embedding_model = GPT4AllEmbeddings(model_file =  model_data)

#     #Dua vao Faiss Vector DB
#     db = FAISS.from_documents(documents = chunks, embedding=embedding_model)
#     db.save_local(vector_db_path)
#     return db

# create_db_from_text()

from langchain_community.llms import CTransformers
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.faiss import FAISS, VectorStore

model_path = "models/ggml-vistral-7B-chat-q4_0-001.gguf"


# cau hinh
model_file = model_path
vector_db_path = "vectorstores/db_faiss"
#load LLM
def load_llm(model_file):
    llm = CTransformers(
        model = model_file ,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.05
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

template= '''<s>[INST] <<SYS>>Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.<</SYS>>
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
print(response['result'])

import pandas as pd

# Đọc file CSV và chuyển thành DataFrame
df = pd.read_csv('Q&A.csv', encoding = "utf-8")

# Lựa chọn ngẫu nhiên 100 hàng từ DataFrame
df_random_100 = df.sample(n=100)

# Hiển thị 100 hàng ngẫu nhiên
print(df_random_100)

ls_answer = []
count = 0

import pandas as pd
import time

for i in range(0,100):
  # Bắt đầu đo thời gian
  start_time = time.time()
  question = df_random_100['question'].iloc[i]
  response = llm_chain.invoke({"query": question})
  ls_answer.append(response['result'])
  # Kết thúc đo thời gian
  end_time = time.time()
  # Tính thời gian thực thi
  execution_time = end_time - start_time

  count+=1
  print(f'lần {count}: {execution_time} giây')

df_random_100['answer'] = ls_answer

# Lưu DataFrame thành tệp CSV với encoding là 'utf-8'
df_random_100.to_csv('my_data.csv', index=False, encoding='utf-8')

print("Đã lưu DataFrame thành công thành tệp CSV với encoding là 'utf-8'.")

# Lưu DataFrame thành tệp CSV với encoding là 'utf-8'
df_random_100.to_csv('data.csv', index=False, encoding='utf-8')

print("Đã lưu DataFrame thành công thành tệp CSV với encoding là 'utf-8'.")