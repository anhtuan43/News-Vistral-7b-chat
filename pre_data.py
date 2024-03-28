from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS, VectorStore
from langchain_community.embeddings import GPT4AllEmbeddings

#Khai bao bien
pdf_data_path = "data"

vector_db_path = "vectorstores/db_faiss"

# Đường dẫn đến file txt
file_path = 'data/all_context.txt'

# Mở file txt để đọc
with open(file_path, 'r', encoding='utf-8') as f:
    # Đọc nội dung từ file và lưu vào biến content
    content = f.read()


def create_db_from_text():
    raw_text = content


    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    #Embedding
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")

    #Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts = chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_file():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    #Embedding
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")

    #Dua vao Faiss Vector DB
    db = FAISS.from_documents(documents = chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_text()