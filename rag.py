import os
import shutil
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "doc/virus-book.pdf"
DB_PATH = "chroma_db"

def load_pdf_documents(pdf_path):
    """Загрузка и разбивка PDF на оптимальные чанки для Qwen-1.5B"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
    
    print(f"Processing PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pdf = loader.load()
    
  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      
        chunk_overlap=50,    
        separators=["\n\n", "\n", ".", "!", "?", ";", " "]
    )
    return splitter.split_documents(pdf)

def create_embedding():
    """Инициализация легковесной модели эмбеддингов"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

def initialize_vectorstore(pdf_path, persist_dir, force_recreate=False):
    """Создание или загрузка векторной базы данных"""
    embedding = create_embedding()
    
    if force_recreate and os.path.exists(persist_dir):
        print(f"Удаление старой базы данных в {persist_dir}...")
        shutil.rmtree(persist_dir)

    if os.path.exists(persist_dir) and not force_recreate:
        print(f"Загрузка существующего векторного хранилища из {persist_dir}")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    
    print(f"Создание нового векторного хранилища...")
    docs = load_pdf_documents(pdf_path)
    print(f"Загружено документов: {len(docs)}")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_dir
    )
    return vectorstore

def build_rag_chain(persist_dir="chroma_db", k=4):
    """Сборка RAG цепочки с улучшенным промптом"""
    
    llm = Ollama(
        model="qwen2:1.5b",
        temperature=0.1,     
        num_ctx=4096,        
        keep_alive="5m"     
    )
    
    embedding = create_embedding()
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    
    retriever = db.as_retriever(search_kwargs={"k": k})

    template = """You are a strict Data Analyst. Use ONLY the CONTEXT below to answer the QUESTION.

### CONTEXT:
{context}

### QUESTION:
{question}

### INSTRUCTIONS:
1. **Strict Source:** Answer using ONLY the information from the Context. Do not use outside knowledge.
2. **Analysis:** If comparing viruses, look for keywords like "damage", "year", or "type".
3. **Format:** Keep the answer concise and professional.
4. **No Data:** If the answer is not in the Context, reply exactly: "I don't have enough data in the provided context."

### ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    try:
        # Шаг 1: Инициализация базы (force_recreate=True при первом запуске, если изменили PDF)
        print("=" * 50)
        initialize_vectorstore(
            PDF_PATH,
            DB_PATH,
            force_recreate=False # Поставьте True, если заменили файл virus-book.pdf
        )
        
        # Шаг 2: Создание цепочки
        print("=" * 50)
        chain = build_rag_chain(DB_PATH)
        
        # Тестовые вопросы
        questions = [
            "What is the most dangerous virus mentioned?",
            "List the infection methods for email viruses.",
            "What happened in 1999?"
        ]
        
        print("\n" + "=" * 50)
        for question in questions:
            print(f"\n❓ Вопрос: {question}")
            print("-" * 50)
            
            # Streaming вывод для ощущения скорости (появляется по словам)
            full_response = ""
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")