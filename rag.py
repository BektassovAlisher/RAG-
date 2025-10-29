from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
import os

PDF_PATH = "doc/virus-book.pdf"

# 1. Загрузка и разбиение PDF
def csv_loader(pdf_path="doc/virus-book.pdf"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pdf = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";", " "]
    )
    return splitter.split_documents(pdf)


# 2. Создание эмбеддингов
def create_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # используйте 'cuda' если есть GPU
    )

# 3. Векторное хранилище
def create_vectorstore(pdf_documents, persist_dir="chroma_db"):
    embedding = create_embedding()
    
    vectorstore = Chroma.from_documents(
        documents=pdf_documents,
        embedding=embedding,
        persist_directory=persist_dir
    )
    return vectorstore

# 4. Загрузка существующего векторного хранилища
def load_vectorstore(persist_dir="chroma_db"):
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Векторное хранилище не найдено: {persist_dir}")
    
    embedding = create_embedding()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )

# 5. Инициализация векторного хранилища
def initialize_vectorstore(csv_path="doc/virus-book.pdf", 
                          persist_dir="chroma_db", 
                          force_recreate=False):
    if os.path.exists(persist_dir) and not force_recreate:
        print(f"Загрузка существующего векторного хранилища из {persist_dir}")
        return load_vectorstore(persist_dir)
    else:
        print(f"Создание нового векторного хранилища...")
        csv_documents = csv_loader(csv_path)
        print(f"Загружено {len(csv_documents)} документов")
        return create_vectorstore(csv_documents, persist_dir)

# 6. RAG Chain
def build_rag_chain(persist_dir="chroma_db", k=3):
    # Проверяем наличие векторного хранилища
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Векторное хранилище не найдено в {persist_dir}. "
            "Запустите initialize_vectorstore() сначала."
        )
    
    # Инициализация компонентов
    llm = Ollama(model="qwen2:1.5b",temperature=0.1)  
    embedding = create_embedding()
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": k})

    # Промпт
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant that answers questions based on computer virus data.
Use only the provided context from the database to answer the question.

Context from :
{context}

Question:
{question}

Instructions:
- Answer concisely and accurately based only on the provided virus-related context
- If the question requires analysis (for example, "Which virus is the most dangerous?"),
  analyze the available information (such as damage level, infection method, or impact)
  and provide a reasoned answer based on the context.
- If the information is insufficient, say: "I don’t have enough data to answer this question."
- Do not include this phrase if the context already contains enough data to answer
- When possible, cite exact data such as the virus name, year of appearance, type, infection method, or affected systems
- Do not include any information that is not found in the context

""")

    # Собираем цепочку
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



# Использование
if __name__ == "__main__":
    try:
        # Шаг 1: Инициализация векторного хранилища (только один раз)
        print("=" * 50)
        print("Инициализация векторного хранилища...")
        print("=" * 50)
        initialize_vectorstore(
            PDF_PATH,
            force_recreate=False  
        )
        
        # Шаг 2: Создание RAG цепочки
        print("\n" + "=" * 50)
        print("Создание RAG цепочки...")
        print("=" * 50)
        chain = build_rag_chain()
        
        # Шаг 3: Задаем вопросы
        questions = [
            "What is the virus"
        ]
        
        print("\n" + "=" * 50)
        print("Ответы на вопросы:")
        print("=" * 50)
        
        for question in questions:
            print(f"\nВопрос: {question}")
            print("-" * 50)
            response = chain.invoke(question)
            print(f"Ответ: {response}\n")
            
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("\nПроверьте наличие следующих файлов:")
        print("1. PDF файл: doc/virus-book.csv")
        print("2. Если используете кэш: папка chroma_db/")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
