import ollama
from langchain_docling.loader import DoclingLoader

model = "llama3:8b"

FILE_PATH = r"C:/Users/kotar/Documents/Downloads/Environment_Essay_No_Tables_Images.pdf"
loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=60)
texts = text_splitter.split_documents(docs)
print("#"*50)
print(texts[0])


from langchain_huggingface import HuggingFaceEmbeddings
embeddings =HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

from langchain_chroma import Chroma

vector_store=Chroma(
    collection_name="sample_RAG",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
print(vector_store._collection.count())

from langchain_community.vectorstores.utils import filter_complex_metadata
texts = filter_complex_metadata(texts)
vector_store.add_documents(texts)

vector_store._collection.count()
print("#"*30)

def ask_llm(prompt):
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True
    )

    for res in response:
        print(res["message"]["content"], end="", flush=True)



while True:
    retriver = vector_store.as_retriever()
    query = input("Enter the query:")
    context = retriver.invoke(query)
    finalprompt = f"""you are a rag assistant and you need to answer questions on provided contexts only.if answer is not available in provided context then gently respond " i don't know".

        The input is {query}
        The retrived context are {context}
    """
    ask_llm(finalprompt)
