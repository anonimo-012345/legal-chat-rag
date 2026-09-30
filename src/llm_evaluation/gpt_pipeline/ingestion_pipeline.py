import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()


# Assist class for loading PDF files using PyMuPDF
class MuPDFLoader:
    # Initialize with the path to the PDF file

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        documents = []
        pdf = fitz.open(self.file_path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")  # page text as string

            if text.strip():  # only add if there's text content
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "page": page_num + 1,
                        "total_pages": len(pdf),
                    }
                ))

        pdf.close()
        return documents
    
# 1: Load the documents
def load_documents(docs_path=Path(__file__).resolve().parents[2] / "docs"):
    # Load .txt and .pdf files from the specified directory
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"O diretório '{docs_path}' não existe. Crie-o e adicione seus arquivos."
        )

    documents = []

    # Load .txt
    txt_files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]
    for filename in txt_files:
        file_path = os.path.join(docs_path, filename)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
        print(f"  [TXT] {filename}")

    # Load .pdf with MuPDF
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    for filename in pdf_files:
        file_path = os.path.join(docs_path, filename)
        loader = MuPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
        print(f"  [PDF] {filename} — {len(docs)} página(s) carregada(s)")

    if not documents:
        raise FileNotFoundError(
            f"Nenhum arquivo .txt ou .pdf encontrado em '{docs_path}'."
        )

    print(f"\nTotal: {len(documents)} documento(s) carregado(s).")

    return documents

# 2: Split the documents into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:20]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 20:
            print(f"\n... and {len(chunks) - 20} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"} # cosine can be replaced for another distance metric if needed
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print ("main function called")

    # 1 Loading the files
    documents = load_documents(docs_path= "docs")  

    # 2 Chinking the files
    chunks = split_documents(documents)

    # 3 Creating the embeddings and Store
    print(f"Diretório atual: {os.getcwd()}")
    print(f"Arquivos na pasta: {os.listdir('.')}")
    vectorstore = create_vector_store(chunks, persist_directory="db/chroma_db")


if __name__ == "__main__":
    main()