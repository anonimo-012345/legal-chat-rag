import os
from typing import List, Tuple
from anyio import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"
DOCS_PATH = Path(__file__).resolve().parent.parent / "docs"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o")

class MuPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        documents = []
        pdf = fitz.open(self.file_path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")
            if text.strip():
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


def load_documents(docs_path: str = DOCS_PATH) -> List[Document]:
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"O diretório '{docs_path}' não existe.")

    documents: List[Document] = []

    txt_files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]
    for filename in txt_files:
        file_path = os.path.join(docs_path, filename)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    for filename in pdf_files:
        file_path = os.path.join(docs_path, filename)
        documents.extend(MuPDFLoader(file_path).load())

    if not documents:
        raise FileNotFoundError(f"Nenhum arquivo .txt ou .pdf encontrado em '{docs_path}'.")

    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 0) -> List[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def create_vector_store(chunks: List[Document], persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


def run_ingestion(docs_path: str = DOCS_PATH, persist_directory: str = PERSIST_DIRECTORY) -> int:
    """Executa o pipeline completo de ingestão. Retorna o nº de chunks criados."""
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    create_vector_store(chunks, persist_directory)
    return len(chunks)


def get_vectorstore(persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _to_langchain_messages(history: List[Tuple[str, str]]):
    messages = []
    for role, content in history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def rewrite_question(user_question: str, chat_history: List[Tuple[str, str]]) -> str:
    """Reescreve a pergunta como consulta independente, usando o histórico."""
    if not chat_history:
        return user_question

    messages = [
        SystemMessage(content="""
Reescreva a pergunta do usuário como uma consulta jurídica
independente para recuperação vetorial.

Preserve:
- contexto jurídico
- intenção do usuário
- termos legais relevantes

Retorne apenas a consulta reescrita.
""")
    ] + _to_langchain_messages(chat_history) + [
        HumanMessage(content=f"Nova pergunta: {user_question}")
    ]

    result = llm.invoke(messages)
    return result.content.strip()


def retrieve_documents(search_question: str, k: int = 3) -> List[Document]:
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(search_question)


def generate_answer(user_question: str, docs: List[Document], chat_history: List[Tuple[str, str]]) -> str:
    combined_input = f"""Use os textos abaixo para responder a pergunta.

Textos:
{chr(10).join([doc.page_content for doc in docs])}

Pergunta:
{user_question}

Responda de forma simples, como se estivesse explicando para alguém leigo.
Se não tiver informação suficiente, diga isso.
    """

    messages = [
        SystemMessage(content="""
Você é um assistente que explica assuntos jurídicos para pessoas com pouca escolaridade.

Regras obrigatórias:
- Use frases curtas e simples.
- Evite palavras difíceis e termos jurídicos.
- Explique como se estivesse falando com alguém que não entende de leis.
- Use exemplos simples quando possível.
- Vá direto ao ponto.

Limites:
- Use apenas as informações dos documentos.
- Se não tiver certeza, diga que não sabe.
- Não invente leis ou regras.

Formato da resposta:
- No máximo 5 ou 7 linhas.
- Linguagem clara, como uma conversa.
- Sem palavras técnicas.

Se precisar usar um termo difícil, explique com palavras simples.""")
    ] + _to_langchain_messages(chat_history) + [
        HumanMessage(content=combined_input)
    ]

    result = llm.invoke(messages)
    return result.content


def ask_question(user_question: str, chat_history: List[Tuple[str, str]]) -> dict:
    """
    Função principal chamada pela API.
    chat_history: lista de tuplas (role, content) já carregadas do banco,
                  em ordem cronológica, ANTES desta nova pergunta.
    Retorna um dict com a resposta e os documentos usados (para debug/exibição).
    """
    search_question = rewrite_question(user_question, chat_history)
    docs = retrieve_documents(search_question)
    answer = generate_answer(user_question, docs, chat_history)

    return {
        "answer": answer,
        "search_question": search_question,
        "sources": [
            {
                "source": doc.metadata.get("source", "desconhecido"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:200],
            }
            for doc in docs
        ],
    }
