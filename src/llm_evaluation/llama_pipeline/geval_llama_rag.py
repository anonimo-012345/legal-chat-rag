import re
import json
import datetime

from dotenv import load_dotenv
import fitz  # pymupdf / mupdf
import chromadb
import requests
import os
from pathlib import Path
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


OLLAMA_BASE_URL    = "http://localhost:11434"
LLM_MODEL          = "llama3.1:8b"
EMBED_MODEL        = "nomic-embed-text"   # rode: ollama pull nomic-embed-text

PDF_FOLDER         = Path(__file__).resolve().parents[2] / "docs"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
RESULTS_DIR        = Path(__file__).parent / "resultados"

CHUNK_SIZE         = 800   # caracteres por chunk
CHUNK_OVERLAP      = 150   # sobreposição entre chunks
N_RESULTS          = 5     # trechos recuperados por query
COLLECTION_NAME    = "juridico_br"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ATENÇÃO: Insira sua chave da OpenAI aqui para o G-Eval funcionar como Juiz
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
AVALIAR_COM_GPT = True # Mude para False se quiser rodar sem avaliar


DATASET_QUERIES: list[str] = [
   # insira o dataset aqui, por exemplo:
    "O banco pode descontar a parcela do meu empréstimo direto da minha conta salário e me deixar sem dinheiro nenhum?",
]


# Métrica 1: DEPENDE DO RAG (Avalia alucinação em relação aos documentos)
faithfulness_metric = GEval(
    name="Fidelidade ao Contexto (Com RAG)",
    criteria="Determine se a resposta atual é baseada estritamente no contexto recuperado. A resposta não pode conter leis, regras ou informações que não estejam presentes nos textos recuperados.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT, 
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    model="gpt-4o" 
)

# Métrica 2: INDEPENDENTE DO RAG (Avalia a persona/linguagem)
simplicity_metric = GEval(
    name="Simplicidade para Leigos (Sem RAG)",
    criteria="Avalie se a resposta é compreensível para uma pessoa com pouca escolaridade. A resposta deve ter frases curtas, ser direta e estar livre de jargões jurídicos complexos.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o"
)

# Métrica 3: INDEPENDENTE DO RAG (Avalia se respondeu bem à pergunta)
relevance_metric = GEval(
    name="Relevância da Resposta (Sem RAG)",
    criteria="Avalie se a resposta aborda de forma útil, direta e completa a pergunta do usuário. A resposta não deve conter informações irrelevantes, não deve fugir do tema e deve fazer sentido lógico independentemente de onde a informação foi tirada.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o"
)

def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def garantir_pastas():
    for p in [PDF_FOLDER, CHROMA_PERSIST_DIR, RESULTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    log(f"Pastas verificadas/criadas.")

def extrair_texto_pdf(caminho_pdf: Path) -> str:
    """Extrai todo o texto de um PDF usando MuPDF (pymupdf)."""
    doc = fitz.open(str(caminho_pdf))
    paginas = []
    for i, pagina in enumerate(doc):
        texto = pagina.get_text("text")
        texto = re.sub(r'\n{3,}', '\n\n', texto)   # remove linhas em branco excessivas
        texto = re.sub(r'[ \t]+', ' ', texto)       # normaliza espaços
        paginas.append(f"[Página {i+1}]\n{texto.strip()}")
    doc.close()
    return "\n\n".join(paginas)

def chunkar_texto(texto: str, fonte: str) -> list[dict]:
    """Divide o texto em chunks com sobreposição."""
    chunks = []
    inicio = 0
    idx = 0
    while inicio < len(texto):
        fim = inicio + CHUNK_SIZE
        trecho = texto[inicio:fim]
        chunks.append({
            "texto": trecho.strip(),
            "fonte": fonte,
            "chunk_id": idx,
        })
        inicio += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


# OLLAMA — EMBEDDINGS E LLM

def gerar_embedding(texto: str) -> list[float]:
    """Gera embedding via Ollama (nomic-embed-text)."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": texto},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def chamar_llm(prompt: str) -> str:
    """Chama o Llama 3.1 8B via Ollama (modo não-streaming)."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

def verificar_ollama():
    """Verifica se o Ollama está rodando."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        modelos = [m["name"] for m in r.json().get("models", [])]
        log(f"Ollama OK. Modelos disponíveis: {modelos}")
        if not any(LLM_MODEL.split(":")[0] in m for m in modelos):
            log(f"  Modelo '{LLM_MODEL}' não encontrado. Execute: ollama pull {LLM_MODEL}")
        if not any(EMBED_MODEL.split(":")[0] in m for m in modelos):
            log(f"  Modelo '{EMBED_MODEL}' não encontrado. Execute: ollama pull {EMBED_MODEL}")
    except Exception as e:
        raise RuntimeError(f"Ollama não está acessível em {OLLAMA_BASE_URL}: {e}")


# CHROMADB — INDEXAÇÃO

def obter_colecao() -> chromadb.Collection:
    """Retorna (ou cria) a coleção ChromaDB persistente."""
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    colecao = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return colecao

def ja_indexado(colecao: chromadb.Collection, nome_arquivo: str) -> bool:
    resultado = colecao.get(where={"fonte": nome_arquivo}, limit=1)
    return len(resultado["ids"]) > 0

def indexar_pdf(colecao: chromadb.Collection, caminho_pdf: Path):
    nome = caminho_pdf.name
    if ja_indexado(colecao, nome):
        log(f"  → '{nome}' já indexado. Pulando.")
        return
    log(f"  → Extraindo texto de '{nome}'...")
    texto = extrair_texto_pdf(caminho_pdf)
    chunks = chunkar_texto(texto, nome)
    log(f"     {len(chunks)} chunks gerados.")
    ids, embeddings, documentos, metadados = [], [], [], []
    for i, chunk in enumerate(chunks):
        if len(chunk["texto"]) < 30:   # ignora chunks muito pequenos
            continue
        chunk_id = f"{nome}__chunk_{chunk['chunk_id']}"
        emb = gerar_embedding(chunk["texto"])
        ids.append(chunk_id)
        embeddings.append(emb)
        documentos.append(chunk["texto"])
        metadados.append({"fonte": nome, "chunk_id": chunk["chunk_id"]})
        if (i + 1) % 20 == 0:
            log(f"     Embeddings: {i+1}/{len(chunks)}")
    LOTE = 100
    for inicio in range(0, len(ids), LOTE):
        colecao.add(
            ids=ids[inicio:inicio+LOTE],
            embeddings=embeddings[inicio:inicio+LOTE],
            documents=documentos[inicio:inicio+LOTE],
            metadatas=metadados[inicio:inicio+LOTE],
        )
    log(f"  ✓ '{nome}' indexado com {len(ids)} chunks.")

def indexar_todos_pdfs():
    colecao = obter_colecao()
    pdfs = list(PDF_FOLDER.glob("*.pdf")) + list(PDF_FOLDER.glob("*.PDF"))
    if not pdfs:
        log(f" !!!  Nenhum PDF encontrado em '{PDF_FOLDER}'. Adicione PDFs e reexecute.")
        return colecao
    log(f"\n📚 Indexando {len(pdfs)} PDF(s)...")
    for pdf in pdfs:
        indexar_pdf(colecao, pdf)
    total = colecao.count()
    log(f"\n✅ ChromaDB contém {total} chunks no total.")
    return colecao


PROMPT_TEMPLATE = """Você é um assistente que explica assuntos jurídicos para pessoas com pouca escolaridade.
Responda de forma simples, como se estivesse explicando para alguém leigo.
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
Se precisar usar um termo difícil, explique com palavras simples.
Use os textos abaixo para responder a pergunta. 
{contexto}

PERGUNTA:
{pergunta}
"""

def recuperar_trechos(colecao: chromadb.Collection, query: str) -> list[dict]:
    """Recupera os N_RESULTS chunks mais relevantes para a query."""
    emb_query = gerar_embedding(query)
    resultado = colecao.query(
        query_embeddings=[emb_query],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    trechos = []
    for doc, meta, dist in zip(
        resultado["documents"][0],
        resultado["metadatas"][0],
        resultado["distances"][0],
    ):
        trechos.append({
            "texto": doc,
            "fonte": meta.get("fonte", "?"),
            "chunk_id": meta.get("chunk_id", "?"),
            "distancia": round(dist, 4),
        })
    return trechos

def avaliar_com_juiz(pergunta: str, resposta: str, trechos: list[dict]) -> dict:
    """Avalia a resposta gerada utilizando o GPT via DeepEval."""
    log("  Acionando GPT para avaliação (G-Eval)...")
    
    contexto_lista = [t["texto"] for t in trechos]
    
    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        retrieval_context=contexto_lista
    )
    
    faithfulness_metric.measure(test_case)
    simplicity_metric.measure(test_case)
    relevance_metric.measure(test_case)
    
    return {
        "Fidelidade": {"score": faithfulness_metric.score, "reason": faithfulness_metric.reason},
        "Simplicidade": {"score": simplicity_metric.score, "reason": simplicity_metric.reason},
        "Relevancia": {"score": relevance_metric.score, "reason": relevance_metric.reason}
    }


def responder(colecao: chromadb.Collection, pergunta: str) -> dict:
    """Pipeline RAG completo: recupera → gera → avalia → retorna."""
    trechos = recuperar_trechos(colecao, pergunta)
    contexto = "\n\n---\n\n".join(
        f"[Fonte: {t['fonte']} | chunk {t['chunk_id']} | score {t['distancia']}]\n{t['texto']}"
        for t in trechos
    )
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
    resposta = chamar_llm(prompt)
    
    resultado_final = {
        "pergunta": pergunta, 
        "resposta": resposta, 
        "trechos": trechos,
        "metricas": None
    }

    if AVALIAR_COM_GPT:
        try:
            metricas = avaliar_com_juiz(pergunta, resposta, trechos)
            resultado_final["metricas"] = metricas
        except Exception as e:
            log(f" !!! Erro ao avaliar com GPT: {e}")
            
    return resultado_final


def gerar_arquivos_finais(acumulador: list):
    if not acumulador:
        return
        
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    path1 = RESULTS_DIR / f"1_perguntas_respostas_{ts}.txt"
    path2 = RESULTS_DIR / f"2_avaliacoes_geval_{ts}.txt"
    path3 = RESULTS_DIR / f"3_contextos_recuperados_{ts}.txt"
    path4 = RESULTS_DIR / f"4_scores_excel_{ts}.txt"

    # ── 1. Arquivo de Perguntas e Respostas
    with open(path1, "w", encoding="utf-8") as f1:
        for i, r in enumerate(acumulador, 1):
            f1.write(f"[{i}] PERGUNTA: {r['pergunta']}\n")
            f1.write(f"RESPOSTA:\n{r['resposta']}\n")
            f1.write("-" * 80 + "\n\n")

    # ── 2. Arquivo de Avaliação G-Eval
    with open(path2, "w", encoding="utf-8") as f2:
        for i, r in enumerate(acumulador, 1):
            f2.write(f"[{i}] PERGUNTA: {r['pergunta']}\n")
            if r.get("metricas"):
                for nome_metrica, dados in r["metricas"].items():
                    f2.write(f"► {nome_metrica.upper()}\n")
                    f2.write(f"  Score : {dados['score']}\n")
                    f2.write(f"  Motivo: {dados['reason']}\n\n")
            else:
                f2.write("⚠️ Nenhuma métrica gerada ou ocorreu erro na avaliação.\n\n")
            f2.write("-" * 80 + "\n\n")

    # ── 3. Arquivo de Contextos (RAG)
    with open(path3, "w", encoding="utf-8") as f3:
        for idx, r in enumerate(acumulador, 1):
            f3.write(f"[{idx}] PERGUNTA: {r['pergunta']}\n")
            for i, t in enumerate(r["trechos"], 1):
                f3.write(f"─── Trecho {i} | Fonte: {t['fonte']} | Score: {t['distancia']} ───\n")
                f3.write(f"{t['texto']}\n\n")
            f3.write("-" * 80 + "\n\n")

    # ── 4. Arquivo com Escores Prontos para o Excel (Tab Separated)
    with open(path4, "w", encoding="utf-8") as f4:
        # Cabeçalho separado por tabulação (\t)
        f4.write("ID\tFidelidade\tSimplicidade\tRelevância\n")
        
        for i, r in enumerate(acumulador, 1):
            if r.get("metricas"):
                fid = r["metricas"]["Fidelidade"]["score"]
                sim = r["metricas"]["Simplicidade"]["score"]
                rel = r["metricas"]["Relevancia"]["score"]
                f4.write(f"{i}\t{fid}\t{sim}\t{rel}\n")
            else:
                f4.write(f"{i}\tN/A\tN/A\tN/A\n")

    log("\n ARQUIVOS FINAIS GERADOS COM SUCESSO:")
    log(f"   ✓ {path1.name}")
    log(f"   ✓ {path2.name}")
    log(f"   ✓ {path3.name}")
    log(f"   ✓ {path4.name}")


def rodar_dataset(colecao: chromadb.Collection):
    log(f"\n Rodando dataset com {len(DATASET_QUERIES)} queries...\n")
    acumulador = [] 
    for i, query in enumerate(DATASET_QUERIES, 1):
        log(f"[{i}/{len(DATASET_QUERIES)}] {query[:80]}...")
        resultado = responder(colecao, query)
        acumulador.append(resultado)
        
        print(f"\n{'─'*60}")
        print(f"P: {resultado['pergunta']}")
        print(f"R: {resultado['resposta'][:300]}...")
        if resultado.get('metricas'):
            print("\nMÉTRICAS:")
            for k, v in resultado['metricas'].items():
                print(f"- {k}: {v['score']}")
        print(f"{'─'*60}\n")

    gerar_arquivos_finais(acumulador)
    log("✅ Dataset completo. Resultados na pasta de resultados.")


def modo_interativo(colecao: chromadb.Collection):
    log("\n💬 Modo interativo. Digite 'sair' para encerrar.\n")
    acumulador = []
    idx = 1
    while True:
        pergunta = input("Pergunta: ").strip()
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        if not pergunta:
            continue
        log("Processando...")
        resultado = responder(colecao, pergunta)
        acumulador.append(resultado)
        
        print(f"\nResposta:\n{resultado['resposta']}\n")
        if resultado.get("metricas"):
            print("AVALIAÇÃO DO GPT:")
            for k, v in resultado["metricas"].items():
                print(f"[{k}] Score: {v['score']} -> {v['reason']}")
        idx += 1
    gerar_arquivos_finais(acumulador)

def main():
    print("\n" + "═"*60)
    print("  Sistema RAG Jurídico Brasileiro c/ Avaliador GPT")
    print("  LLM: Llama 3.1 8B | DB: ChromaDB | Juiz: OpenAI GPT")
    print("═"*60 + "\n")

    garantir_pastas()
    verificar_ollama()

    # 1) Indexa PDFs (pula os já indexados)
    colecao = indexar_todos_pdfs()

    # 2) Menu de execução
    print("\nEscolha o modo de execução:")
    print("  [1] Rodar dataset de teste (DATASET_QUERIES)")
    print("  [2] Modo interativo (perguntas avulsas)")
    print("  [3] Ambos (dataset + interativo)")
    opcao = input("\nOpção [1/2/3]: ").strip()

    if opcao == "1":
        rodar_dataset(colecao)
    elif opcao == "2":
        modo_interativo(colecao)
    elif opcao == "3":
        rodar_dataset(colecao)
        modo_interativo(colecao)
    else:
        log("Opção inválida. Rodando dataset por padrão.")
        rodar_dataset(colecao)

if __name__ == "__main__":
    main()