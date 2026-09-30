import os
import re
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import fitz  # pymupdf / mupdf
import chromadb

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ATENÇÃO: Insira sua chave da OpenAI aqui para o G-Eval funcionar como Juiz
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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

DATASET_QUERIES: list[str] = [
    # insira o dataset aqui, por exemplo:
    "O banco pode descontar a parcela do meu empréstimo direto da minha conta salário e me deixar sem dinheiro nenhum?",
]

FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "pergunta": "Tenho direito a troca se o produto quebrar em 2 dias?",
        "textos": "Art. 26 CDC: O prazo para reclamar de defeitos aparentes é de 30 dias para produtos não duráveis e 90 dias para duráveis.",
        "resposta": (
            "Sim! Se o produto quebrou em 2 dias, você tem direito a reclamar.\n"
            "Para produtos que duram muito tempo (como geladeira, celular), você tem até 90 dias "
            "para ir à loja e pedir troca ou conserto.\n"
            "Exemplo: comprou um celular e ele parou de funcionar na primeira semana — "
            "volte à loja, mostre a nota fiscal e exija a troca ou conserto."
        ),
    },
    # ... (demais exemplos omitidos para brevidade, mas você pode manter os seus originais aqui)
    {
        "pergunta": "A loja pode se recusar a me dar nota fiscal?",
        "textos": "Art. 40 CDC: O fornecedor é obrigado a entregar ao consumidor documento fiscal na venda de produto ou prestação de serviço.",
        "resposta": (
            "Não! A loja é obrigada por lei a te dar nota fiscal.\n"
            "Se ela recusar, isso é crime e você pode denunciar na Receita Federal ou no Procon.\n"
            "Guarde sempre a nota — ela é sua prova de compra se algo der errado."
        ),
    }
]

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

simplicity_metric = GEval(
    name="Simplicidade para Leigos (Sem RAG)",
    criteria="Avalie se a resposta é compreensível para uma pessoa com pouca escolaridade. A resposta deve ter frases curtas, ser direta e estar livre de jargões jurídicos complexos.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o"
)

relevance_metric = GEval(
    name="Relevância da Resposta (Sem RAG)",
    criteria="Avalie se a resposta aborda de forma útil, direta e completa a pergunta do usuário. A resposta não deve conter informações irrelevantes, não deve fugir do tema e deve fazer sentido lógico independentemente de onde a informação foi tirada.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o"
)

def avaliar_com_geval(pergunta: str, resposta: str, trechos: list[dict]) -> dict:
    """Executa as métricas do DeepEval usando o GPT-4o como juiz."""
    contextos = [t["texto"] for t in trechos]
    
    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        retrieval_context=contextos
    )
    
    faithfulness_metric.measure(test_case)
    simplicity_metric.measure(test_case)
    relevance_metric.measure(test_case)
    
    return {
        "Fidelidade": {
            "score": faithfulness_metric.score,
            "reason": faithfulness_metric.reason
        },
        "Simplicidade": {
            "score": simplicity_metric.score,
            "reason": simplicity_metric.reason
        },
        "Relevância": {
            "score": relevance_metric.score,
            "reason": relevance_metric.reason
        }
    }


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def garantir_pastas():
    for p in [PDF_FOLDER, CHROMA_PERSIST_DIR, RESULTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    log("Pastas verificadas/criadas.")

def extrair_texto_pdf(caminho_pdf: Path) -> str:
    doc = fitz.open(str(caminho_pdf))
    paginas = []
    for i, pagina in enumerate(doc):
        texto = pagina.get_text("text")
        texto = re.sub(r'\n{3,}', '\n\n', texto)
        texto = re.sub(r'[ \t]+', ' ', texto)
        paginas.append(f"[Página {i+1}]\n{texto.strip()}")
    doc.close()
    return "\n\n".join(paginas)

def chunkar_texto(texto: str, fonte: str) -> list[dict]:
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

def gerar_embedding(texto: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": texto},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def chamar_llm(prompt: str) -> str:
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

def obter_colecao() -> chromadb.Collection:
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
    
    ids, embeddings, documentos, metadados = [], [], [], []
    for i, chunk in enumerate(chunks):
        if len(chunk["texto"]) < 30:
            continue
        chunk_id = f"{nome}__chunk_{chunk['chunk_id']}"
        emb = gerar_embedding(chunk["texto"])
        ids.append(chunk_id)
        embeddings.append(emb)
        documentos.append(chunk["texto"])
        metadados.append({"fonte": nome, "chunk_id": chunk["chunk_id"]})

    LOTE = 100
    for inicio in range(0, len(ids), LOTE):
        colecao.add(
            ids=ids[inicio:inicio+LOTE],
            embeddings=embeddings[inicio:inicio+LOTE],
            documents=documentos[inicio:inicio+LOTE],
            metadatas=metadados[inicio:inicio+LOTE],
        )
    log(f"  ✓ '{nome}' indexado.")

def indexar_todos_pdfs():
    colecao = obter_colecao()
    pdfs = list(PDF_FOLDER.glob("*.pdf")) + list(PDF_FOLDER.glob("*.PDF"))
    if not pdfs:
        log(f"⚠️  Nenhum PDF encontrado em '{PDF_FOLDER}'. Adicione PDFs e reexecute.")
        return colecao
    for pdf in pdfs:
        indexar_pdf(colecao, pdf)
    log(f"✅ ChromaDB contém {colecao.count()} chunks no total.")
    return colecao

def _formatar_few_shot() -> str:
    blocos = []
    for ex in FEW_SHOT_EXAMPLES:
        blocos.append(
            f"PERGUNTA DE EXEMPLO:\n{ex['pergunta']}\n\n"
            f"RESPOSTA DE EXEMPLO:\n{ex['resposta']}"
        )
    return "\n\n---\n\n".join(blocos)

PROMPT_TEMPLATE = """Você é um assistente que explica assuntos jurídicos para pessoas com pouca escolaridade.
Responda de forma simples, como se estivesse explicando para alguém leigo.
 
Regras obrigatórias:
- Use frases curtas e simples.
- Evite palavras difíceis e termos jurídicos.
- Use exemplos simples quando possível.
- Vá direto ao ponto.
- Utilize primordialmente as informações dos documentos recuperados.
- Não invente leis ou regras.
 
Formato da resposta:
- No máximo 5 a 7 linhas.
- Linguagem clara, como uma conversa.
- Sem palavras técnicas. Se usar um termo difícil, explique com palavras simples.
 
Veja abaixo alguns exemplos de como você deve responder:
 
{exemplos}
 
---
 
Agora use os documentos abaixo para responder a nova pergunta:
 
{contexto}
 
PERGUNTA:
{pergunta}
 
RESPOSTA:"""

def recuperar_trechos(colecao: chromadb.Collection, query: str) -> list[dict]:
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

def responder(colecao: chromadb.Collection, pergunta: str) -> dict:
    trechos = recuperar_trechos(colecao, pergunta)
    contexto = "\n\n---\n\n".join(
        f"[Fonte: {t['fonte']} | chunk {t['chunk_id']} | score {t['distancia']}]\n{t['texto']}"
        for t in trechos
    )
    prompt = PROMPT_TEMPLATE.format(
        exemplos=_formatar_few_shot(),
        contexto=contexto,
        pergunta=pergunta,
    )
    resposta = chamar_llm(prompt)
    return {"pergunta": pergunta, "resposta": resposta, "trechos": trechos}

def salvar_consolidados(acumulador: list):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    path1 = RESULTS_DIR / f"1_perguntas_e_respostas_{ts}.txt"
    with open(path1, "w", encoding="utf-8") as f:
        for idx, r in enumerate(acumulador, 1):
            f.write(f"[{idx}] PERGUNTA: {r['pergunta']}\n")
            f.write(f"RESPOSTA:\n{r['resposta']}\n")
            f.write("-" * 80 + "\n\n")

    path2 = RESULTS_DIR / f"2_documentos_recuperados_{ts}.txt"
    with open(path2, "w", encoding="utf-8") as f:
        for idx, r in enumerate(acumulador, 1):
            f.write(f"[{idx}] PERGUNTA: {r['pergunta']}\n")
            for i, t in enumerate(r["trechos"], 1):
                f.write(f"  Trecho {i} | Fonte: {t['fonte']} | Chunk: {t['chunk_id']} | Distância: {t['distancia']}\n")
                f.write(f"  {t['texto']}\n\n")
            f.write("=" * 80 + "\n\n")

    path3 = RESULTS_DIR / f"3_metricas_geval_{ts}.txt"
    with open(path3, "w", encoding="utf-8") as f:
        for idx, r in enumerate(acumulador, 1):
            f.write(f"[{idx}] PERGUNTA: {r['pergunta']}\n")
            if "metricas" in r:
                for nome, dados in r["metricas"].items():
                    f.write(f"  Métrica: {nome}\n")
                    f.write(f"  Score: {dados['score']}\n")
                    f.write(f"  Razão: {dados['reason']}\n\n")
            else:
                f.write("  (Métricas não avaliadas)\n")
            f.write("-" * 80 + "\n\n")

    path4 = RESULTS_DIR / f"4_tabela_scores_excel_{ts}.txt"
    with open(path4, "w", encoding="utf-8") as f:
        f.write("Índice\tPergunta\tFidelidade\tSimplicidade\tRelevância\n")
        for idx, r in enumerate(acumulador, 1):
            pergunta_limpa = r['pergunta'].replace('\n', ' ').replace('\t', ' ')
            if "metricas" in r:
                score_fid = r["metricas"]["Fidelidade"]["score"]
                score_sim = r["metricas"]["Simplicidade"]["score"]
                score_rel = r["metricas"]["Relevância"]["score"]
            else:
                score_fid = score_sim = score_rel = "N/A"
            f.write(f"{idx}\t{pergunta_limpa}\t{score_fid}\t{score_sim}\t{score_rel}\n")

    log("📄 Consolidados salvos:")
    log(f"   {path1.name}")
    log(f"   {path2.name}")
    log(f"   {path3.name}")
    log(f"   {path4.name}")

def rodar_dataset(colecao: chromadb.Collection):
    log(f"\n🔎 Rodando dataset com {len(DATASET_QUERIES)} queries...\n")
    acumulador = []

    for i, query in enumerate(DATASET_QUERIES, 1):
        log(f"[{i}/{len(DATASET_QUERIES)}] {query[:80]}...")
        resultado = responder(colecao, query)
        
        log(f"  Avaliando métricas com GPT-4o...")
        resultado["metricas"] = avaliar_com_geval(resultado["pergunta"], resultado["resposta"], resultado["trechos"])
        
        acumulador.append(resultado)
        print(f"\n{'─'*60}\nP: {resultado['pergunta']}\nR: {resultado['resposta'][:300]}...\n{'─'*60}\n")

    salvar_consolidados(acumulador)
    log("✅ Dataset completo. Resultados em: " + str(RESULTS_DIR))

def modo_interativo(colecao: chromadb.Collection):
    log("\n💬 Modo interativo. Digite 'sair' para encerrar.\n")
    acumulador = []

    while True:
        pergunta = input("Pergunta: ").strip()
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        if not pergunta:
            continue
            
        log("Gerando Resposta (Ollama)...")
        resultado = responder(colecao, pergunta)
        print(f"\nResposta:\n{resultado['resposta']}\n")
        
        log("Avaliando métricas (GPT-4o)...")
        resultado["metricas"] = avaliar_com_geval(resultado["pergunta"], resultado["resposta"], resultado["trechos"])
        
        acumulador.append(resultado)

    if acumulador:
        salvar_consolidados(acumulador)
        log(f" {len(acumulador)} pergunta(s) salvas nos consolidados.")

def main():
    print("\n" + "═"*60)
    print("  Sistema RAG Jurídico Brasileiro c/ Avaliação G-Eval (GPT-4o)")
    print("  LLM: Llama 3.1 8B | DB: ChromaDB | PDFs: MuPDF")
    print("═"*60 + "\n")
 
    if not os.getenv("OPENAI_API_KEY"):
        print("  AVISO: OPENAI_API_KEY não foi encontrada nas variáveis de ambiente.")
        print("   O processo de avaliação (G-Eval) com GPT-4o irá falhar se ela não estiver configurada.\n")

    garantir_pastas()
    verificar_ollama()
    colecao = indexar_todos_pdfs()
 
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