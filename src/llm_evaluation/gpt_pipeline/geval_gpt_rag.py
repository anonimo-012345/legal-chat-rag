from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(search_kwargs={"k": 5})

model = ChatOpenAI(model="gpt-4o")


# Métrica 1: DEPENDE DO RAG (Avalia alucinação em relação aos documentos)
faithfulness_metric = GEval(
    name="Fidelidade ao Contexto (Com RAG)",
    criteria="Determine se a resposta atual é baseada estritamente no contexto recuperado. A resposta não pode conter leis, regras ou informações que não estejam presentes nos textos recuperados.",
    evaluation_params=[
        SingleTurnParams.INPUT, 
        SingleTurnParams.ACTUAL_OUTPUT, 
        SingleTurnParams.RETRIEVAL_CONTEXT
    ],
)

# Métrica 2: INDEPENDENTE DO RAG (Avalia a persona/linguagem)
simplicity_metric = GEval(
    name="Simplicidade para Leigos (Sem RAG)",
    criteria="Avalie se a resposta é compreensível para uma pessoa com pouca escolaridade. A resposta deve ter frases curtas, ser direta e estar livre de jargões jurídicos complexos.",
    evaluation_params=[
        SingleTurnParams.INPUT, 
        SingleTurnParams.ACTUAL_OUTPUT
    ],
)

# Métrica 3: INDEPENDENTE DO RAG (Avalia se respondeu bem à pergunta)
relevance_metric = GEval(
    name="Relevância da Resposta (Sem RAG)",
    criteria="Avalie se a resposta aborda de forma útil, direta e completa a pergunta do usuário. A resposta não deve conter informações irrelevantes, não deve fugir do tema e deve fazer sentido lógico independentemente de onde a informação foi tirada.",
    evaluation_params=[
        SingleTurnParams.INPUT, 
        SingleTurnParams.ACTUAL_OUTPUT
    ],
)

questions = [
    # insira o dataset aqui, por exemplo:
    "O banco pode descontar a parcela do meu empréstimo direto da minha conta salário e me deixar sem dinheiro nenhum?",
]


answers_lines = []
retrieved_lines = []
metrics_lines = []


excel_lines = ["Num_Pergunta\tFidelidade_RAG\tRelevancia_Resposta\tSimplicidade"]

for i, query in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] Processando: {query}")

    # 1. Recuperação (Retrieval)
    relevant_docs = retriever.invoke(query)
    retrieved_texts_list = [doc.page_content for doc in relevant_docs]

    retrieved_texts = "\n".join(
        [f"  [{j}] {text}" for j, text in enumerate(retrieved_texts_list, 1)]
    )

    # 2. Geração (Generation)
    combined_input = f"""Use os textos abaixo para responder a pergunta. 
Pergunta: {query}
Textos:
{chr(10).join([f"- {text}" for text in retrieved_texts_list])}
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
Se precisar usar um termo difícil, explique com palavras simples."""),
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    answer = result.content.strip()

    print(f"  -> Avaliando qualidade da resposta...")
    
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieved_texts_list
    )


    faithfulness_metric.measure(test_case)
    simplicity_metric.measure(test_case)
    relevance_metric.measure(test_case)


    excel_lines.append(f"{i}\t{faithfulness_metric.score}\t{relevance_metric.score}\t{simplicity_metric.score}")

    metric_report = (
        f"--- [ PERGUNTA {i} ] ---\n"
        f"Texto da Pergunta: {query}\n\n"
        f"Métrica DEPEDENTE DO RAG:\n"
        f"• Fidelidade ao Contexto (Score: {faithfulness_metric.score})\n"
        f"  Razão: {faithfulness_metric.reason}\n\n"
        f"Métricas INDEPENDENTES DO RAG (Para comparação):\n"
        f"• Relevância da Resposta (Score: {relevance_metric.score})\n"
        f"  Razão: {relevance_metric.reason}\n\n"
        f"• Simplicidade para Leigos (Score: {simplicity_metric.score})\n"
        f"  Razão: {simplicity_metric.reason}\n"
    )
    metrics_lines.append(metric_report)
    metrics_lines.append("=" * 80)

    answers_lines.append(f"\n--- [ PERGUNTA {i} ] ---\nPergunta: {query}\n\nResposta:\n{answer}\n")
    answers_lines.append("-" * 60)

    retrieved_lines.append(f"--- [ PERGUNTA {i} ] ---\nPergunta:\n{query}\n\nTextos recuperados:\n{retrieved_texts}\n")
    retrieved_lines.append("-" * 60)

with open("respostas.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(answers_lines))

with open("textos_recuperados.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(retrieved_lines))

with open("metricas_avaliacao.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(metrics_lines))

with open("planilha_excel.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(excel_lines))

print("\n✅ Concluído! Arquivos gerados:")
print("   • respostas.txt")
print("   • textos_recuperados.txt")
print("   • metricas_avaliacao.txt")
print("   • planilha_excel.txt")