import os
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(search_kwargs={"k": 5})
model = ChatOpenAI(model="gpt-4o")

# ── Few-shot examples ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
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
    {
        "pergunta": "A loja pode se recusar a me dar nota fiscal?",
        "textos": "Art. 40 CDC: O fornecedor é obrigado a entregar ao consumidor documento fiscal na venda de produto ou prestação de serviço.",
        "resposta": (
            "Não! A loja é obrigada por lei a te dar nota fiscal.\n"
            "Se ela recusar, isso é crime e você pode denunciar na Receita Federal ou no Procon.\n"
            "Guarde sempre a nota — ela é sua prova de compra se algo der errado."
        ),
    },
    {
        "pergunta": "Posso desistir de uma compra feita pela internet?",
        "textos": "Art. 49 CDC: O consumidor pode desistir do contrato em 7 dias a contar da assinatura ou do recebimento do produto, nas compras feitas fora do estabelecimento comercial.",
        "resposta": (
            "Sim! Quando você compra pela internet, tem 7 dias para desistir — sem precisar dar nenhuma explicação.\n"
            "Esse prazo começa a contar a partir do dia que o produto chega na sua casa.\n"
            "A loja é obrigada a devolver todo o dinheiro que você pagou, incluindo o frete."
        ),
    },
]

def build_few_shot_block() -> str:
    lines = ["## Exemplos de como responder:\n"]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"Pergunta: {ex['pergunta']}")
        lines.append(f"Textos da lei: {ex['textos']}")
        lines.append(f"Resposta: {ex['resposta']}")
        lines.append("---")
    return "\n".join(lines)

FEW_SHOT_BLOCK = build_few_shot_block()

faithfulness_metric = GEval(
    name="Fidelidade ao Contexto (Com RAG)",
    criteria="Determine se a resposta atual é baseada estritamente no contexto recuperado. A resposta não pode conter leis, regras ou informações que não estejam presentes nos textos recuperados.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT, 
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    model="gpt-4o",
)

simplicity_metric = GEval(
    name="Simplicidade para Leigos (Sem RAG)",
    criteria="Avalie se a resposta é compreensível para uma pessoa com pouca escolaridade. A resposta deve ter frases curtas, ser direta e estar livre de jargões jurídicos complexos.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o",
)

relevance_metric = GEval(
    name="Relevância da Resposta (Sem RAG)",
    criteria="Avalie se a resposta aborda de forma útil, direta e completa a pergunta do usuário. A resposta não deve conter informações irrelevantes, não deve fugir do tema e deve fazer sentido lógico independentemente de onde a informação foi tirada.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    model="gpt-4o",
)
questions = [
    # insira o dataset aqui, por exemplo:
    "O banco pode descontar a parcela do meu empréstimo direto da minha conta salário e me deixar sem dinheiro nenhum?",
]

with open("1_respostas.txt", "w", encoding="utf-8") as f: f.write("")
with open("2_textos_recuperados.txt", "w", encoding="utf-8") as f: f.write("")
with open("3_metricas_detalhadas.txt", "w", encoding="utf-8") as f: f.write("")
with open("4_tabela_metricas_excel.txt", "w", encoding="utf-8") as f: 
    f.write("Index\tFidelidade_Score\tSimplicidade_Score\tRelevância_Score\n")
# ─────────────────────────────────────────────────────────────────────────────

for i, query in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] Processando e Avaliando: {query}")

    sucesso = False
    tentativas = 0
    
    while not sucesso and tentativas < 5:
        try:
            relevant_docs = retriever.invoke(query)
            
            # Textos formatados para os arquivos
            retrieved_texts = "\n".join([f"  [{j}] {doc.page_content}" for j, doc in enumerate(relevant_docs, 1)])
            docs_text = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])
            docs_list = [doc.page_content for doc in relevant_docs]

            combined_input = f"""{FEW_SHOT_BLOCK}

## Agora responda esta nova pergunta seguindo exatamente o mesmo estilo dos exemplos acima:

Pergunta: {query}

Textos da lei disponíveis:
{docs_text}

Responda de forma simples, como nos exemplos. Evite inventar informações.
"""

            messages = [
                SystemMessage(content="""Você é um assistente que explica assuntos jurídicos para pessoas com pouca escolaridade.
Regras obrigatórias:
- Use frases curtas e simples.
- Evite palavras difíceis e termos jurídicos.
- Explique como se estivesse falando com alguém que não entende de leis.
- Use exemplos simples quando possível.
- Vá direto ao ponto.
Limites:
- Utilize primordialmente as informações dos documentos recuperados.
- Não invente leis ou regras.
Formato da resposta:
- No máximo 5 ou 7 linhas.
- Linguagem clara, como uma conversa.
- Sem palavras técnicas.
"""),
                HumanMessage(content=combined_input),
            ]

            result = model.invoke(messages)
            answer = result.content.strip()

            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=docs_list
            )

            # Medindo as 3 métricas
            faithfulness_metric.measure(test_case)
            simplicity_metric.measure(test_case)
            relevance_metric.measure(test_case)

            f_score = faithfulness_metric.score
            s_score = simplicity_metric.score
            r_score = relevance_metric.score
            
            f_reason = faithfulness_metric.reason
            s_reason = simplicity_metric.reason
            r_reason = relevance_metric.reason

            with open("1_respostas.txt", "a", encoding="utf-8") as f:
                f.write(f"[{i}] Pergunta:\n{query}\n\nResposta:\n{answer}\n{'-'*60}\n")
                
            with open("2_textos_recuperados.txt", "a", encoding="utf-8") as f:
                f.write(f"[{i}] Pergunta:\n{query}\n\nTextos recuperados:\n{retrieved_texts}\n{'-'*60}\n")
                
            with open("3_metricas_detalhadas.txt", "a", encoding="utf-8") as f:
                f.write(f"[{i}] Pergunta: {query}\n")
                f.write(f"Fidelidade: {f_score} | Motivo: {f_reason}\n")
                f.write(f"Simplicidade: {s_score} | Motivo: {s_reason}\n")
                f.write(f"Relevância: {r_score} | Motivo: {r_reason}\n")
                f.write("-" * 60 + "\n")
                
            with open("4_tabela_metricas_excel.txt", "a", encoding="utf-8") as f:
                f.write(f"{i}\t{f_score}\t{s_score}\t{r_score}\n")

            sucesso = True 

            print("  [Sucesso] Salvou no HD. Pausando 12s para esfriar a API...")
            time.sleep(12)

        except Exception as e:
            tentativas += 1
            print(f"  [Aviso] Erro detectado (Rate Limit ou Conexão). Tentativa {tentativas} de 5 falhou. Detalhes: {e}")
            print(f"  Aguardando 45 segundos antes de tentar a pergunta {i} novamente...")
            time.sleep(45) 

    if not sucesso:
        print(f"  [ERRO FATAL] Pulando a pergunta {i} após 5 tentativas falhas.")

print("\nProcessamento Finalizado! Todos os arquivos foram atualizados.")