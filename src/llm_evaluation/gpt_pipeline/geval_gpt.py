import os
from openai import OpenAI
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

simplicity_metric = GEval(
    name="Simplicidade para Leigos (Sem RAG)",
    criteria="Avalie se a resposta é compreensível para uma pessoa com pouca escolaridade. A resposta deve ter frases curtas, ser direta e estar livre de jargões jurídicos complexos.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
)

relevance_metric = GEval(
    name="Relevância da Resposta (Sem RAG)",
    criteria="Avalie se a resposta aborda de forma útil, direta e completa a pergunta do usuário. A resposta não deve conter informações irrelevantes, não deve fugir do tema e deve fazer sentido lógico independentemente de onde a informação foi tirada.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
)

dataset = [
    # insira o dataset aqui, por exemplo:
    "O banco pode descontar a parcela do meu empréstimo direto da minha conta salário e me deixar sem dinheiro nenhum?",
]

def gerar_resposta_chatgpt(pergunta):
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "Você é um assistente jurídico para leigos, responda a perguta do usuário entre 5 a 7 linhas."},
            {"role": "user", "content": pergunta}
        ],
        temperature=0.0,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

resultados_qa = []
resultados_metricas = []
resultados_excel = []

resultados_excel.append("Índice\tScore_Simplicidade\tScore_Relevancia")

print("Iniciando avaliação das perguntas...")

for index, pergunta in enumerate(dataset, start=1):
    print(f"\nProcessando Pergunta {index}/{len(dataset)}: {pergunta}")
    
    resposta = gerar_resposta_chatgpt(pergunta)
    
    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta
    )
    
    # 1 Medir Simplicidade
    simplicity_metric.measure(test_case)
    score_simp = simplicity_metric.score
    razao_simp = simplicity_metric.reason
    
    # 2 Medir Relevância
    relevance_metric.measure(test_case)
    score_rel = relevance_metric.score
    razao_rel = relevance_metric.reason

    
    # Arquivo 1: Q&A
    resultados_qa.append(f"--- PERGUNTA {index} ---\n{pergunta}\n\n--- RESPOSTA {index} ---\n{resposta}\n\n")
    
    # Arquivo 2: Métricas Detalhadas
    detalhe_metrica = (
        f"=== PERGUNTA {index} ===\n"
        f"MÉTRICA 1: Simplicidade\nScore: {score_simp}\nJustificativa: {razao_simp}\n\n"
        f"MÉTRICA 2: Relevância\nScore: {score_rel}\nJustificativa: {razao_rel}\n"
        f"{'='*30}\n\n"
    )
    resultados_metricas.append(detalhe_metrica)
    
    # Arquivo 3: Para Excel (Índice | Score 1 | Score 2)
    resultados_excel.append(f"{index}\t{score_simp}\t{score_rel}")

print("\nGerando os arquivos TXT...")

# Arquivo 1: Perguntas e Respostas
with open("1_perguntas_e_respostas.txt", "w", encoding="utf-8") as f:
    f.writelines(resultados_qa)

# Arquivo 2: Métricas Detalhadas
with open("2_metricas_detalhadas.txt", "w", encoding="utf-8") as f:
    f.writelines(resultados_metricas)

# Arquivo 3: Índices e Scores para Excel
with open("3_scores_para_excel.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(resultados_excel))

print("Avaliação concluída! Os 3 arquivos foram gerados com sucesso na pasta atual.")