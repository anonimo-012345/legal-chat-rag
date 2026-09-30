import os
import ollama
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

NOME_MODELO = 'llama3.1:8b' 

system_prompt = """
Você é um assistente jurídico para leigos, responda a pergunta do usuário entre 5 a 7 linhas.
"""
instrucao_adicional = "Resuma entre 5 a 7 linhas"

dataset_perguntas = [
    # insira o dataset aqui, por exemplo:
    "O banco pode descontar a parcela do meu empréstimo direto da minha conta salário e me deixar sem dinheiro nenhum?",

]

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

resultados_qa = []
resultados_metricas = []
resultados_excel = []

resultados_excel.append("Índice\tScore_Simplicidade\tScore_Relevancia")

print(f"Iniciando o processamento e avaliação de {len(dataset_perguntas)} perguntas...")

for index, pergunta in enumerate(dataset_perguntas, start=1):
    print(f"\n[{index}/{len(dataset_perguntas)}] Processando: {pergunta}")
    
    prompt_usuario = f"{pergunta}\n\nInstrução: {instrucao_adicional}"
    
    try:
        response = ollama.chat(model=NOME_MODELO, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt_usuario}
        ])
        resposta_ollama = response['message']['content'].strip()
    except Exception as e:
        print(f"  -> Erro ao gerar resposta com Ollama: {e}")
        resposta_ollama = f"ERRO NA GERAÇÃO: {e}"

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_ollama
    )
    
    try:
        # 1. Medir Simplicidade
        simplicity_metric.measure(test_case)
        score_simp = simplicity_metric.score
        razao_simp = simplicity_metric.reason
        
        # 2. Medir Relevância
        relevance_metric.measure(test_case)
        score_rel = relevance_metric.score
        razao_rel = relevance_metric.reason
    except Exception as e:
        print(f"  -> Erro ao avaliar com GPT-4o: {e}")
        score_simp, razao_simp = 0.0, f"Erro: {e}"
        score_rel, razao_rel = 0.0, f"Erro: {e}"
    
    
    resultados_qa.append(f"--- PERGUNTA {index} ---\n{pergunta}\n\n--- RESPOSTA {index} ---\n{resposta_ollama}\n\n")
    
    detalhe_metrica = (
        f"=== PERGUNTA {index} ===\n"
        f"MÉTRICA 1: Simplicidade\nScore: {score_simp}\nJustificativa: {razao_simp}\n\n"
        f"MÉTRICA 2: Relevância\nScore: {score_rel}\nJustificativa: {razao_rel}\n"
        f"{'='*40}\n\n"
    )
    resultados_metricas.append(detalhe_metrica)
    

    resultados_excel.append(f"{index}\t{score_simp}\t{score_rel}")

print("\nGerando os arquivos TXT...")

with open("1_perguntas_e_respostas1.txt", "w", encoding="utf-8") as f:
    f.writelines(resultados_qa)

with open("2_metricas_detalhadas1.txt", "w", encoding="utf-8") as f:
    f.writelines(resultados_metricas)

with open("3_scores_para_excel1.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(resultados_excel))

print("Avaliação concluída! Os 3 arquivos foram gerados com sucesso na pasta atual.")