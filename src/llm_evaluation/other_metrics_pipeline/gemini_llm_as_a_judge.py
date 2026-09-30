import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Inicialize o cliente utilizando a nova biblioteca google.genai.
# Certifique-se de ter a variável de ambiente GEMINI_API_KEY configurada.
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def g_eval_juridico(perguntas: list[str], respostas: list[str]):
    if len(perguntas) != len(respostas):
        raise ValueError("As listas de perguntas e respostas devem ter o mesmo tamanho.")

    resultados = []
    
    prompt_base = """
    Você é um juiz especialista em direito brasileiro avaliando a qualidade de respostas fornecidas por um sistema de IA para leigos.
    Sua métrica de avaliação é a 'Veracidade e Corretude Jurídica'.
    
    CRITÉRIOS:
    1. A resposta baseia-se em leis, códigos brasileiros e entendimentos jurídicos reais (sem alucinações legais)?
    2. A aplicação do direito do consumidor ao caso concreto do usuário está correta e precisa?
    3. As informações prejudicariam um usuário leigo caso ele seguisse essa orientação?
    
    ESCALA DE NOTAS:
    1 - Totalmente incorreta, cita leis inexistentes ou orienta de forma contrária ao direito.
    2 - Contém erros graves de interpretação jurídica que podem prejudicar o usuário.
    3 - Parcialmente correta, mas omite exceções vitais ou peca na precisão técnica.
    4 - Quase perfeita, juridicamente correta, mas com pequenas imprecisões irrelevantes.
    5 - Impecável, 100% verdadeira, correta juridicamente e perfeitamente aplicável ao caso.
    
    PERGUNTA DO USUÁRIO: {pergunta}
    RESPOSTA DA IA: {resposta}
    
    Pense passo a passo sobre a corretude e depois atribua a nota.
    Retorne APENAS um objeto JSON neste formato exato, sem formatação markdown extra:
    {{
        "justificativa": "<análise jurídica detalhada passo a passo>",
        "nota": <numero inteiro de 1 a 5>
    }}
    """

    print("Iniciando avaliação G-Eval com Gemini...")
    
    for i, (pergunta, resposta) in enumerate(zip(perguntas, respostas)):
        print(f"Avaliando par {i+1}/{len(perguntas)}...")
        prompt_formatado = prompt_base.format(pergunta=pergunta, resposta=resposta)
        
        try:

            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt_formatado,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1 
                )
            )

            avaliacao = json.loads(response.text)
            
            resultados.append({
                "pergunta": pergunta,
                "resposta": resposta,
                "nota": avaliacao.get("nota"),
                "justificativa": avaliacao.get("justificativa")
            })
            
        except Exception as e:
            print(f"Erro ao avaliar o par {i+1}: {e}")
            resultados.append({
                "pergunta": pergunta,
                "resposta": resposta,
                "nota": "ERRO",
                "justificativa": str(e)
            })

    with open("avaliacao_detalhada.txt", "w", encoding="utf-8") as f_detalhado:
        f_detalhado.write("=== RELATÓRIO DE AVALIAÇÃO JURÍDICA (G-EVAL) ===\n\n")
        for i, res in enumerate(resultados):
            f_detalhado.write(f"--- AVALIAÇÃO {i+1} ---\n")
            f_detalhado.write(f"Pergunta: {res['pergunta']}\n")
            f_detalhado.write(f"Resposta IA: {res['resposta']}\n")
            f_detalhado.write(f"Nota: {res['nota']}/5\n")
            f_detalhado.write(f"Justificativa: {res['justificativa']}\n")
            f_detalhado.write("\n" + "="*50 + "\n\n")
            
    with open("tabela_excel.txt", "w", encoding="utf-8") as f_tabela:
        f_tabela.write("ID\tPergunta\tResposta\tNota\tJustificativa\n")
        for i, res in enumerate(resultados):
            p_clean = res['pergunta'].replace('\n', ' ').replace('\t', ' ')
            r_clean = res['resposta'].replace('\n', ' ').replace('\t', ' ')
            j_clean = res['justificativa'].replace('\n', ' ').replace('\t', ' ')
            
            f_tabela.write(f"{i+1}\t{p_clean}\t{r_clean}\t{res['nota']}\t{j_clean}\n")

    print("\nAvaliação concluída com sucesso!")
    print("Arquivos gerados: 'avaliacao_detalhada.txt' e 'tabela_excel.txt'")

if __name__ == "__main__":
    lista_perguntas = [
        # insira o dataset aqui.
    ]
    
    lista_respostas = [
     #adicione as respostas geradas aqui, exemplo: usar respotas em results/

]

    g_eval_juridico(lista_perguntas, lista_respostas)