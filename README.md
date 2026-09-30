# ic-rag-llm-juridico

Textual Accessibility in the Brazilian Consumer Protection Code: A Comparative Analysis of LLMs Applied to a Legal Assistant

## Estrutura do Repositório

* `src/llm_evaluation/`: Scripts em Python para testar os LLMs, incluindo o pipeline de RAG (Retrieval-Augmented Generation) e os testes de acurácia com a base de dados.
* `src/web_prototype/`: Código-fonte do protótipo da interface web.
* `src/docs/`: documentos de referência utilizados na avaliação.
* `requirements.txt`: Lista de dependências Python necessárias para rodar o projeto.
* `dataset.txt`: Lista de perguntas utilizadas nos testes.
* `results/`: arquivos com métricas, repostas, documentos recuperados e resultados.

## Como rodar web_prototype

1. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure o `.env`:**
   ```bash
   # edite .env e coloque OPENAI_API_KEY e um SECRET_KEY aleatório
   ```

4. **Rode a ingestão** (cria a base vetorial ChromaDB).
   Existem duas formas:

   - **Via API** (depois de já estar rodando o servidor e logado): `POST /api/ingest`
   - **Via script direto**, equivalente ao seu `ingestion_pipeline.py` original:
     ```bash
     python3 -c "from rag_pipeline import run_ingestion; run_ingestion()"
     ```

5. **Inicie o servidor:**
   ```bash
   uvicorn main:app --reload
   ```

6. Acesse **http://localhost:8000** no navegador. Você será levado para
   a tela de **cadastro/login**. Depois de logar, cai direto no chat.
