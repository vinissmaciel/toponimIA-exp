# toponimIA-exp
Repositório com código-fonte do experimento executado no TCC ToponímIA.

## Estrutura de Arquivos

- `experimento.py` – experimento do projeto contendo construção da base de conhecimento, workflow RAG e cálculo dos scores
- `final.py` – junção dos lotes e geração do CSV e gráficos finais
- `50_exp.py` – geração do CSV da amostra de 50 linhas aleatórias

## Execução

Certifique-se de que o Ollama está rodando localmente e o modelo Llama3.2 está carregado.

## Requisitos
pip install pandas torch langchain-huggingface langchain-ollama langchain-core langgraph sentence-transformers matplotlib numpy

```bash
python experimento.py
python final.py
python 50_exp.py
