import pandas as pd

import torch
import gc

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama

from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph

from sentence_transformers import SentenceTransformer, util

# ============= Modelo de embedding =====================

model_name = "intfloat/multilingual-e5-base"

model_kwargs = {'device': 'cpu'} # Inicializar modelo no processador
encode_kwargs = {'normalize_embeddings': True} # Normalizar embeddings

# Inicializa o modelo de embedding
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Inicializa banco de dados vetorial em memória com modelo de embedding
vector_store = InMemoryVectorStore(embeddings)

# ============= Dataset =====================

df = pd.read_csv('./docs_rag/dataset_toponimia_lisboa_portugal.csv')

cols_para_texto = ['DESIGNACAO', 'LEGENDA', 'FREGUESIAS', 'TEMAS', 'DENOMINACOES_ANTERIORES', 'HISTORIAL']
for col in cols_para_texto:
    df[col] = df[col].fillna('').astype(str).str.strip() # Garante que tudo seja string

df['FREGUESIAS'] = df['FREGUESIAS'].str.replace(r'\(Nova Freguesia\)', '', regex=True).str.strip() # Retira o termo 'Nova Freguesia' das freguesias

# Cria o texto apenas com os campos que não estão vazios
def gerar_resumo(row):
    partes = []
    # Mapeamento de nome da coluna para rótulo no texto final
    mapa_campos = {
        'DESIGNACAO': 'Nome da rua',
        'FREGUESIAS': 'Freguesias',
        'HISTORIAL': 'Explicação/origem/história',
        'TEMAS': 'Temas',
        'LEGENDA': 'Legenda',
        'DENOMINACOES_ANTERIORES': 'Nomes anteriores'
    }
    
    for col, rotulo in mapa_campos.items():
        valor = row[col]
        if valor != '':  # Só adiciona se a string não for vazia
            partes.append(f"{rotulo}: {valor}")
            
    return "\n".join(partes)

# Lista para armazenar os documentos formatados
docs = []

# Definição dos valores do intervalo processado no lote
inicio = 0
fim = 500

# Itera sobre cada linha do DataFrame
for index, row in df.iterrows():

    resumo_limpo = gerar_resumo(row)
    
    # Se após a limpeza o resumo ficou vazio (linha sem dados úteis), pula
    if not resumo_limpo:
        continue

    # Prefixo "passage: " para o modelo de embedding
    page_content = "passage: " + resumo_limpo
    
    doc = Document(
        page_content=page_content,
        metadata={
            'row_index': index,
            'author': 'Câmara Municipal de Lisboa',
            'source': 'dataset_toponimia_lisboa_portugal.csv',
            'nome_rua_original': row['DESIGNACAO'],
            'freguesias': row['FREGUESIAS']
        }
    )
    docs.append(doc)

# ============= Carregamento dos embeddings no banco de dados =====================

document_ids = vector_store.add_documents(documents=docs)

# ============= Prompt =====================

template = """Você é um gerador de explicações sobre nomes de ruas (toponímia).
Sua tarefa é gerar explicações sobre a origem/história/significado da rua em questão.
Use os seguintes elementos de contexto e seus dados de treinamento para responder à pergunta final.
Caso o contexto esteja disponível utilize os dados dele com maior prioridade em relação aos de seu treinamento.
Se os dados sobre uma rua não estiver no contexto, tente responder com seu conhecimento paramétrico do treinamento, mas apenas se tiver certeza que a informação está correta.
Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.
Responda em português do Brasil.
A resposta deve ser apenas o texto, sem nada antes.
Desenvolva uma resposta completa, podendo extender e trazer detalhes. Caso seja uma pessoa, acrescente parte de sua biografia. 
Caso seja uma data, acrescente o que aconteceu ou o que aquela data representa para o local. Explore a etimologia se for apropriado. Tente não resumir muito a resposta.
Caso não tenha certeza sobre uma informação adicional fora do contexto, não adicione na resposta final.


Contexto: {context}

Questão: {question}

Resposta:"""
prompt = PromptTemplate.from_template(template) 

# ============= Modelo LLM =====================

llm = ChatOllama(
    model="llama3.2",
    temperature=0.1, # Temperatura baixa para factualidade
    keep_alive=-1, # Mantém a conexão aberta
)

# ============= Nós do LangGraph =====================

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search("query: " + state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# ============= Funções para gerar as explicações =====================

# IA padrão
def get_answer_IA(freguesia: str, rua: str):
    result = llm.invoke(f"Qual é o significado por trás do nome da rua {rua} em {freguesia}, Lisboa, Portugal?")
    return result.content

# Workflow RAG
def get_answer_RAG(freguesia: str, rua: str):
    question = f"Qual é o significado por trás do nome da rua {rua} em {freguesia}, Lisboa, Portugal?"
    result = graph.invoke({"question": question})

    if "answer" in result:
        return result["answer"], result["context"]
    
# ============= Função para gerar scores =====================
    
# Modelo de embedding para validação
model = SentenceTransformer('intfloat/multilingual-e5-base') 

def validacao(texto_oficial: str, texto_ia_padrao: str, texto_ia_rag: str):

    emb_oficial = model.encode("query: " + texto_oficial, normalize_embeddings=True)
    emb_padrao = model.encode("query: " + texto_ia_padrao, normalize_embeddings=True)
    emb_rag = model.encode("query: " + texto_ia_rag, normalize_embeddings=True)

    score_ia_padrao = util.cos_sim(emb_oficial, emb_padrao).item()
    score_ia_rag = util.cos_sim(emb_oficial, emb_rag).item()
    diferenca = score_ia_rag - score_ia_padrao

    return score_ia_padrao, score_ia_rag, diferenca

# ============= Execução do experimento =====================

counter = 0
results = []

num_exp = 7 # Número do lote
output_path = f'./results/validos/valid{num_exp}.csv'

print(f"Iniciando processamento de {inicio} até {fim}...")

for index, row in df.iloc[inicio:fim].iterrows():
    counter +=1
    if row['HISTORIAL'] != '':
        ia_generation = get_answer_IA(row['FREGUESIAS'], row['DESIGNACAO'])

        rag_generation, context = get_answer_RAG(row['FREGUESIAS'], row['DESIGNACAO'])
        
        ia_score, rag_score, diff = validacao(row['HISTORIAL'], ia_generation, rag_generation)

        results.append({
            'Logradouro': row['DESIGNACAO'],
            'Explicação oficial': row['HISTORIAL'],
            'Contexto': "\n\n".join(doc.page_content for doc in context),
            'Explicação IA': ia_generation,
            'Explicação RAG': rag_generation,
            'Score IA': ia_score,
            'Score RAG': rag_score,
            'Diferença': diff
        })

        print(f"Finished {counter}")
    else:
        print(f"Skipped {counter} - Missing explanation on dataset")

    if counter % 100 == 0:
        print(f"--> Checkpoint atingido (linha {counter}). Salvando parcial...")
        df_checkpoint = pd.DataFrame(results)
        df_checkpoint.to_csv(output_path, index=False)
        print(f"--> Parcial salvo com sucesso: {len(df_checkpoint)} registros.")

df_results = pd.DataFrame(results)
print("------------ FINISHED ------------")

df_results.to_csv(output_path, index=False)

print(f"Arquivo final salvo em: {output_path}")
print(df_results.head())

print(df_results.mean(numeric_only=True))

# ============= Geração dos gráficos do lote =====================

import matplotlib.pyplot as plt

df_plot = df_results
summary = pd.DataFrame({
        "mean": df_plot.mean(numeric_only=True),
        "median": df_plot.median(numeric_only=True),
        "std": df_plot.std(ddof=1, numeric_only=True),
    })

# 1) Média com barras de desvio padrão
plt.figure()
plt.title("Média com barras de desvio padrão")
plt.ylabel("valor")
means = summary["mean"]
stds = summary["std"]
plt.bar(means.index, means.values, yerr=stds.values, capsize=6)
plt.tight_layout()
plt.savefig(f'./results/validos/valid{num_exp}_media_desvio.png', dpi=160)
plt.show()

# 2) Medianas
plt.figure()
plt.title("Medianas por métrica")
plt.ylabel("valor")
medians = summary["median"]
plt.bar(medians.index, medians.values)
plt.tight_layout()
plt.savefig(f'./results/validos/valid{num_exp}_medianas.png', dpi=160)
plt.show()

# 3) Boxplots (distribuição)
cols = [ 'Score IA', 'Score RAG']
plt.figure()
plt.title("Distribuição (boxplot)")
plt.ylabel("valor")
plt.boxplot([df_plot[c].values for c in cols], labels=cols, showmeans=True)
plt.tight_layout()
plt.savefig(f'./results/validos/valid{num_exp}_boxplots.png', dpi=160)
plt.show()

# Salva resumo para consulta rápida
summary.to_csv(f'./results/validos/resumo_estatistico{num_exp}.csv', float_format='%.6f')