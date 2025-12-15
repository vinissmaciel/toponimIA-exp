import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_final = pd.read_csv('./results/final/final.csv')

# Pega 50 linhas aleatórias do df_final
df_amostra = df_final.sample(n=50)

def embaralhar_explicacoes(row):
    """
    Recebe uma linha (row), pega os valores de 'Explicação IA' e 'Explicação RAG',
    sorteia a ordem e retorna os novos valores.
    """
    exp_ia = row['Explicação IA']
    exp_rag = row['Explicação RAG']
    
    # Sorteia um número: se for > 0.5, RAG vem primeiro
    if np.random.rand() > 0.5:
        # Caso 1: RAG é a 'Explicação 1'
        exp1 = exp_rag
        exp2 = exp_ia
        posicao_rag = 1
    else:
        # Caso 2: IA é a 'Explicação 1' (e RAG é a 2)
        exp1 = exp_ia
        exp2 = exp_rag
        posicao_rag = 2
    
    # Retorna os três valores que virarão as novas colunas
    return exp1, exp2, posicao_rag

# 2. Aplica a função em todas as linhas (axis=1)
# O result_type='expand' permite que o retorno da função (exp1, exp2, posicao_rag)
# seja "explodido" em três novas colunas no DataFrame.
df_amostra[['Explicação 1', 'Explicação 2', 'Posição RAG']] = df_amostra.apply(
    embaralhar_explicacoes, 
    axis=1, 
    result_type='expand'
)

# 3. Adiciona as colunas vazias para a avaliação humana
df_amostra['Melhor explicação'] = ''
df_amostra['Status IA'] = ''
df_amostra['Status RAG'] = ''
df_amostra['Tipo alucinação RAG'] = ''
df_amostra['Observações'] = ''

# Salva o CSV COMPLETO com o gabarito (Posição RAG)
df_amostra.to_csv(
    './results/final/amostra_50_linhas.csv', 
    index=False
)