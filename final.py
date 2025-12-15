import matplotlib.pyplot as plt
import pandas as pd

# Caminho base e os números dos arquivos
caminho_base = './results/validos/'
numeros_arquivos = range(1, 8)  # Gera os números de 1 a 7

# Lê os CSVs, printa o tamanho e adiciona à lista
print("--- Verificando linhas por arquivo ---")
lista_de_dfs = []
for i in numeros_arquivos:
    nome_arquivo = f'{caminho_base}valid{i}.csv'
    
    # Lê o arquivo
    df = pd.read_csv(nome_arquivo)
    
    # Imprime a quantidade de linhas
    print(f"Arquivo valid{i}.csv: {len(df)} linhas")
    
    # Adiciona o df à lista
    lista_de_dfs.append(df)

print("--------------------------------------\n")

# Concatenar todos os DataFrames da lista em um único DataFrame
df_final = pd.concat(lista_de_dfs, ignore_index=True)

# Verificar o resultado final
print(f"Total de linhas no df_final: {len(df_final)}")

# ============= NORMALIZAÇÃO (Escala 0.7-1.0 para 0.0-1.0) =================
# Define o mínimo (0.7) e o range (0.3)
minimo_original = 0.7
range_original = 1.0 - minimo_original

df_final['Score IA Normalizado'] = (df_final['Score IA'] - minimo_original) / range_original
df_final['Score RAG Normalizado'] = (df_final['Score RAG'] - minimo_original) / range_original

# CÁLCULO DA NOVA DIFERENÇA
df_final['Diferença Normalizado'] = df_final['Score RAG Normalizado'] - df_final['Score IA Normalizado']

# Salva CSV
df_final.to_csv(f'./results/final/final.csv', float_format='%.6f')

# ============= Geração dos gráficos finais =====================

summary = pd.DataFrame({
    "mean": df_final.mean(numeric_only=True),
    "median": df_final.median(numeric_only=True),
    "std": df_final.std(ddof=1, numeric_only=True),
})

# 1) Média com barras de desvio padrão
plt.figure(figsize=(10, 6))
plt.title("Média com barras de desvio padrão")
plt.ylabel("valor")
means = summary["mean"]
stds = summary["std"]
plt.bar(means.index, means.values, yerr=stds.values, capsize=6)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right') # Gira os labels do eixo X
plt.tight_layout()
plt.savefig(f'./results/final/final_media_desvio.png', dpi=160)
plt.show()

# -------------------------------------------------------------------

# 2) Medianas
plt.figure(figsize=(10, 6))
plt.title("Medianas por métrica")
plt.ylabel("valor")
medians = summary["median"]
plt.bar(medians.index, medians.values)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right') # Gira os labels do eixo X
plt.tight_layout() 
plt.savefig(f'./results/final/final_medianas.png', dpi=160)
plt.show()

# -------------------------------------------------------------------

# 3) Boxplots (distribuição)
cols = ['Score IA', 'Score RAG', 'Score IA Normalizado', 'Score RAG Normalizado']
plt.figure(figsize=(10, 6)) 
plt.title("Distribuição (boxplot)")
plt.ylabel("valor")
plt.boxplot([df_final[c].values for c in cols], labels=cols, showmeans=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right') # Gira os labels do eixo X
plt.tight_layout() 
plt.savefig(f'./results/final/final_boxplots.png', dpi=160)
plt.show()

# -------------------------------------------------------------------

# Salva resumo para consulta rápida
summary.to_csv(f'./results/final/resumo_estatistico_final.csv', float_format='%.6f')