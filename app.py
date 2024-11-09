import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Função para carregar e exibir dados (suporte para CSV, Excel e JSON)
def carregar_dados(arquivo):
    try:
        if arquivo.name.endswith('.csv'):
            df = pd.read_csv(arquivo)
        elif arquivo.name.endswith('.xlsx'):
            df = pd.read_excel(arquivo)
        elif arquivo.name.endswith('.json'):
            df = pd.read_json(arquivo)
        else:
            st.error("Formato de arquivo não suportado. Por favor, envie um arquivo CSV, Excel ou JSON.")
            return None
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função para tratamento básico de dados
def tratamento_basico(df):
    st.write("Primeiras linhas da base de dados:")
    st.write(df.head())

    st.write("Informações estatísticas:")
    st.write(df.describe())

    # Verificar dados ausentes
    st.write("Verificação de dados ausentes:")
    st.write(df.isnull().sum())

    # Opção para remover linhas com dados ausentes
    if st.checkbox("Remover linhas com dados ausentes"):
        df = df.dropna()
        st.success("Linhas com dados ausentes removidas!")
    
    return df

# Função para visualização de gráficos
def plotar_graficos(df):
    st.write("Escolha colunas numéricas para visualização gráfica:")
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(colunas_numericas) > 1:
        grafico = st.selectbox("Escolha o tipo de gráfico", ['Gráfico de Dispersão', 'Histograma', 'Correlação', 'Radar Chart'])

        if grafico == 'Gráfico de Dispersão':
            x = st.selectbox("Escolha a coluna para o eixo X", colunas_numericas)
            y = st.selectbox("Escolha a coluna para o eixo Y", colunas_numericas)
            fig, ax = plt.subplots()
            ax = sns.scatterplot(data=df, x=x, y=y)
            st.pyplot(fig)

        elif grafico == 'Histograma':
            coluna = st.selectbox("Escolha a coluna para o histograma", colunas_numericas)
            fig, ax = plt.subplots()
            ax = sns.histplot(df[coluna], kde=True)
            st.pyplot(fig)

        
        elif grafico == 'Radar Chart':
            categorias = st.multiselect("Escolha categorias para o Radar Chart", colunas_numericas)
            if len(categorias) > 2:
                df_mean = df[categorias].mean()
                fig = plt.figure()
                ax = fig.add_subplot(111, polar=True)
                ax.plot(df_mean, label="Médias das variáveis")
                ax.fill(df_mean, alpha=0.3)
                st.pyplot(fig)
            else:
                st.write("Selecione pelo menos 3 categorias para o gráfico de radar.")
    else:
        st.write("Não há colunas numéricas suficientes para gerar gráficos.")

# Função para aplicar o KMeans e segmentar clientes
def aplicar_kmeans(df):
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(colunas_numericas) > 1:
        n_clusters = st.slider("Escolha o número de clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        df['cluster'] = kmeans.fit_predict(df[colunas_numericas])

        st.write(f"Clusters formados: {n_clusters}")
        st.write(df[['cluster'] + colunas_numericas].head())

        # Plot clusters
        if 'cluster' in df.columns:
            x = st.selectbox("Eixo X para plotar clusters", colunas_numericas)
            y = st.selectbox("Eixo Y para plotar clusters", colunas_numericas)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x, y=y, hue='cluster', palette='viridis')
            st.pyplot(fig)
    else:
        st.write("Dados insuficientes para clustering.")


# Aplicativo principal
def main():
    st.title("Agrupamento de Clientes com K-Means")

    # Organize o layout em colunas
    col1, col2 = st.columns(2)

    arquivo = col1.file_uploader("Faça o upload do arquivo CSV/Excel/JSON", type=['csv', 'xlsx', 'json'])

    if arquivo is not None:
        df = carregar_dados(arquivo)
        if df is not None:
            with col2:
                st.write("Informações do arquivo")
                st.write(df.head())

            df_tratado = tratamento_basico(df)

            # Exibir gráficos
            plotar_graficos(df_tratado)


if __name__ == '__main__':
    main()
