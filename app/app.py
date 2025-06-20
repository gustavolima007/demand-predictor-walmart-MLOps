import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuração inicial do Streamlit
st.set_page_config(layout="wide")
st.title("Walmart Demand Predictor - Visualização de Previsões")
st.write("Interface interativa para visualizar previsões de vendas semanais por loja e departamento.")

# Exibir diretório de trabalho atual para depuração
st.write(f"Diretório de trabalho atual: {os.getcwd()}")

# Função para carregar dados localmente com verificação
@st.cache_data
def load_data():
    base_path = r"C:\Users\01701805\Desktop\MLOps\demand-predictor-walmart-MLOps\notebooks\data"
    file_paths = {
        "X_full_train": "X_full_train.csv",
        "y_full_train": "y_full_train.csv",
        "X_validacao": "X_validacao.csv",
        "y_validacao": "y_validacao.csv",
        "X_teste": "X_teste.csv"
    }

    data = {}
    for name, file in file_paths.items():
        full_path = os.path.join(base_path, file)
        if os.path.exists(full_path):
            if name in ["y_full_train", "y_validacao"]:
                data[name] = pd.read_csv(full_path)['Weekly_Sales']
            else:
                data[name] = pd.read_csv(full_path)
            st.write(f"Carregado: {full_path}")
        else:
            st.error(f"Arquivo não encontrado: {full_path}")
            raise FileNotFoundError(f"Verifique se o arquivo {file} existe em {base_path}")
    
    return data["X_full_train"], data["y_full_train"], data["X_validacao"], data["y_validacao"], data["X_teste"]

# Função para carregar o modelo
@st.cache_resource
def load_model():
    model_path = r"C:\Users\01701805\Desktop\MLOps\demand-predictor-walmart-MLOps\notebooks\data\final_rf_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Modelo não encontrado: {model_path}")
        raise FileNotFoundError(f"Verifique se o arquivo {model_path} existe")

# Carregar dados e modelo
try:
    X_full_train, y_full_train, X_validacao, y_validacao, X_teste = load_data()
    final_rf_model = load_model()
except FileNotFoundError as e:
    st.error(f"Erro ao carregar dados ou modelo: {e}")
    st.stop()

# Verificar colunas para filtragem
has_store = 'Store' in X_full_train.columns or 'Store' in X_validacao.columns
has_dept = 'Dept' in X_full_train.columns or 'Dept' in X_validacao.columns
has_date = 'Date' in X_full_train.columns or 'Date' in X_validacao.columns

# Usar X_full_train como base para filtros, se disponível
if has_store or has_dept:
    if has_store:
        stores = X_full_train['Store'].unique() if 'Store' in X_full_train.columns else X_validacao['Store'].unique()
        selected_store = st.sidebar.selectbox("Selecione uma Loja", stores)
    else:
        selected_store = None

    if has_dept:
        departments = (X_full_train[X_full_train['Store'] == selected_store]['Dept'].unique() 
                      if 'Store' in X_full_train.columns and 'Dept' in X_full_train.columns 
                      else X_validacao[X_validacao['Store'] == selected_store]['Dept'].unique())
        selected_dept = st.sidebar.selectbox("Selecione um Departamento", departments)
    else:
        selected_dept = None

    # Filtrar dados
    if has_store and has_dept:
        mask = (X_validacao['Store'] == selected_store) & (X_validacao['Dept'] == selected_dept) if 'Store' in X_validacao.columns and 'Dept' in X_validacao.columns else (X_full_train['Store'] == selected_store) & (X_full_train['Dept'] == selected_dept)
        X_store_dept = X_validacao[mask] if 'Store' in X_validacao.columns and 'Dept' in X_validacao.columns else X_full_train[mask]
        y_store_dept = y_validacao[:len(X_store_dept)] if 'Store' in X_validacao.columns and 'Dept' in X_validacao.columns else y_full_train[:len(X_store_dept)]
    elif has_store:
        mask = X_validacao['Store'] == selected_store if 'Store' in X_validacao.columns else X_full_train['Store'] == selected_store
        X_store_dept = X_validacao[mask] if 'Store' in X_validacao.columns else X_full_train[mask]
        y_store_dept = y_validacao[:len(X_store_dept)] if 'Store' in X_validacao.columns else y_full_train[:len(X_store_dept)]
    else:
        X_store_dept = X_validacao.copy()
        y_store_dept = y_validacao.copy()
else:
    st.write("Aviso: Não foi possível carregar filtros por loja e departamento. Visualização será baseada em todos os dados de validação.")
    X_store_dept = X_validacao.copy()
    y_store_dept = y_validacao.copy()

# Fazer previsões
y_pred_rf = final_rf_model.predict(X_store_dept)

# Métricas
mae_rf = mean_absolute_error(y_store_dept, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_store_dept, y_pred_rf))
r2_rf = r2_score(y_store_dept, y_pred_rf)

st.write(f"### Métricas para {'Todos os Dados de Validação' if not (has_store or has_dept) else f'Loja {selected_store}, Dept {selected_dept}' if has_dept else f'Loja {selected_store}'}")
st.write(f"- **R²**: {r2_rf:.4f}")
st.write(f"- **RMSE**: {rmse_rf:.2f}")
st.write(f"- **MAE**: {mae_rf:.2f}")

# Gráficos Interativos com Plotly
st.header("Visualizações")

# 1. Valores Reais vs. Previstos
st.subheader("Valores Reais vs. Previstos")
fig1 = px.scatter(x=y_store_dept, y=y_pred_rf, trendline="ols", 
                  labels={"x": "Valores Reais (Weekly_Sales)", "y": "Valores Previstos (Weekly_Sales)"},
                  title=f"Valores Reais vs. Previstos {'(Todos os Dados de Validação)' if not (has_store or has_dept) else f'(Loja {selected_store}, Dept {selected_dept})' if has_dept else f'(Loja {selected_store})'}\nR²: {r2_rf:.4f}")
fig1.add_shape(type="line", x0=min(y_store_dept.min(), y_pred_rf.min()), y0=min(y_store_dept.min(), y_pred_rf.min()),
               x1=max(y_store_dept.max(), y_pred_rf.max()), y1=max(y_store_dept.max(), y_pred_rf.max()),
               line=dict(color="black", width=2, dash="dash"), name="Previsão Perfeita (y=x)")
st.plotly_chart(fig1)

# 2. Gráfico de Resíduos
st.subheader("Gráfico de Resíduos")
residuos_rf = y_store_dept - y_pred_rf
fig2 = px.scatter(x=y_pred_rf, y=residuos_rf, 
                  labels={"x": "Valores Previstos (Weekly_Sales)", "y": "Resíduos (Real - Previsto)"},
                  title=f"Gráfico de Resíduos {'(Todos os Dados de Validação)' if not (has_store or has_dept) else f'(Loja {selected_store}, Dept {selected_dept})' if has_dept else f'(Loja {selected_store})'}")
fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Erro Zero")
fig2.add_hline(y=residuos_rf.mean(), line_dash="dot", line_color="green", annotation_text=f"Média: {residuos_rf.mean():.2f}")
fig2.add_vrect(x0=y_pred_rf.min(), x1=y_pred_rf.max(), y0=residuos_rf.mean() - residuos_rf.std(), 
               y1=residuos_rf.mean() + residuos_rf.std(), fillcolor="green", opacity=0.1, 
               annotation_text=f"±1 SD: {residuos_rf.std():.2f}")
st.plotly_chart(fig2)

# 3. Histograma dos Resíduos
st.subheader("Histograma dos Resíduos")
fig4 = px.histogram(residuos_rf, nbins=30, title=f"Histograma dos Resíduos {'(Todos os Dados de Validação)' if not (has_store or has_dept) else f'(Loja {selected_store}, Dept {selected_dept})' if has_dept else f'(Loja {selected_store})'}\nSkewness: {residuos_rf.skew():.2f}",
                    labels={"value": "Resíduos (Real - Previsto)", "count": "Densidade"},
                    marginal="rug")
fig4.add_vline(x=residuos_rf.mean(), line_dash="dash", line_color="red", annotation_text=f"Média: {residuos_rf.mean():.2f}")
fig4.add_vline(x=residuos_rf.median(), line_dash="dot", line_color="purple", annotation_text=f"Mediana: {residuos_rf.median():.2f}")
st.plotly_chart(fig4)

# 4. Importância das Features
st.subheader("Importância das Features")
importances = final_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_store_dept.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
fig5 = px.bar(feature_importance_df.head(15), x='Importance', y='Feature', 
              title=f"Top 15 Features Mais Importantes {'(Todos os Dados de Validação)' if not (has_store or has_dept) else f'(Loja {selected_store}, Dept {selected_dept})' if has_dept else f'(Loja {selected_store})'}",
              labels={'Importance': 'Importância', 'Feature': 'Feature'},
              orientation='h')
st.plotly_chart(fig5)

# 5. Série Temporal - Vendas Reais vs. Previstas
st.subheader("Série Temporal - Vendas Reais vs. Previstas")
if has_date:
    # Usar a coluna 'Date' como eixo x, se disponível
    df_plot = pd.DataFrame({
        'Date': X_store_dept['Date'] if 'Date' in X_store_dept.columns else X_full_train['Date'][:len(y_store_dept)],
        'Vendas Reais': y_store_dept,
        'Vendas Previstas': y_pred_rf
    })
    df_plot = df_plot.sort_values('Date')
    fig3 = px.line(df_plot, x='Date', y=['Vendas Reais', 'Vendas Previstas'],
                   labels={'value': 'Weekly_Sales', 'Date': 'Data'},
                   title=f"Série Temporal - Vendas Reais vs. Previstas {'(Todos os Dados de Validação)' if not (has_store or has_dept) else f'(Loja {selected_store}, Dept {selected_dept})' if has_dept else f'(Loja {selected_store})'}")
else:
    # Usar índice numérico como proxy, se 'Date' não estiver disponível
    df_plot = pd.DataFrame({
        'Index': range(len(y_store_dept)),
        'Vendas Reais': y_store_dept,
        'Vendas Previstas': y_pred_rf
    })
    fig3 = px.line(df_plot, x='Index', y=['Vendas Reais', 'Vendas Previstas'],
                   labels={'value': 'Weekly_Sales', 'Index': 'Índice Temporal'},
                   title=f"Série Temporal - Vendas Reais vs. Previstas {'(Todos os Dados de Validação)' if not (has_store or has_dept) else f'(Loja {selected_store}, Dept {selected_dept})' if has_dept else f'(Loja {selected_store})'} (Sem Data)")
st.plotly_chart(fig3)

# Botão para download das previsões
st.subheader("Download")
st.download_button(
    label="Baixar Previsões (CSV)",
    data=pd.DataFrame({'Weekly_Sales': y_store_dept, 'Weekly_Sales_Prevista': y_pred_rf}).to_csv(index=False),
    file_name=f"previsoes_{'validacao' if not (has_store or has_dept) else f'loja_{selected_store}_dept_{selected_dept}' if has_dept else f'loja_{selected_store}'}.csv",
    mime="text/csv"
)

# Adicionar informações adicionais
st.sidebar.header("Sobre")
st.sidebar.write("Projeto desenvolvido por Gustavo Ferreira Lima (RA: 2023611300) para CET0621 - Aprendizado de Máquina na Análise de Dados (Unicamp).")
st.sidebar.write("Repositório: [GitHub](https://github.com/gustavolima007/demand-predictor-walmart-MLOps)")