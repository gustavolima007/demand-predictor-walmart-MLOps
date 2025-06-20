import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração inicial do Streamlit
st.title("Walmart Demand Predictor - Visualização de Previsões")
st.write("Interface interativa para visualizar previsões de vendas semanais por loja.")

# Carregar dados (simulação - substitua pelo seu carregamento real)
@st.cache_data
def load_data():
    # Substitua pelo seu carregamento de df_train_merged, X_teste, etc.
    df_validacao = pd.read_csv("data/validacao.csv")  # Exemplo
    X_validacao = pd.read_csv("data/X_validacao.csv")  # Exemplo
    y_validacao = pd.read_csv("data/y_validacao.csv")['Weekly_Sales']  # Exemplo
    return df_validacao, X_validacao, y_validacao

df_validacao, X_validacao, y_validacao = load_data()

# Treinar modelo (simulação - use seu código existente)
@st.cache_resource
def train_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=30, max_depth=7, min_samples_split=50,
                                     min_samples_leaf=50, max_features='sqrt', random_state=42, n_jobs=2)
    rf_model.fit(X_train, y_train)
    return rf_model

# Supondo que X_full_train e y_full_train estejam disponíveis
final_rf_model = train_model(X_validacao, y_validacao)  # Ajuste para X_full_train e y_full_train

# Seleção de loja
stores = df_validacao['Store'].unique()
selected_store = st.selectbox("Selecione uma Loja", stores)

# Filtrar dados pela loja selecionada
df_store = df_validacao[df_validacao['Store'] == selected_store]
X_store = X_validacao[df_validacao['Store'] == selected_store]
y_store = y_validacao[df_validacao['Store'] == selected_store]

# Fazer previsões para a loja selecionada
y_pred_rf = final_rf_model.predict(X_store)

# Métricas
mae_rf = mean_absolute_error(y_store, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_store, y_pred_rf))
r2_rf = r2_score(y_store, y_pred_rf)

st.write(f"**Métricas para Loja {selected_store}:**")
st.write(f"R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

# Gráficos
st.subheader("Visualizações")

# 1. Valores Reais vs. Previstos
st.subheader("Valores Reais vs. Previstos")
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.scatter(y_store, y_pred_rf, alpha=0.4, c='blue', s=30, edgecolors='none')
lims = [min(y_store.min(), y_pred_rf.min()), max(y_store.max(), y_pred_rf.max())]
ax1.plot(lims, lims, 'k--', lw=2, label='Previsão Perfeita (y=x)')
ax1.set_xlabel("Valores Reais (Weekly_Sales)")
ax1.set_ylabel("Valores Previstos (Weekly_Sales)")
ax1.set_title(f"Valores Reais vs. Previstos (Loja {selected_store})\nR²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")
ax1.legend()
st.pyplot(fig1)

# 2. Gráfico de Resíduos
st.subheader("Gráfico de Resíduos")
residuos_rf = y_store - y_pred_rf
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.scatter(y_pred_rf, residuos_rf, alpha=0.4, c='blue', s=30, edgecolors='none')
ax2.axhline(y=0, color='red', linestyle='--', lw=2, label='Erro Zero')
ax2.axhline(y=residuos_rf.mean(), color='green', linestyle=':', lw=1.5, label=f'Média dos Resíduos: {residuos_rf.mean():.2f}')
ax2.fill_between([y_pred_rf.min(), y_pred_rf.max()], 
                 residuos_rf.mean() - residuos_rf.std(), 
                 residuos_rf.mean() + residuos_rf.std(), 
                 color='green', alpha=0.1, label=f'±1 Desvio Padrão: {residuos_rf.std():.2f}')
ax2.set_xlabel("Valores Previstos (Weekly_Sales)")
ax2.set_ylabel("Resíduos (Real - Previsto)")
ax2.set_title(f"Gráfico de Resíduos (Loja {selected_store})")
ax2.legend()
st.pyplot(fig2)

# 3. Série Temporal
st.subheader("Série Temporal - Vendas Reais vs. Previstas")
df_store_plot = df_store[['Date', 'Weekly_Sales']].copy()
df_store_plot['Weekly_Sales_Prevista'] = y_pred_rf
df_store_plot.sort_values('Date', inplace=True)
fig3, ax3 = plt.subplots(figsize=(14, 7))
ax3.plot(df_store_plot['Date'], df_store_plot['Weekly_Sales'], label='Vendas Reais', marker='o', linestyle='-', linewidth=2, color='blue')
ax3.plot(df_store_plot['Date'], df_store_plot['Weekly_Sales_Prevista'], label='Vendas Previstas (RF)', marker='x', linestyle='--', linewidth=2, color='green')
ax3.set_xlabel("Data")
ax3.set_ylabel("Weekly_Sales")
ax3.set_title(f"Vendas Reais vs. Previstas (Loja {selected_store})")
ax3.legend()
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# 4. Histograma dos Resíduos
st.subheader("Histograma dos Resíduos")
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.histplot(residuos_rf, kde=True, bins=50, color='blue', stat='density', ax=ax4)
ax4.axvline(residuos_rf.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Média: {residuos_rf.mean():.2f}')
ax4.axvline(residuos_rf.median(), color='purple', linestyle=':', linewidth=1.5, label=f'Mediana: {residuos_rf.median():.2f}')
ax4.set_xlabel("Resíduos (Real - Previsto)")
ax4.set_ylabel("Densidade")
ax4.set_title(f"Histograma dos Resíduos (Loja {selected_store})\nSkewness: {residuos_rf.skew():.2f}")
ax4.legend()
st.pyplot(fig4)

# 5. Importância das Features
st.subheader("Importância das Features")
importances = final_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_store.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
fig5, ax5 = plt.subplots(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis', ax=ax5)
ax5.set_title(f'Top 15 Features Mais Importantes (Loja {selected_store})')
ax5.set_xlabel('Importância')
ax5.set_ylabel('Feature')
st.pyplot(fig5)

# Adicionar botão para download das previsões
st.download_button(
    label="Baixar Previsões (CSV)",
    data=df_store_plot.to_csv(index=False),
    file_name=f"previsoes_loja_{selected_store}.csv",
    mime="text/csv"
)