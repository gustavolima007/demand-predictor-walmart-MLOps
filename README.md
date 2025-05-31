# 🛍️ Walmart Demand Predictor MLOps: Previsão de Demanda para Categorias Chave de Produtos no Varejo Alimentício 🚀

📊 Projeto final da disciplina CET0621 - Aprendizado de Máquina na Análise de Dados (Unicamp - Faculdade de Tecnologia). Este trabalho foca no desenvolvimento de um sistema de previsão de vendas semanais para diferentes categorias de produtos (departamentos) da rede Walmart, utilizando uma abordagem estruturada de aprendizado de máquina.

---

## 📝 Descrição do Projeto

Este projeto desenvolve um modelo de aprendizado de máquina para prever as vendas semanais por categoria (departamento) e loja na rede Walmart. Utilizando um pipeline completo de ciência de dados – da limpeza à modelagem avançada com dados históricos e contextuais – buscamos gerar previsões precisas para otimizar estoques e o planejamento estratégico no varejo alimentício, seguindo boas práticas de organização e reprodutibilidade.

---

## 🎯 Problema Abordado

A previsão acurada da demanda é vital no varejo alimentício, onde erros geram perdas por excesso de estoque (desperdício) ou por falta de produtos (vendas perdidas e clientes insatisfeitos). Este projeto enfrenta esse desafio aplicando técnicas de regressão do aprendizado de máquina para estimar as vendas semanais por categoria de produto, visando maior eficiência e rentabilidade.

---

## 💾 Dataset Utilizado

O estudo é baseado no dataset "Walmart Recruiting - Store Sales Forecasting", disponibilizado na plataforma Kaggle. Este conjunto de dados oferece um rico histórico contendo:
* Vendas semanais detalhadas por loja e departamento (nossa proxy para "categoria chave de produto").
* Características das lojas (ex: tipo, tamanho).
* Fatores contextuais e promocionais (ex: temperatura, preço do combustível, remarcações promocionais, CPI, taxa de desemprego, indicadores de feriados).

🔗 **Fonte do Dataset:** [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

---

## 🛠️ Metodologia e Workflow Aplicados

O projeto seguiu as etapas fundamentais do processo de Descoberta de Conhecimento em Dados (KDD), adaptadas para um fluxo de trabalho de aprendizado de máquina:

1.  **📥 Carregamento e Unificação dos Dados:** Combinação dos arquivos CSV (`train`, `test`, `stores`, `features`) em dataframes estruturados.
2.  **🧹 Limpeza e Validação de Dados:**
    * Conversão e validação de tipos de dados (datas, numéricos, etc.).
    * Tratamento de valores inconsistentes (ex: vendas negativas, markdowns negativos foram ajustados para 0).
    * Verificação e tratamento de dados duplicados (não foram encontradas duplicatas significativas).
3.  **🩹 Tratamento de Valores Ausentes (NaNs):**
    * Preenchimento de NaNs nas colunas `MarkDown1-5` com 0 (indicando ausência de promoção).
    * Preenchimento de NaNs em `CPI` e `Unemployment` usando estratégias de preenchimento progressivo (`ffill`) e regressivo (`bfill`) agrupados por loja, garantindo consistência temporal.
4.  **🔄 Transformação de Variáveis:**
    * Aplicação de One-Hot Encoding para a variável categórica `Type` (tipo de loja).
    * Conversão da variável booleana `IsHoliday` para formato numérico (0/1).
5.  **✨ Engenharia de Features (Crucial para Performance):**
    * **Temporais:** Extração de Ano, Mês, Dia, Semana do Ano, Dia da Semana, Dia do Ano.
    * **De Lag:** Criação de features baseadas em `Weekly_Sales` de semanas anteriores (ex: lags de 1, 4, 12, 52 semanas) para capturar autocorrelação e sazonalidade.
    * **De Janela Móvel:** Cálculo de estatísticas (média, mediana, soma, desvio padrão) de `Weekly_Sales` sobre janelas de tempo passadas (ex: janelas de 4, 8, 12, 26, 52 semanas) para capturar tendências recentes e suavizar ruídos.
    * Tratamento de NaNs introduzidos pela criação de lags e janelas móveis (preenchidos com 0).
6.  **📊 Divisão Estratégica dos Dados:** Separação do conjunto de treino em um novo subconjunto de treino e um conjunto de validação, utilizando uma abordagem cronológica para simular um cenário de previsão real.
7.  **🤖 Modelagem Preditiva e Avaliação:**
    * Definição das variáveis X (features) e y (alvo - `Weekly_Sales`).
    * Treinamento e avaliação comparativa de múltiplos modelos de regressão:
        * Regressão Linear (como baseline).
        * Árvore de Decisão Regressora.
        * Random Forest Regressor.
        * Gradient Boosting Regressor.
    * Avaliação de performance utilizando métricas chave: Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE) e Coeficiente de Determinação (R²), todas calculadas no conjunto de validação.
8.  **🚀 Treinamento do Modelo Final e Geração de Previsões:**
    * Treinamento do modelo de melhor desempenho (Random Forest) utilizando o conjunto de treino completo (treino + validação).
    * Geração de previsões para o conjunto de teste.
9.  **📄 Formatação para Submissão:** Preparação do arquivo `random_forest_predictions_walmart.csv` com as previsões finais.

---

## 💻 Tecnologias e Bibliotecas

* **Linguagem:** Python 3.x
* **Bibliotecas Centrais:**
    * `pandas`: Para manipulação e análise eficiente de dados tabulares.
    * `numpy`: Para operações numéricas e suporte a arrays.
    * `scikit-learn`: Para pré-processamento, implementação dos modelos de machine learning e cálculo das métricas de avaliação.
    * `matplotlib` e `seaborn`: Para a criação de visualizações e gráficos.
* **Ambiente de Desenvolvimento:** Google Colaboratory (Colab) e VS Code