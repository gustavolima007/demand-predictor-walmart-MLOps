# 🛒 Walmart Demand Predictor MLOps: Previsão de Demanda para Categorias Chave no Varejo Alimentício 🚀

📊 **Projeto Final da Disciplina CET0621 - Aprendizado de Máquina na Análise de Dados**  
*Unicamp - Faculdade de Tecnologia*  
Este projeto desenvolve um sistema de previsão de vendas semanais por categoria (departamento) e loja na rede Walmart, utilizando um pipeline completo de ciência de dados. O objetivo é gerar previsões precisas para otimizar estoques e estratégias no varejo alimentício, seguindo boas práticas de MLOps e reprodutibilidade.

---

## 📖 Descrição do Projeto

Este trabalho implementa um modelo de aprendizado de máquina baseado em Random Forest para prever vendas semanais, integrando dados históricos e contextuais em um pipeline estruturado. Desde a limpeza e engenharia de features até a avaliação e otimização do modelo, o projeto busca oferecer uma solução prática para o desafio de previsão de demanda, reduzindo perdas por excesso ou falta de estoque no varejo alimentício.

---

## 🎯 Problema Abordado

A previsão precisa da demanda é essencial no varejo alimentício, onde erros podem levar a desperdícios (excesso de estoque) ou perda de vendas e insatisfação do cliente (falta de produtos). Este projeto aborda esse desafio aplicando técnicas de regressão para estimar vendas semanais por departamento, visando melhorar a eficiência operacional e a rentabilidade da rede Walmart.

---

## 💾 Dataset Utilizado

O projeto utiliza o dataset "Walmart Recruiting - Store Sales Forecasting", disponível na plataforma Kaggle. Este conjunto contém:
- **Vendas semanais**: Detalhadas por loja e departamento (proxy para categorias de produtos).
- **Características das lojas**: Tipo (A, B, C) e tamanho.
- **Fatores contextuais**: Temperatura, preço do combustível, descontos promocionais (MarkDown1-5), Índice de Preços ao Consumidor (CPI), taxa de desemprego, e indicadores de feriados (IsHoliday).
- **Período**: Treino (fevereiro de 2010 a outubro de 2012) e teste (novembro de 2012 a julho de 2013).

🔗 **[Fonte do Dataset](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)**

---

## 🛠️ Metodologia e Workflow

O desenvolvimento seguiu um fluxo de trabalho estruturado baseado no processo de Descoberta de Conhecimento em Dados (KDD), adaptado para MLOps:

1. **📥 Carregamento e Integração de Dados**  
   - Combinação dos arquivos CSV (`train.csv`, `test.csv`, `stores.csv`, `features.csv`) em dataframes unificados.

2. **🧹 Limpeza e Validação**  
   - Validação de tipos de dados (datas, numéricos).
   - Correção de valores inconsistentes (ex.: vendas e markdowns negativos ajustados para 0).
   - Ausência de duplicatas significativas detectada.

3. **🩹 Tratamento de Valores Ausentes (NaNs)**  
   - Preenchimento de `MarkDown1-5` com 0 (ausência de promoções).
   - Uso de preenchimento progressivo (`ffill`) e regressivo (`bfill`) agrupado por loja para `CPI` e `Unemployment`, preservando a consistência temporal.

4. **🔄 Transformação de Variáveis**  
   - One-Hot Encoding para `Type` (gerando `Type_A`, `Type_B`, `Type_C`).
   - Conversão de `IsHoliday` para formato numérico (0/1).

5. **✨ Engenharia de Features**  
   - **Temporais**: Extração de `Year`, `Month`, `Day`, `WeekOfYear`, `DayOfWeek`, `DayOfYear`.
   - **Lags**: Features baseadas em `Weekly_Sales` de semanas anteriores (ex.: lags de 1, 4, 12, 52 semanas) para capturar autocorrelação e sazonalidade.
   - **Janelas Móveis**: Estatísticas (média, mediana, soma, desvio padrão) sobre janelas de 4, 8, 12, 26 e 52 semanas para identificar tendências.
   - Tratamento de NaNs resultantes com preenchimento por 0.

6. **📊 Divisão dos Dados**  
   - Separação cronológica em treino (2010-02-05 a 2012-07-06), validação (2012-07-13 a 2012-10-26) e teste (novembro de 2012 a julho de 2013), simulando cenários reais.

7. **🤖 Modelagem e Avaliação**  
   - Modelos testados: Regressão Linear, Árvore de Decisão, Random Forest, Gradient Boosting.
   - Métricas de avaliação: MAE, RMSE e R² no conjunto de validação.
   - Random Forest otimizado com `n_estimators=30`, `max_depth=7`, etc., via RandomizedSearchCV.

8. **🚀 Treinamento Final e Previsões**  
   - Retrainamento com conjunto completo e geração de previsões para o teste.
   - Exportação do arquivo `random_forest_predictions_walmart.csv`.

---

## 📈 Resultados

O modelo Random Forest otimizado alcançou R² de 0.9833, MAE de 1397.38 e RMSE de 2825.27 no conjunto de validação, superando os baselines. A validação cruzada de 5-fold apresentou RMSE de 5938.36 (±910.41), indicando robustez. Features como `Weekly_Sales_lag_1` e `Weekly_Sales_roll_mean_4` foram as mais influentes, capturando padrões temporais. Apesar da alta precisão, uma skewness de 1.66 nos resíduos sugere leve subestimação em vendas altas, apontando áreas para refinamento.

---

## 💻 Tecnologias e Bibliotecas

- **Linguagem**: Python 3.13.4
- **Bibliotecas**:
  - `pandas`: Manipulação de dados tabulares.
  - `numpy`: Operações numéricas.
  - `scikit-learn`: Modelagem e métricas.
  - `matplotlib` e `seaborn`: Visualizações.
- **Ambiente**: Google Colab e VS Code.

---

## 📋 Instruções de Uso

1. **Pré-requisitos**:
   - Instale as dependências: `pip install -r requirements.txt` (crie um `requirements.txt` com as bibliotecas listadas).
   - Faça o clone do repositorio.

2. **Execução**:
   - Execute o notebook principal (`Previsão_de_Demanda_para_Categorias_Chave_de_Produtos_no_Varejo_Alimentício.ipynb`) no Colab/VS Code.
   - Ajuste os caminhos dos arquivos no código, se necessário.

3. **Saída**:
   - Previsões salvas em `random_forest_predictions_walmart.csv`.