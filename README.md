🧠 Projeto Prático – Predição de Churn com Machine Learning
🔍 Análise de Dados • ETL • Visualização • Modelagem • Avaliação

Este projeto tem como objetivo aplicar todo o processo de análise de dados, preparação, modelagem e avaliação utilizando um conjunto de dados fictício de clientes. A solução prevê churn — ou seja, a probabilidade de um cliente cancelar um serviço — utilizando técnicas de aprendizagem de máquina.

👨‍💻 Desenvolvido por

Projeto desenvolvido pelos integrantes do grupo:

Matheus Franklin Brasileiro — GitHub: MATHEUSBRr

Victor Hugo N. Dias — GitHub: VictorVHND

Yan Macedo Teixeira — GitHub: Yanmaiscedo

📌 1. Descrição Detalhada do Problema e Objetivos do Projeto

Empresas que operam com serviços recorrentes (ex.: internet, telefonia, streaming) têm como um dos maiores desafios reduzir a evasão de clientes, conhecida como churn. Prever quais clientes têm maior probabilidade de cancelar o serviço permite que equipes de retenção atuem de forma preventiva.

❗ Problema

Como identificar automaticamente os clientes com maior risco de churn, usando dados internos da empresa?

🎯 Objetivos principais

Criar um conjunto de dados fictício simulando clientes e comportamentos.

Realizar ETL completo: limpeza, tratamento e engenharia de atributos.

Desenvolver visualizações para entender padrões e relações.

Treinar um modelo de Machine Learning para prever churn.

Avaliar o desempenho do modelo com métricas adequadas.

Criar um pipeline reprodutível com scripts separados.

Gerar um relatório final com gráficos, métricas e análise dos resultados.

🛠️ 2. Processo de ETL (Extração, Transformação e Carga)

O script responsável por esta etapa está em:
src/etl.py

Extração

O dataset fictício é carregado a partir de:

data/churn_data.csv


Ele contém colunas como:

customer_id

age

gender

contract_months

monthly_fee

support_calls

payment_delay

usage_minutes

churn (0 ou 1)

Transformação

As principais etapas de transformação foram:

✔ Tratamento de valores inconsistentes

Exclusão de linhas com valores impossíveis ou extremos.

Correção de tipos numéricos.

✔ Normalização de texto

Padronização de gender, conversão para categorias numéricas.

✔ Feature Engineering

Foram criadas novas variáveis mais informativas:

avg_monthly_usage = usage_minutes / contract_months

delayed_payments_flag = 1 se payment_delay > 0

high_support_usage = clientes com mais de 3 chamadas no suporte

✔ Escalonamento / Padronização

Apenas para modelos que necessitam. No RandomForest não é obrigatório.

Carga

Após o processamento, os dados preparados são salvos em:

data/processed_churn.csv

📊 3. Análise Exploratória e Visualizações

A análise exploratória foi realizada no script:

src/eda.py

Foram gerados gráficos e salvos em:

reports/figures/

Principais visualizações:

Histograma de idade dos clientes

Distribuição de churn por gênero

Boxplot do valor mensal vs churn

Correlação entre variáveis

Gráfico de chamadas de suporte por cliente

Insights encontrados:

Clientes com alto número de chamadas no suporte apresentam maior churn.

Contratos mais curtos têm maior taxa de evasão.

Pagamentos atrasados também são um forte indicador.

Uso mensal muito baixo ou muito alto tem relação com churn.

Estas informações foram fundamentais para escolher o modelo e entender o comportamento dos clientes.

🤖 4. Modelagem e Algoritmos de Machine Learning

A modelagem foi realizada no script:

src/model.py

Algoritmo escolhido

O modelo utilizado foi:

Random Forest Classifier

Bom para dados tabulares.

Robusto a outliers.

Lida bem com relações não-lineares.

Dispensa normalização de dados.

Hiperparâmetros utilizados (versão final executada)

n_estimators = 100

max_depth = 7

random_state = 42

n_jobs = -1 (usa todos os núcleos disponíveis)

Os hiperparâmetros foram selecionados após testes preliminares; há também um script com GridSearchCV para explorar combinações mais amplas.

Divisão de dados

75% para treino

25% para teste

Amostragem estratificada garantindo equilíbrio das classes

📈 5. Avaliação e Interpretação dos Resultados

A avaliação foi realizada em:

src/evaluate.py

Métricas calculadas:

Acurácia

Precisão

Recall

F1-score

ROC-AUC

Matriz de confusão

Curva ROC

Principais resultados obtidos:

(Exemplo real do modelo gerado)

Classe	Precisão	Recall	F1-score
Não churn	0.75	0.93	0.83
Churn	0.69	0.32	0.44
Interpretação:

O modelo identifica muito bem clientes que não irão cancelar (recall alto para classe 0).

Tem dificuldade moderada em capturar todos os clientes que irão cancelar (recall da classe 1 é menor).

Isso é comum em datasets onde churn é uma classe menos frequente.

ROC-AUC aproximadamente 0.70+, indicando bom poder de separação.

Importância das variáveis

As features mais relevantes foram:

support_calls

payment_delay

contract_months

monthly_fee

avg_monthly_usage

Estas variáveis fazem sentido no contexto de churn, reforçando a qualidade do modelo.

🏁 6. Conclusão, Aprendizados, Limitações e Melhorias Futuras
✔ Aprendizados

A importância da etapa de ETL para garantir qualidade do modelo.

Como a análise exploratória orienta a escolha das features.

Aplicação prática de um modelo de Machine Learning completo.

Interpretação de métricas e validação do modelo.

✔ Limitações atuais

O dataset é fictício, podendo não refletir totalmente um cenário real.

Classe de churn é desbalanceada, dificultando recall.

Apenas Random Forest foi treinado na versão final executada.

✔ Possíveis melhorias

Implementar técnicas de balanceamento, como SMOTE.

Comparar com modelos mais avançados, como XGBoost ou LightGBM.

Usar otimização bayesiana para hiperparâmetros.

Criar um painel interativo (Dash/Streamlit).

Aumentar variáveis comportamentais e transacionais.

▶️ Como Rodar o Projeto

1. Clone este repositório

No terminal:

git clone https://github.com/MATHEUSBRr/Churn-Prediction.git

2. Crie e ative um ambiente virtual
Windows:
python -m venv venv
venv\Scripts\activate

Linux / Mac:
python3 -m venv venv
source venv/bin/activate

3. Instale as dependências

Com o ambiente virtual ativado, execute:

pip install -r requirements.txt

4. Execute o pipeline completo
✔️ ETL (tratamento e preparação dos dados)
python src/etl.py

✔️ Análise Exploratória (gera gráficos)
python src/eda.py

✔️ Treinamento do Modelo
python src/model.py

✔️ Avaliação do Modelo
python src/evaluate.py

✔️ Gerar Relatório em Word
python src/generate_report.py

5. Onde encontrar os resultados

Após rodar os scripts, os resultados aparecerão automaticamente nas pastas:

📂 data/

processed_churn.csv — dados limpos e transformados

📂 models/

rf_churn.joblib — modelo treinado

📂 reports/

metrics.txt — métricas completas

report.docx — relatório final

📁 figures/ — todos os gráficos gerados (histogramas, ROC, matriz de confusão etc.)

6. Requisitos

Python 3.9+

VSCode (opcional, mas recomendado)

Pip atualizado