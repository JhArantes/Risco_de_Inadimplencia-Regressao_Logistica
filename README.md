# 📊 Previsão de Inadimplência com Regressão Logística

Projeto de classificação binária para prever se um cliente será **inadimplente** com base em características financeiras e comportamentais, utilizando Regressão Logística com balanceamento via SMOTE.

---

## 📁 Estrutura do Projeto

```
├── inadimplencia.ipynb       # Notebook principal com toda a análise
├── Exercício Logística.csv   # Dataset utilizado
└── README.md
```

---

## 🎯 Objetivo

Construir um modelo de Machine Learning capaz de identificar clientes com risco de inadimplência a partir de variáveis como renda, dívidas, pontuação de crédito e comportamento de compras online.

---

## 🗂️ Dataset

O dataset contém informações financeiras e demográficas de clientes. Após a limpeza, foram utilizadas **235 amostras**.

### Variáveis

| Coluna                | Descrição                                  | Tipo       |
|-----------------------|--------------------------------------------|------------|
| `idade`               | Idade do cliente (anos)                    | Numérica   |
| `tempo_emprego`       | Tempo de emprego atual (anos)              | Numérica   |
| `salario`             | Salário mensal (R$)                        | Numérica   |
| `dividas`             | Total de dívidas (R$)                      | Numérica   |
| `pontuacao_credito`   | Score de crédito                           | Numérica   |
| `qtde_compras_online` | Quantidade de compras online               | Numérica   |
| `inadimplente`        | **Variável alvo** (0 = Não, 1 = Sim)       | Binária    |

### Estatísticas Descritivas

| Estatística | Idade  | Tempo Emprego | Salário     | Dívidas      | Pontuação Crédito | Compras Online |
|-------------|--------|---------------|-------------|--------------|-------------------|----------------|
| Média       | 31.73  | 3.78 anos     | R$ 4.914,14 | R$ 7.857,00  | 151.63            | 9.18           |
| Desvio Pad. | 8.25   | 2.62          | R$ 2.804,52 | R$ 5.994,35  | 107.85            | 6.67           |
| Mín         | 18     | 0             | R$ 16,00    | R$ 46,99     | 1                 | 0              |
| Máx         | 65     | 13.5          | R$ 11.659,00| R$ 33.516,11 | 578               | 33             |

### Distribuição da Variável Alvo

| Classe | Descrição       | Qtde |
|--------|-----------------|------|
| 0      | Não inadimplente | 152  |
| 1      | Inadimplente     | 83   |

> Dataset desbalanceado (~35% inadimplentes), tratado com SMOTE.

---

## 🔧 Pré-processamento

- **Filtragem de idades**: remoção de registros com idade < 18 anos
- **Filtragem de salários**: remoção de salários negativos
- **Remoção de nulos**: `dropna()` para garantir integridade dos dados
- **Padronização**: `StandardScaler` aplicado no conjunto de treino (e transformado no teste)

---

## ⚖️ Balanceamento de Classes

Aplicado **SMOTE (Synthetic Minority Over-sampling Technique)** exclusivamente no conjunto de treino, resultando em 76 amostras por classe após o balanceamento.

---

## 🤖 Modelagem

**Algoritmo:** Regressão Logística (`sklearn.linear_model.LogisticRegression`)

**Divisão treino/teste:** 70% / 30% (Holdout, `random_state=42`)

### Comparação de Solvers

| Solver       | Acurácia Treino | Acurácia Teste |
|--------------|-----------------|----------------|
| `lbfgs`      | 87.50%          | **85.59%**     |
| `saga`       | 87.50%          | **85.59%**     |
| `newton-cg`  | 87.50%          | **85.59%**     |
| `sag`        | 87.50%          | **85.59%**     |
| `liblinear`  | 87.50%          | 84.75%         |

🏆 **Melhor solver:** `lbfgs`

---

## 📈 Resultados Finais

### Relatório de Classificação

| Classe       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (Não)      | 0.97      | 0.71   | 0.82     | 49      |
| 1 (Sim)      | 0.60      | 0.95   | 0.74     | 22      |
| **Accuracy** |           |        | **0.79** | 71      |
| Macro Avg    | 0.79      | 0.83   | 0.78     | 71      |
| Weighted Avg | 0.86      | 0.79   | 0.80     | 71      |

> O modelo apresenta **alto recall para inadimplentes (95%)**, o que é desejável em contextos de risco de crédito, minimizando falsos negativos.

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas scikit-learn imbalanced-learn numpy
```

### Execução

1. Clone o repositório:
```bash
git clone https://github.com/JhArantes/Risco_de_Inadimplencia-Regressao_Logistica
cd seu-repositorio
```

2. Coloque o arquivo `Exercício Logística.csv` na raiz do projeto.

3. Abra e execute o notebook:
```bash
jupyter notebook inadimplencia.ipynb
```

Ou acesse diretamente pelo [Google Colab](https://colab.research.google.com/).

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Pandas** — manipulação e limpeza de dados
- **Scikit-learn** — modelagem, pré-processamento e métricas
- **Imbalanced-learn (SMOTE)** — balanceamento de classes
- **Google Colab** — ambiente de execução

---

## 👤 Autor

**João Henrique Arantes Vieira**

---

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
