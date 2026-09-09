# Aula 6 — O Painel do Clube do Café

Este arquivo é a aula. Não existem slides: a gente constrói o sistema
lendo daqui, de cima para baixo, e no fim ele roda na sua máquina.

O negócio é o **Clube do Café**, uma assinatura mensal de café em grãos.
Ele tem duas perguntas, e você já sabe responder as duas:

| Pergunta | Modelo | De onde veio |
|---|---|---|
| Quais assinantes correm risco de cancelar? | Regressão logística | Aula 3 |
| Quanto o clube vai faturar no próximo mês? | Prophet | Aula 5 |

O que muda hoje é o entorno. As duas respostas vão morar na mesma tela, e
a tela vai rodar no seu computador.

![A aba de risco do painel](../figuras/tela_risco.png)

![A aba de previsão do painel](../figuras/tela_previsao.png)

---

## Índice

1. [O modelo é uma peça, não o sistema](#1-o-modelo-é-uma-peça-não-o-sistema)
2. [Deixar a máquina pronta](#2-deixar-a-máquina-pronta)
3. [Escrever o `modelos.py`](#3-escrever-o-modelospy)
4. [Checkpoint: testar sem tela](#4-checkpoint-testar-sem-tela)
5. [Escrever o `app.py`](#5-escrever-o-apppy)
6. [Rodar o painel](#6-rodar-o-painel)
7. [Cinco desafios](#7-cinco-desafios)
8. [O que este sistema ainda não é](#8-o-que-este-sistema-ainda-não-é)
9. [Quando alguma coisa não funciona](#9-quando-alguma-coisa-não-funciona)

---

## 1. O modelo é uma peça, não o sistema

![As quatro caixas do sistema](../figuras/arquitetura.png)

Repare que o modelo ocupa uma caixa das quatro. Nos projetos reais essa
proporção se mantém: a maior parte do trabalho está em levar dados até o
modelo, e em levar a resposta do modelo até quem decide.

O sistema tem quatro arquivos, e cada um faz uma coisa só.

```
sistema/
├── dados/
│   ├── clientes.csv   400 assinantes, com a coluna cancelou
│   └── vendas.csv     731 dias de receita
├── modelos.py         carrega, treina e prevê. Não sabe nada de tela
├── app.py             desenha a tela. Não sabe nada de modelo
└── requirements.txt   as quatro bibliotecas
```

Essa separação não é frescura. Quando a previsão sai errada, você sabe em
que arquivo procurar. Quando a tela precisa de um botão novo, você mexe em
um arquivo só.

> **Pergunta para a turma.** Por que separar `modelos.py` de `app.py`, se
> tudo caberia num arquivo só?

---

## 2. Deixar a máquina pronta

Você precisa de **Python 3.10 ou mais novo** e do **VS Code**. Confira a
versão do Python com `python --version` (no macOS e no Linux, às vezes é
`python3 --version`).

**Passo 1.** Abra esta pasta no VS Code: menu `File` → `Open Folder` →
escolha a pasta `sistema`. Abra o terminal integrado com `Ctrl + '` (no
Mac, `Control + '`).

**Passo 2.** Crie o ambiente virtual, a caixa de bibliotecas deste
projeto.

No Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
```

No macOS ou Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Deu certo quando aparece `(.venv)` no começo da linha do terminal. **Não
siga sem ver isso.**

**Passo 3.** Instale as bibliotecas. Leva de dois a cinco minutos na
primeira vez, e o `prophet` é o mais demorado.

```bash
pip install -r requirements.txt
```

> **O erro número um da aula.** Rodar `pip install` sem o ambiente ativo,
> e depois o `streamlit` responder `command not found`. O sintoma é
> sempre esse, e a causa quase sempre é a mesma: a linha do terminal não
> começa com `(.venv)`. Ative o ambiente e repita.

---

## 3. Escrever o `modelos.py`

Este arquivo não sabe nada de tela: ele só carrega dados, treina e prevê.
São oito funções curtas, e nenhuma delas é novidade.

### 3.1 O cabeçalho

```python
from pathlib import Path

import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LogisticRegression

# Caminho da pasta deste arquivo. Usar isso em vez de "dados/clientes.csv"
# faz o sistema funcionar de qualquer pasta em que você abrir o terminal.
PASTA = Path(__file__).parent

# As informações que o classificador usa para calcular o risco.
COLUNAS_DO_CLIENTE = ["meses_de_casa", "valor_mensal", "entregas_atrasadas", "plano"]
```

O `PASTA` resolve um problema chato: se alguém abrir a pasta de cima no VS
Code, um caminho relativo simples quebraria.

### 3.2 Carregar os dados

```python
def carregar_clientes():
    return pd.read_csv(PASTA / "dados" / "clientes.csv")


def carregar_vendas():
    return pd.read_csv(PASTA / "dados" / "vendas.csv", parse_dates=["data"])
```

O `parse_dates` faz o pandas entender que `data` é data de verdade, e não
texto. É o que permite ordenar e extrair o dia da semana.

### 3.3 O classificador de risco

```python
def preparar_tabela(clientes):
    # O plano é um nome, não um número: vira colunas de 0 e 1
    tabela = pd.get_dummies(clientes[COLUNAS_DO_CLIENTE], columns=["plano"],
                            drop_first=True)
    return tabela.astype(float)


def treinar_classificador(clientes):
    tabela = preparar_tabela(clientes)
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(tabela, clientes["cancelou"])
    return modelo


def calcular_risco(modelo, clientes):
    # A coluna 1 do predict_proba é a probabilidade de cancelar
    tabela = preparar_tabela(clientes)
    return modelo.predict_proba(tabela)[:, 1]
```

`get_dummies` é da Aula 2, `LogisticRegression` é da Aula 3, e
`predict_proba` devolve probabilidade, não decisão. Nada aqui é novo: o
que é novo é ter isso guardado com nome, num arquivo.

### 3.4 A função do simulador, e a linha que evita o bug clássico

```python
def risco_de_um_cliente(modelo, meses_de_casa, valor_mensal, entregas_atrasadas, plano):
    cliente = pd.DataFrame({
        "meses_de_casa": [meses_de_casa],
        "valor_mensal": [valor_mensal],
        "entregas_atrasadas": [entregas_atrasadas],
        "plano": [plano],
    })
    tabela = pd.get_dummies(cliente, columns=["plano"])
    # Põe as colunas na mesma ordem do treino, com 0 no que faltar
    tabela = tabela.reindex(columns=modelo.feature_names_in_, fill_value=0)
    return float(modelo.predict_proba(tabela.astype(float))[0][1])
```

Esta é a única armadilha técnica da aula, e vale parar nela.

Quando a tela manda **um cliente só**, o `get_dummies` cria coluna apenas
para o plano daquele cliente. Faltam as outras, e a ordem muda. O
`reindex` põe as colunas na mesma ordem do treino e preenche com zero o
que faltar. O `feature_names_in_` é o próprio scikit-learn lembrando com
que colunas ele foi treinado.

Sem essa linha, o modelo recebe as colunas trocadas e devolve um número
sem sentido, **sem dar erro nenhum**. Esse é o tipo de bug que passa
despercebido.

### 3.5 O previsor de receita

```python
def treinar_previsor(vendas):
    # O Prophet exige as colunas com os nomes ds e y
    serie = vendas.rename(columns={"data": "ds", "receita": "y"})
    feriados = pd.DataFrame({
        "holiday": "feriado",
        "ds": vendas[vendas["feriado"] == 1]["data"],
    })
    modelo = Prophet(holidays=feriados)
    modelo.fit(serie[["ds", "y"]])
    return modelo


def prever(modelo, dias):
    futuro = modelo.make_future_dataframe(periods=dias)
    return modelo.predict(futuro)
```

É a Aula 5 inteira em duas funções. A lista de feriados continua sendo o
que o modelo não adivinha sozinho.

---

## 4. Checkpoint: testar sem tela

Antes de montar a tela, teste a peça. Quem for depurar modelo dentro do
navegador vai sofrer.

No terminal, com o ambiente ativo:

```bash
python -c "import modelos; print(modelos.carregar_clientes().shape)"
```

Deve responder `(400, 6)`. Depois teste o classificador inteiro:

```bash
python -c "import modelos; c = modelos.carregar_clientes(); m = modelos.treinar_classificador(c); print(modelos.risco_de_um_cliente(m, 4, 129.0, 3, 'Premium'))"
```

Deve responder algo perto de `0.89`: um cliente novo, caro e com três
entregas atrasadas tem risco alto.

> **Pergunta para a turma.** Esse número mudaria se você tirasse a linha
> do `reindex`? Por quê?

---

## 5. Escrever o `app.py`

Este arquivo não sabe nada de modelo: ele chama as funções do
`modelos.py` e desenha o resultado. Usa **seis comandos** do Streamlit.

| Comando | O que faz |
|---|---|
| `st.title` e `st.subheader` | escrevem títulos |
| `st.tabs` | cria as abas |
| `st.slider` e `st.selectbox` | pedem um número e uma opção |
| `st.metric` | mostra um número grande, com rótulo |
| `st.dataframe` | mostra uma tabela |
| `st.line_chart` | desenha um gráfico de linha |

### 5.1 O cabeçalho e o cache

```python
import pandas as pd
import streamlit as st

import modelos

st.set_page_config(page_title="Painel do Clube do Café", layout="wide")


def reais(valor):
    # Formata 12345.6 como "R$ 12.346"
    return "R$ " + f"{valor:,.0f}".replace(",", ".")


# O @st.cache_resource faz isso rodar uma vez só, e não a cada clique.
@st.cache_resource
def preparar_sistema():
    clientes = modelos.carregar_clientes()
    vendas = modelos.carregar_vendas()
    classificador = modelos.treinar_classificador(clientes)
    previsor = modelos.treinar_previsor(vendas)
    return clientes, vendas, classificador, previsor


clientes, vendas, classificador, previsor = preparar_sistema()

st.title("Painel do Clube do Café")
st.caption("Duas perguntas do mesmo negócio, respondidas por dois modelos.")

aba_risco, aba_receita = st.tabs(["Risco de cancelamento", "Previsão de receita"])
```

**O Streamlit funciona de um jeito diferente do que você talvez espere.**
Não existe botão para programar, nem evento para tratar. Toda vez que
alguém mexe num controle, ele executa **o arquivo inteiro de novo**, de
cima para baixo, com o valor novo.

É isso que faz a tela reagir sem você escrever nada de interface. E é isso
que torna o `@st.cache_resource` necessário: sem ele, os dois modelos
seriam treinados a cada clique.

> **Experimento em aula.** Comente a linha do `@st.cache_resource`, mexa
> no controle do limiar e conte os segundos. Descomente e repita.

### 5.2 A primeira aba: a lista de risco

```python
with aba_risco:
    st.subheader("A lista de risco de hoje")

    limiar = st.slider("A partir de que risco entrar na lista", 0.10, 0.90, 0.50, 0.05)

    # Trabalhamos numa cópia para não bagunçar a tabela original
    tabela = clientes.copy()
    tabela["risco"] = modelos.calcular_risco(classificador, clientes)
    em_risco = tabela[tabela["risco"] >= limiar]
    em_risco = em_risco.sort_values("risco", ascending=False)

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric("Clientes na base", len(tabela))
    coluna2.metric("Na lista de risco", len(em_risco))
    coluna3.metric("Receita mensal em risco", reais(em_risco["valor_mensal"].sum()))

    # Uma cópia só para mostrar na tela, com o risco em porcentagem
    mostrar = em_risco.head(20).copy()
    mostrar["risco"] = (mostrar["risco"] * 100).round().astype(int).astype(str) + "%"
    mostrar = mostrar[["id_cliente", "plano", "meses_de_casa", "valor_mensal",
                       "entregas_atrasadas", "risco"]]
    mostrar.columns = ["Cliente", "Plano", "Meses de casa", "Valor mensal",
                       "Entregas atrasadas", "Risco"]

    st.dataframe(mostrar, hide_index=True, use_container_width=True)
    st.caption("Os 20 clientes de maior risco. Baixe o limiar para a lista crescer.")
```

O controle do limiar é a Aula 3 virando operação. Mexa nele e veja:

| Limiar | Clientes na lista | Receita mensal em risco | Da lista, cancelaram de fato |
|---|---|---|---|
| 0,40 | 168 | R$ 13.265 | 61% |
| 0,50 | 119 | R$ 9.748 | 67% |

Suba o limiar e a lista encolhe, mas fica mais certeira. Desça e você
alcança mais gente que ia cancelar, ao custo de ligar para quem ia ficar.

Por isso o limiar mora **na tela**, e não escondido no código: quem decide
o tamanho da operação é a pessoa que vai fazer as ligações.

### 5.3 Ainda na primeira aba: o simulador

```python
    st.divider()
    st.subheader("Simule um cliente")

    coluna1, coluna2, coluna3, coluna4 = st.columns(4)
    meses = coluna1.slider("Meses de casa", 1, 48, 6)
    valor = coluna2.slider("Valor mensal (R$)", 30, 200, 79)
    atrasos = coluna3.slider("Entregas atrasadas", 0, 6, 2)
    plano = coluna4.selectbox("Plano", ["Degustação", "Clássico", "Premium"])

    risco = modelos.risco_de_um_cliente(classificador, meses, valor, atrasos, plano)

    st.metric("Risco de cancelar", f"{risco:.0%}")
    if risco >= limiar:
        st.warning("Este cliente entraria na lista de risco de hoje.")
    else:
        st.success("Este cliente ficaria fora da lista de hoje.")
```

Repare que o `if` compara com o **mesmo** `limiar` do controle lá de cima.
Como o arquivo roda inteiro a cada clique, a variável já está lá.

### 5.4 A segunda aba: a receita que vem

```python
with aba_receita:
    st.subheader("Quanto o clube vai faturar")

    dias = st.slider("Quantos dias prever", 7, 90, 30, 7)

    previsao = modelos.prever(previsor, dias)
    futuro = previsao.tail(dias)

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric(f"Receita prevista em {dias} dias", reais(futuro["yhat"].sum()))
    coluna2.metric("Média por dia", reais(futuro["yhat"].mean()))
    coluna3.metric("Pior caso da faixa", reais(futuro["yhat_lower"].sum()))

    # Um gráfico com as duas coisas: o que já aconteceu e o que vem
    grafico = pd.DataFrame({
        "Receita realizada": vendas.set_index("data")["receita"].tail(120),
        "Previsão": futuro.set_index("ds")["yhat"],
    })
    st.line_chart(grafico)

    st.caption("Os últimos 120 dias de histórico e a previsão à frente.")

    st.divider()
    st.subheader("As peças que o modelo enxergou")
    st.pyplot(previsor.plot_components(previsao))
```

Com 30 dias, o painel mostra R$ 89.853 no total, R$ 2.995 por dia em
média, e R$ 84.475 no pior caso da faixa. O terceiro número é o que
interessa para quem vai assumir um compromisso de compra.

![A previsão que o painel mostra](../figuras/previsao_receita.png)

---

## 6. Rodar o painel

```bash
streamlit run app.py
```

O navegador abre sozinho em `http://localhost:8501`. Para parar, volte ao
terminal e aperte `Ctrl + C`.

Toda vez que você salvar um arquivo, o Streamlit oferece recarregar a
página. Clique em `Rerun` e veja a mudança na hora.

---

## 7. Cinco desafios

Agora é com você. Mexa no seu próprio sistema.

1. **Mude o texto.** Troque o título e a legenda do painel pelo nome de um
   negócio que você conhece.
2. **Mostre mais gente.** Faça a tabela listar 30 clientes em vez de 20.
3. **Some o plano na conta.** No simulador, mostre também qual seria o
   risco do mesmo cliente se ele estivesse no plano Premium.
4. **Mude o padrão.** Deixe o limiar começar em 0,40 em vez de 0,50, e
   explique em uma frase por que isso muda a operação.
5. **Invente uma métrica.** Acrescente um `st.metric` com um número que
   você acha que falta no painel. Justifique a escolha para a turma.

> **Dica para o desafio 3.** Você já tem a função pronta: basta chamar
> `modelos.risco_de_um_cliente` de novo, trocando o último argumento por
> `"Premium"`.

---

## 8. O que este sistema ainda não é

O painel funciona, mas ele não está pronto para uma empresa depender dele.
Vale saber a diferença.

| O que falta | Por quê |
|---|---|
| Retreinar sozinho | os modelos aprenderam com os dados de hoje. Daqui a três meses, o mundo mudou |
| Guardar o que previu | sem registro, você nunca descobre se as previsões estavam certas |
| Monitorar a entrada | se o CSV chegar com uma coluna a menos, o painel quebra na cara do usuário |
| Rodar em algum lugar | hoje ele roda na sua máquina. Para a equipe usar, precisa de um servidor |
| Controlar quem vê | a lista tem número de cliente e valor de contrato |

Nenhum desses itens é difícil sozinho. Juntos, eles são a diferença entre
um protótipo e um produto, e é neles que mora a maior parte do trabalho de
quem faz sistemas de dados.

Os três primeiros passos depois desta aula seriam: um script de retreino
que roda toda semana, um registro das previsões com data, e a publicação
do painel num servidor com senha. O
[Streamlit Community Cloud](https://streamlit.io/cloud) resolve o terceiro
de graça, a partir de um repositório do GitHub.

---

## 9. Quando alguma coisa não funciona

| Sintoma | O que fazer |
|---|---|
| `streamlit: command not found` | O ambiente virtual não está ativo. Repita o passo 2 e confira se aparece `(.venv)`. |
| `ModuleNotFoundError: No module named 'prophet'` | O `pip install` não terminou. Rode o passo 3 de novo e leia a última linha da saída. |
| A instalação do `prophet` falha | Atualize o instalador antes: `pip install --upgrade pip setuptools wheel` e tente de novo. |
| `Port 8501 is already in use` | Já existe um painel rodando. Feche a outra janela do terminal, ou use `streamlit run app.py --server.port 8502`. |
| A página abre em branco | Volte ao terminal: o erro em vermelho está lá, com o número da linha. |
| `FileNotFoundError: dados/clientes.csv` | Você abriu uma pasta diferente no VS Code. Abra a pasta `sistema`, não a pasta de cima. |
| O painel está lento a cada clique | Falta o `@st.cache_resource` em cima da função `preparar_sistema`. |

Se travar de vez, a pasta deste repositório já tem os dois arquivos
prontos: compare o seu com o que está aqui, linha por linha.
