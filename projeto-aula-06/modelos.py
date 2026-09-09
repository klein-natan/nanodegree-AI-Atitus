"""Os dois modelos do Clube do Café.

Este arquivo não sabe nada de tela: ele só carrega dados, treina e prevê.
Quem desenha o painel é o app.py. Separar assim deixa cada arquivo pequeno
o bastante para caber na cabeça.
"""

from pathlib import Path

import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LogisticRegression

# Caminho da pasta deste arquivo. Usar isso em vez de "dados/clientes.csv"
# faz o sistema funcionar de qualquer pasta em que você abrir o terminal.
PASTA = Path(__file__).parent

# As informações que o classificador usa para calcular o risco.
COLUNAS_DO_CLIENTE = ["meses_de_casa", "valor_mensal", "entregas_atrasadas", "plano"]


# --- Carregar os dados ----------------------------------------------------

def carregar_clientes():
    return pd.read_csv(PASTA / "dados" / "clientes.csv")


def carregar_vendas():
    return pd.read_csv(PASTA / "dados" / "vendas.csv", parse_dates=["data"])


# --- Modelo 1: quem corre risco de cancelar -------------------------------

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


# --- Modelo 2: quanto o clube vai faturar ---------------------------------

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
