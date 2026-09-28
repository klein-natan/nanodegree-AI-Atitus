"""Modelagem do sistema de risco de cancelamento do Clube do Café.

Ordem do arquivo, que é a ordem do trabalho:
dados -> separação -> preparo -> modelo -> baselines -> tuning -> teste
-> explicabilidade -> previsões.
"""

import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ALVO = "cancelou"
COLUNAS_NUMERICAS = [
    "meses_de_casa",
    "valor_mensal",
    "entregas_atrasadas",
    "chamados_suporte",
    "avaliacao_media",
    "idade",
]
COLUNAS_ASSIMETRICAS = ["dias_sem_acessar"]
COLUNAS_CATEGORICAS = ["plano", "forma_pagamento"]
COLUNAS_BINARIAS = ["entrou_com_cupom"]
ENTRADAS = COLUNAS_NUMERICAS + COLUNAS_ASSIMETRICAS + COLUNAS_CATEGORICAS + COLUNAS_BINARIAS

SEMENTE = 42
PASTA = Path(__file__).parent


# ---------------------------------------------------------------- dados
def carregar_dados():
    return pd.read_csv(PASTA / "dados" / "clientes.csv")


def separar(clientes):
    # O teste fica guardado até o fim: nenhuma escolha pode olhar para ele.
    X = clientes[ENTRADAS]
    y = clientes[ALVO]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMENTE)


# -------------------------------------------------------------- preparo
def montar_preparo():
    # Cada tipo de coluna recebe o seu próprio tratamento.
    numericas = Pipeline([
        # add_indicator cria a coluna "avaliação estava faltando": não saber
        # a nota do cliente também é informação.
        ("preencher", SimpleImputer(strategy="median", add_indicator=True)),
        ("padronizar", StandardScaler()),
    ])
    assimetricas = Pipeline([
        ("preencher", SimpleImputer(strategy="median")),
        # log(1 + dias): 3 dias e 90 dias deixam de estar a 87 unidades de
        # distância, e os poucos clientes sumidos não dominam a escala.
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("padronizar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("preencher", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numericas", numericas, COLUNAS_NUMERICAS),
        ("assimetricas", assimetricas, COLUNAS_ASSIMETRICAS),
        ("categoricas", categoricas, COLUNAS_CATEGORICAS),
        ("binarias", "passthrough", COLUNAS_BINARIAS),
    ])


# --------------------------------------------------------------- modelo
def montar_modelo(parametros=None):
    # SGDClassifier com loss="log_loss" é uma regressão logística treinada
    # por gradiente descendente estocástico: um cliente de cada vez.
    if parametros is None:
        parametros = {}
    classificador = SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        tol=1e-4,
        random_state=SEMENTE,
        **parametros,
    )
    return Pipeline([("preparo", montar_preparo()), ("classificador", classificador)])


def criar_dobras():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)


# ------------------------------------------------------------ baselines
def avaliar_baselines(X_treino, y_treino):
    # O modelo só vale a pena se ganhar de quem não aprende nada.
    candidatos = {
        "Sempre 'fica'": DummyClassifier(strategy="most_frequent"),
        "Sorteio na proporção": DummyClassifier(strategy="stratified", random_state=SEMENTE),
        "SGD sem tuning": montar_modelo(),
    }
    linhas = []
    for nome, modelo in candidatos.items():
        f1 = cross_val_score(modelo, X_treino, y_treino, cv=criar_dobras(), scoring="f1")
        acuracia = cross_val_score(modelo, X_treino, y_treino, cv=criar_dobras(), scoring="accuracy")
        linhas.append({"modelo": nome, "acuracia_cv": acuracia.mean(), "f1_cv": f1.mean()})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------- tuning
def sugerir_parametros(trial):
    parametros = {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    # l1_ratio só existe quando a penalidade mistura L1 e L2.
    if parametros["penalty"] == "elasticnet":
        parametros["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
    return parametros


def tunar(X_treino, y_treino, tentativas=40):
    dobras = criar_dobras()

    def avaliar(trial):
        modelo = montar_modelo(sugerir_parametros(trial))
        return cross_val_score(modelo, X_treino, y_treino, cv=dobras, scoring="f1").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMENTE)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estudo.optimize(avaliar, n_trials=tentativas)
    return estudo


def historico_do_estudo(estudo):
    linhas = []
    melhor_ate_agora = -1.0
    for trial in estudo.trials:
        melhor_ate_agora = max(melhor_ate_agora, trial.value)
        linha = {"tentativa": trial.number + 1, "f1_cv": trial.value, "melhor_ate_agora": melhor_ate_agora}
        linha.update(trial.params)
        linhas.append(linha)
    return pd.DataFrame(linhas)


# ------------------------------------------------------------ avaliação
def calcular_metricas(y_real, risco, limiar=0.5):
    previsto = (risco >= limiar).astype(int)
    return {
        "acuracia": accuracy_score(y_real, previsto),
        "precisao": precision_score(y_real, previsto, zero_division=0),
        "recall": recall_score(y_real, previsto, zero_division=0),
        "f1": f1_score(y_real, previsto, zero_division=0),
        "auc": roc_auc_score(y_real, risco),
    }


def matriz_de_confusao(y_real, risco, limiar=0.5):
    previsto = (risco >= limiar).astype(int)
    matriz = confusion_matrix(y_real, previsto, labels=[0, 1])
    return pd.DataFrame(
        matriz,
        index=["Real: ficou", "Real: cancelou"],
        columns=["Previsto: fica", "Previsto: cancela"],
    )


# ---------------------------------------------------- explicabilidade
def importancia_por_permutacao(modelo, X, y, repeticoes=20):
    # Embaralha uma coluna por vez e mede quanto a AUC cai. Coluna que o
    # modelo usa de verdade faz a AUC despencar; coluna inútil, não.
    resultado = permutation_importance(
        modelo, X, y, scoring="roc_auc", n_repeats=repeticoes, random_state=SEMENTE
    )
    tabela = pd.DataFrame({
        "variavel": X.columns,
        "queda_auc": resultado.importances_mean,
        "desvio": resultado.importances_std,
    })
    return tabela.sort_values("queda_auc", ascending=False).reset_index(drop=True)


def pesos_do_modelo(modelo):
    # Os w de cada coluna depois do preparo (colunas já padronizadas).
    nomes = modelo.named_steps["preparo"].get_feature_names_out()
    pesos = modelo.named_steps["classificador"].coef_[0]
    tabela = pd.DataFrame({"coluna": nomes, "peso": pesos})
    tabela["coluna"] = tabela["coluna"].str.split("__").str[-1]
    return tabela.sort_values("peso", ascending=False).reset_index(drop=True)


def contribuicoes(modelo, cliente):
    # Num modelo linear, z = w0 + soma(w * x). Cada parcela w * x mostra
    # quanto aquela coluna empurrou o risco deste cliente para cima ou baixo.
    preparo = modelo.named_steps["preparo"]
    classificador = modelo.named_steps["classificador"]
    valores = preparo.transform(cliente[ENTRADAS])[0]
    tabela = pd.DataFrame({
        "coluna": preparo.get_feature_names_out(),
        "contribuicao": valores * classificador.coef_[0],
    })
    tabela["coluna"] = tabela["coluna"].str.split("__").str[-1]
    tabela = tabela[tabela["contribuicao"].abs() > 1e-6]
    return tabela.sort_values("contribuicao", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------ previsões
def prever_risco(modelo, clientes):
    return modelo.predict_proba(clientes[ENTRADAS])[:, 1]


def lucro_por_limiar(y_real, risco, valor_mensal, custo_contato, taxa_sucesso, meses_retidos):
    # Para cada limiar: quem entra na lista recebe contato (custa sempre);
    # se a pessoa ia mesmo cancelar, a oferta segura uma parte delas.
    linhas = []
    for limiar in np.arange(0.05, 0.96, 0.05):
        na_lista = risco >= limiar
        salvaveis = na_lista & (np.asarray(y_real) == 1)
        receita = (np.asarray(valor_mensal)[salvaveis] * meses_retidos * taxa_sucesso).sum()
        custo = na_lista.sum() * custo_contato
        linhas.append({
            "limiar": round(float(limiar), 2),
            "contatos": int(na_lista.sum()),
            "cancelariam": int(salvaveis.sum()),
            "lucro": receita - custo,
        })
    return pd.DataFrame(linhas)


# -------------------------------------------------- tudo de uma vez
def treinar_sistema(clientes, tentativas=40):
    X_treino, X_teste, y_treino, y_teste = separar(clientes)
    baselines = avaliar_baselines(X_treino, y_treino)
    estudo = tunar(X_treino, y_treino, tentativas)

    parametros = sugerir_parametros(optuna.trial.FixedTrial(estudo.best_params))
    modelo = montar_modelo(parametros)
    # Risco de cada cliente do treino dado por um modelo que não o viu:
    # é com ele que se escolhe o limiar, sem tocar no teste.
    risco_validacao = cross_val_predict(
        modelo, X_treino, y_treino, cv=criar_dobras(), method="predict_proba"
    )[:, 1]
    modelo.fit(X_treino, y_treino)

    risco_teste = modelo.predict_proba(X_teste)[:, 1]
    referencia = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
    return {
        "modelo": modelo,
        "parametros": parametros,
        "f1_cv": estudo.best_value,
        "historico": historico_do_estudo(estudo),
        "baselines": baselines,
        "metricas_teste": calcular_metricas(y_teste, risco_teste),
        "metricas_baseline_teste": calcular_metricas(y_teste, referencia.predict_proba(X_teste)[:, 1]),
        "X_treino": X_treino,
        "y_treino": y_treino,
        "risco_validacao": risco_validacao,
        "X_teste": X_teste,
        "y_teste": y_teste,
        "risco_teste": risco_teste,
        "importancia": importancia_por_permutacao(modelo, X_teste, y_teste),
        "pesos": pesos_do_modelo(modelo),
    }
