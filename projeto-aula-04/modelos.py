"""Um classificador de cancelamento, com tuning no conjunto de treino."""

from pathlib import Path

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLUNAS = ["meses_de_casa", "valor_mensal", "entregas_atrasadas", "plano"]


def carregar_dados():
    return pd.read_csv(Path(__file__).parent / "dados" / "clientes.csv")


def preparar_entradas(clientes):
    # Criar as mesmas colunas para treino, teste e lista de clientes.
    planos = pd.Categorical(clientes["plano"], categories=["Degustação", "Clássico", "Premium"])
    entradas = clientes[COLUNAS[:-1]].copy()
    return pd.concat([entradas, pd.get_dummies(planos, prefix="plano", dtype=int)], axis=1)


def treinar(clientes, tentativas=15):
    X = preparar_entradas(clientes)
    y = clientes["cancelou"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    dobras = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def avaliar(trial):
        c = trial.suggest_float("C", 0.01, 10.0, log=True)
        peso = trial.suggest_categorical("class_weight", [None, "balanced"])
        modelo = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=peso))
        return cross_val_score(modelo, X_treino, y_treino, cv=dobras, scoring="f1").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(avaliar, n_trials=tentativas)

    modelo = make_pipeline(StandardScaler(), LogisticRegression(**estudo.best_params))
    modelo.fit(X_treino, y_treino)
    relatorio = classification_report(y_teste, modelo.predict(X_teste), zero_division=0)
    return modelo, estudo.best_params, estudo.best_value, relatorio


def calcular_risco(modelo, clientes):
    return modelo.predict_proba(preparar_entradas(clientes))[:, 1]
