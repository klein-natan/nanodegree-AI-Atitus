"""Treina o sistema e salva o artefato que o painel usa.

Treinar e servir são etapas separadas num sistema de ML: o treino roda de
vez em quando e grava o modelo; o painel só carrega o que foi gravado.

Uso:

    python treinar.py
"""

import time

import joblib

import modelos

ARQUIVO_ARTEFATO = modelos.PASTA / "artefatos" / "sistema.joblib"


def treinar_e_salvar(tentativas=40):
    clientes = modelos.carregar_dados()
    sistema = modelos.treinar_sistema(clientes, tentativas)
    sistema["treinado_em"] = time.strftime("%d/%m/%Y %H:%M")
    sistema["tentativas"] = tentativas
    ARQUIVO_ARTEFATO.parent.mkdir(exist_ok=True)
    joblib.dump(sistema, ARQUIVO_ARTEFATO)
    return sistema


if __name__ == "__main__":
    print("Treinando: baselines, Optuna e avaliação no teste...")
    sistema = treinar_e_salvar()
    print(f"Melhores hiperparâmetros: {sistema['parametros']}")
    print(f"F1 médio nas dobras (treino): {sistema['f1_cv']:.3f}")
    teste = sistema["metricas_teste"]
    print(f"No teste separado: F1 {teste['f1']:.3f} | AUC {teste['auc']:.3f}")
    print(f"Artefato salvo em {ARQUIVO_ARTEFATO}")
