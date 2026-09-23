---
description: Um classificador de cancelamento, da validação cruzada ao painel
---

# Aula 4 — Um Sistema de Ponta a Ponta

{% hint style="info" %}
**O que você leva desta aula**

Você vai transformar a regressão logística da Aula 3 em um sistema pequeno:
um classificador, uma busca de hiperparâmetros com Optuna e uma tela para
usar as probabilidades. O código completo está em `projeto-aula-04/`.
{% endhint %}

## Para que serve

O Clube do Café quer priorizar os clientes com maior risco de cancelar.
Temos 400 assinantes e 146 cancelamentos (36,5%). As classes são
**moderadamente desbalanceadas**: se o sistema disser que ninguém cancela,
acerta 63,5% e não ajuda a equipe. Por isso avaliamos precisão, recall e F1,
além da acurácia.

O sistema responde uma pergunta: **quais clientes entram na lista de
contato?** Ele usa apenas regressão logística, o classificador da aula
anterior.

## Do dado à decisão

O arquivo `clientes.csv` traz tempo de casa, valor mensal, entregas
atrasadas e plano. A coluna `cancelou` é a resposta conhecida. O plano
vira três colunas de 0 e 1; as entradas numéricas são padronizadas.

Separamos 25% dos clientes para o teste final, mantendo a proporção de
cancelamentos com `stratify=y`. Nos outros 75%, fazemos validação cruzada
estratificada de cinco dobras: cada configuração treina cinco vezes e
recebe a média do F1. A padronização está dentro do `Pipeline`, então
cada dobra aprende suas médias somente com a parte usada para treino.

## Tuning com Optuna

**Hiperparâmetros** são escolhas feitas antes do treino. Vamos ajustar
dois: `C`, que controla a força da regularização, e `class_weight`, que
pode dar mais peso à classe com menos exemplos. O Optuna experimenta 15
combinações e guarda a de melhor F1 médio nas dobras.

```python
def avaliar(trial):
    c = trial.suggest_float("C", 0.01, 10.0, log=True)
    peso = trial.suggest_categorical("class_weight", [None, "balanced"])
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=peso))
    return cross_val_score(modelo, X_treino, y_treino, cv=dobras, scoring="f1").mean()

estudo = optuna.create_study(direction="maximize")
estudo.optimize(avaliar, n_trials=15)
```

O teste separado não participa dessa escolha. Depois do tuning,
treinamos a melhor configuração com todo o conjunto de treino e medimos
o resultado uma vez no teste. Esse relatório dá uma estimativa mais
honesta de como o classificador pode funcionar com clientes novos.

## O painel

O `app.py` mostra o F1 da validação cruzada, os melhores hiperparâmetros
e o relatório do teste. Um controle de limiar transforma as probabilidades
em lista de contato: ao baixar o limiar, a lista cresce. Essa lista inclui
os clientes do arquivo usado na aula e serve para explorar a decisão; o
resultado do teste separado é a medida de desempenho.

```bash
cd projeto-aula-04
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

No macOS ou Linux, ative com `source .venv/bin/activate`.

O [roteiro do projeto](https://github.com/klein-natan/nanodegree-AI-Atitus/blob/main/projeto-aula-04/README.md) mostra a implementação
passo a passo. O [notebook da aula](https://colab.research.google.com/github/klein-natan/nanodegree-AI-Atitus/blob/main/notebooks/aula-04-sistema-ponta-a-ponta.ipynb)
permite experimentar o tuning antes de abrir o VS Code.

## Para ir além

- [Optuna: tutorial de primeira otimização](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)
- [Validação cruzada no scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Pipeline no scikit-learn](https://scikit-learn.org/stable/modules/compose.html#pipeline)
