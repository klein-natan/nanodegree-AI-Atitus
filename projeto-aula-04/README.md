# Aula 4 — Sistema de classificação do Clube do Café

Nesta aula, você implementa **um classificador de regressão logística**
para priorizar clientes com risco de cancelar. Dos 400 clientes, 146
cancelaram. O desbalanceamento é moderado e permite comparar a acurácia
com uma métrica que considera a classe de interesse: o F1.

## 1. Preparar o ambiente

Abra a pasta `projeto-aula-04` no VS Code. No terminal:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

No macOS ou Linux, ative com `source .venv/bin/activate`.

## 2. Entender os três arquivos

- `dados/clientes.csv`: exemplos e resposta conhecida (`cancelou`).
- `modelos.py`: prepara os dados, busca hiperparâmetros e treina.
- `app.py`: mostra as probabilidades e deixa você mudar o limiar.

Comece lendo `modelos.py`, de cima para baixo. `preparar_entradas` transforma
o plano em colunas de 0 e 1. `train_test_split(..., stratify=y)` mantém a
proporção das classes no treino e no teste. O teste só é usado **depois**
da busca.

O Optuna ajusta `C` (regularização) e `class_weight` (peso da classe
minoritária). Para cada tentativa, `cross_val_score` calcula o F1 médio
em cinco dobras estratificadas. A padronização fica dentro do `Pipeline`
para ser aprendida de novo em cada dobra, sem ver os dados de validação.

Depois da busca, o modelo vencedor aprende com todo o treino. O
`classification_report` mede seu desempenho no teste separado.

## 3. Rodar e explorar

```bash
streamlit run app.py
```

Observe o F1 da validação cruzada e o relatório do teste. Mova o limiar:
quantos clientes entram na lista em 0,30 e em 0,50? A lista demonstra a
decisão operacional; use o teste separado para julgar o desempenho.

## 4. Sua implementação

1. No notebook da aula, complete a preparação dos dados e a função de
   avaliação do Optuna.
2. Rode a busca com `n_trials=15` e anote `best_params` e `best_value`.
3. Ajuste o modelo escolhido no treino e compare precisão, recall e F1
   no teste.
4. Altere o limiar do painel e explique a mudança no tamanho da lista.

Se alguma biblioteca faltar, confirme que o ambiente está ativo e rode
`pip install -r requirements.txt` novamente.
