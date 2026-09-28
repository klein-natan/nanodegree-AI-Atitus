# Aula 4 — Sistema de risco de cancelamento do Clube do Café

Este projeto é um **exemplo completo** de um sistema de Machine Learning
pequeno, do arquivo de dados até um painel que a equipe usaria. Em aula,
construímos o mesmo sistema passo a passo no notebook do Colab. Aqui ele
aparece organizado como um projeto de verdade, em arquivos separados.

O Clube do Café tem 2.000 assinantes, e 583 deles cancelaram (29,1%). O
sistema estima o risco de cada assinante cancelar e monta a lista de
clientes que a equipe deve contatar.

## Rodar

Abra a pasta `projeto-aula-04` no VS Code. No terminal:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
python treinar.py                  # treina e grava artefatos/sistema.joblib
streamlit run app.py               # abre o painel no navegador
```

No macOS ou Linux, ative com `source .venv/bin/activate`. Se você pular o
`treinar.py`, o painel treina sozinho na primeira vez (leva uns 20 segundos).

## Os arquivos

| Arquivo | Papel no sistema |
|---|---|
| `dados/clientes.csv` | 2.000 assinantes e a resposta conhecida (`cancelou`) |
| `modelos.py` | toda a modelagem: preparo, baselines, tuning, avaliação, explicabilidade |
| `treinar.py` | roda o treino e grava o modelo em `artefatos/` |
| `app.py` | o painel: carrega o modelo gravado e mostra tudo |

Treinar e servir ficam separados de propósito: o treino roda de vez em
quando, e o painel só carrega o que foi gravado.

## As colunas

| Coluna | O que é |
|---|---|
| `meses_de_casa` | há quantos meses a pessoa assina |
| `plano` | `Degustação`, `Clássico` ou `Premium` |
| `valor_mensal` | quanto paga por mês, em reais |
| `forma_pagamento` | `Cartão`, `Pix` ou `Boleto` |
| `entrou_com_cupom` | 1 se assinou usando cupom de desconto |
| `entregas_atrasadas` | entregas atrasadas nos últimos 6 meses |
| `chamados_suporte` | chamados abertos no suporte nos últimos 3 meses |
| `avaliacao_media` | nota média dada aos cafés, de 1 a 5 (vazia para quem nunca avaliou) |
| `dias_sem_acessar` | dias desde o último acesso ao app |
| `idade` | idade em anos |

## O que ler em `modelos.py`, de cima para baixo

1. **Separação.** `train_test_split(..., stratify=y)` guarda 25% para o
   teste final. Nenhuma escolha olha para esse teste.
2. **Preparo com `ColumnTransformer`.** Cada tipo de coluna tem o seu
   próprio `Pipeline`:
   - numéricas: preenche faltas com a mediana, cria a coluna "avaliação
     faltando" (`add_indicator=True`) e padroniza;
   - `dias_sem_acessar`: aplica `log(1 + dias)` antes de padronizar, porque
     poucos clientes sumidos há meses esticariam a escala;
   - categóricas: `OneHotEncoder`, que ignora categorias novas;
   - `entrou_com_cupom`: já é 0 ou 1, passa direto.
3. **Modelo.** `SGDClassifier(loss="log_loss")` é uma regressão logística
   treinada por gradiente descendente estocástico, um cliente por vez.
4. **Baselines com `DummyClassifier`.** "Sempre fica" acerta 71% sem
   aprender nada e tem F1 zero. O modelo precisa ganhar dele.
5. **Tuning com Optuna.** Ajusta `alpha` (força da regularização),
   `penalty` (L2, L1 ou elastic net), `l1_ratio` (só quando a penalidade é
   elastic net) e `class_weight`. Cada tentativa vale o F1 médio em cinco
   dobras estratificadas do treino. Como o preparo está dentro do
   `Pipeline`, cada dobra aprende médias e medianas só com a sua parte.
6. **Teste final.** O modelo vencedor treina com o treino inteiro e é
   medido uma única vez no teste.
7. **Explicabilidade.** A importância por permutação embaralha uma coluna
   do teste por vez e mede quanto a AUC cai. Os pesos `w` do modelo e as
   contribuições `w · x` de cada cliente completam o quadro.

## O painel

| Aba | O que mostra |
|---|---|
| Visão geral | a taxa de cancelamento por grupo de clientes |
| Modelo | baselines contra o modelo, o histórico do Optuna, a matriz de confusão e as métricas no limiar escolhido |
| Explicabilidade | a importância por permutação e os pesos do modelo |
| Lista de contato | o lucro de cada limiar, dado o custo de um contato, e a lista para baixar |
| Simulador | "e se" para um cliente, e para a operação inteira |

Na barra lateral ficam o limiar e os números da economia da retenção, que
valem para todas as abas.
