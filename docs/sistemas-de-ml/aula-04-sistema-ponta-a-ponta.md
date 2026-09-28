---
description: Um sistema de risco de cancelamento, do dado bruto à decisão
---

# Aula 4 — Um Sistema de Ponta a Ponta

{% hint style="info" %}
**O que você leva desta aula**

Você vai transformar a regressão logística da Aula 3 num sistema completo:
um baseline para comparar, um preparo de dados com `ColumnTransformer`, o
modelo treinado por gradiente descendente estocástico, o tuning com Optuna,
a explicação do modelo e a escolha do limiar pelo custo do negócio. Em aula,
construímos tudo no notebook. O projeto `projeto-aula-04/` mostra o mesmo
sistema organizado em arquivos, com um painel.
{% endhint %}

## Para que serve

O Clube do Café é uma assinatura mensal de café em grãos. Tem 2.000
assinantes, e 583 deles cancelaram (29,1%). A equipe pode ligar para um
cliente e oferecer um desconto, mas cada contato custa dinheiro. A pergunta
do negócio é: **para quem ligar esta semana?**

O sistema dá a cada cliente um risco de cancelar e transforma esse risco
numa lista de contato. Um modelo sozinho não faz isso. É preciso um ciclo
inteiro, e cada etapa pode estragar as outras:

| Etapa | Pergunta |
|---|---|
| Separação | com que dados treinamos, e com quais tiramos a prova final? |
| Baseline | quanto acerta quem não aprende nada? |
| Preparo | como transformar a tabela em números limpos? |
| Modelo | como a regressão logística aprende? |
| Tuning | quais ajustes funcionam melhor? |
| Teste final | o sistema funciona com clientes que nunca viu? |
| Explicabilidade | em que o modelo se apoia? |
| Decisão | a partir de que risco vale ligar? |

## Os dados e a separação

Cada assinante tem dez colunas: tempo de casa, plano, valor mensal, forma
de pagamento, se entrou com cupom, entregas atrasadas, chamados ao suporte,
avaliação média dos cafés, dias sem acessar o app e idade. A coluna
`cancelou` é a resposta.

Três coisas chamam atenção logo de cara. A avaliação está vazia para 194
clientes, que nunca deram nota. As escalas são muito diferentes: o valor
mensal vai de 30 a 150, e o cupom só vale 0 ou 1. E os dias sem acessar
têm cauda longa, com média de 12 dias e alguns clientes sumidos há mais de
90.

Antes de qualquer outra coisa, 25% dos clientes vão para o **teste**, com
`stratify=y` para manter a proporção de cancelamentos. O teste fica
trancado até o fim. Toda comparação entre modelos usa **validação cruzada
estratificada** de cinco dobras nos outros 1.500 clientes.

## O baseline

Um `DummyClassifier` que responde sempre "fica" acerta 70,9% dos clientes
sem aprender nada. O F1 dele é zero, porque ele nunca encontra um
cancelamento. É a régua: o sistema precisa de F1 bem acima de zero, e a
acurácia sozinha não serve para julgar.

## O preparo com ColumnTransformer

Cada tipo de coluna tem seu problema e sua ferramenta, encadeados num
`Pipeline`. O `ColumnTransformer` manda cada grupo de colunas para a sua
linha de montagem e junta tudo no fim.

| Colunas | Tratamento |
|---|---|
| numéricas | preenche faltas com a mediana, marca quem estava sem nota, padroniza |
| dias sem acessar | preenche, aplica o logaritmo, padroniza |
| plano e pagamento | uma coluna de 0 e 1 para cada categoria (one-hot) |
| cupom | já é 0 ou 1, passa direto |

A padronização mede cada valor em desvios padrão a partir da média:

$$
x_{\text{padronizado}} = \frac{x - \text{média}}{\text{desvio padrão}}
$$

O valor mensal tem média 78,6 e desvio padrão 33,9. Quem paga R$ 129 vira
(129 − 78,6) / 33,9 = 1,49.

O logaritmo encurta a cauda longa:

$$
x_{\text{novo}} = \log(1 + x)
$$

Com 3 dias sem acessar, o resultado é 1,39; com 90 dias, é 4,51. A
distância entre os dois cai de 87 para 3,1, e os poucos clientes sumidos
deixam de dominar a escala.

O preparo aprende médias e medianas. Por isso ele fica **dentro** do mesmo
`Pipeline` que o modelo: em cada dobra da validação cruzada, ele aprende só
com a parte de treino daquela dobra, e nada do que vai ser avaliado vaza
para o treino.

## O modelo: regressão logística por SGD

O modelo é a mesma regressão logística da Aula 3. A diferença está em como
ela aprende. O `SGDClassifier` com `loss="log_loss"` usa o **gradiente
descendente estocástico**: em vez de olhar os 1.500 clientes antes de cada
passo, como o gradiente descendente da Aula 1, ele dá um passo a cada
cliente. Para cada peso, o passo é:

$$
w_j \leftarrow w_j - \eta \cdot (p_i - y_i) \cdot x_{ij}
$$

O η é a taxa de aprendizado, `p_i` é o risco que o modelo deu ao cliente,
`y_i` é o que aconteceu (1 se cancelou) e `x_ij` é o valor da coluna para
ele. Com peso 0,5, um cliente que cancelou, risco dado de 0,3, coluna
valendo 2 e η de 0,1, o peso vai para 0,5 − 0,1 · (0,3 − 1) · 2 = 0,64. O
peso sobe, e o risco desse cliente sobe junto.

Sem nenhum ajuste, o SGD já chega a F1 de 0,49 na validação cruzada.

## Tuning com Optuna

Os pesos `w` o modelo aprende sozinho. Os **hiperparâmetros** são escolhas
feitas antes do treino. O Optuna ajusta quatro:

| Hiperparâmetro | O que controla |
|---|---|
| `alpha` | a força da regularização, uma multa por pesos grandes |
| `penalty` | o tipo de multa: L2 encolhe todos os pesos, L1 zera os inúteis, elastic net mistura as duas |
| `l1_ratio` | a proporção de L1 na mistura (só existe com elastic net) |
| `class_weight` | com `"balanced"`, errar num cliente que cancelou pesa mais |

Com a multa L2, o custo que o modelo minimiza é:

$$
\text{custo} = \text{perda} + \alpha \sum_j w_j^2
$$

Com `alpha` de 0,01 e dois pesos, 2 e −1, a multa soma 0,01 · (4 + 1) =
0,05. Atenção ao nome: no scikit-learn, `alpha` é a força da multa, não a
taxa de aprendizado.

```python
def avaliar(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    penalidade = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    peso_das_classes = trial.suggest_categorical("class_weight", [None, "balanced"])
    proporcao_l1 = 0.15
    if penalidade == "elasticnet":
        proporcao_l1 = trial.suggest_float("l1_ratio", 0.05, 0.95)
    classificador = SGDClassifier(loss="log_loss", alpha=alpha, penalty=penalidade,
                                  l1_ratio=proporcao_l1, class_weight=peso_das_classes,
                                  max_iter=2000, tol=1e-4, random_state=42)
    modelo = Pipeline([("preparo", preparo), ("classificador", classificador)])
    return cross_val_score(modelo, X_treino, y_treino, cv=dobras, scoring="f1").mean()
```

Em 40 tentativas, o melhor F1 médio nas dobras sobe para 0,58, com elastic
net e `class_weight="balanced"`. O teste não participa dessa escolha.

## O teste final

O modelo vencedor treina com o treino inteiro e é medido uma única vez nos
500 clientes do teste: F1 de 0,64 e AUC de 0,81. Dos 146 clientes do teste
que cancelaram, o modelo aponta 111. O baseline, lembre, apontava zero.

## Explicabilidade

A **importância por permutação** embaralha uma coluna do teste por vez e
mede quanto a AUC cai. Se o modelo depende da coluna, embaralhar destrói a
informação e a AUC despenca. Se não depende, nada muda.

| Coluna | Queda na AUC |
|---|---|
| meses de casa | 0,127 |
| dias sem acessar | 0,045 |
| entregas atrasadas | 0,040 |
| avaliação média | 0,039 |
| chamados ao suporte | 0,018 |
| idade | 0,001 |

O tempo de casa é o que mais pesa. A idade quase não ajuda e poderia sair
do sistema. E importância não é causa: ela mostra no que o modelo se
apoia, não o que faz o cliente sair.

## Do risco à decisão

O modelo dá o risco; o negócio decide o **limiar**. Suponha que cada
contato custe R$ 10, que 30% de quem ia cancelar aceite a oferta, e que
cada cliente salvo pague mais 4 meses:

$$
\text{lucro} = 0{,}30 \cdot 4 \cdot (\text{mensalidades de quem ia cancelar e foi contatado}) - 10 \cdot \text{contatos}
$$

Escolher o limiar também é um ajuste, então ele sai da validação cruzada,
e não do teste. Com esses custos, o melhor limiar é 0,30, e ele rende 23% a
mais que o 0,50. Se o contato custasse R$ 60, o melhor limiar subiria para
0,75, e o 0,50 passaria a dar prejuízo. É o mesmo modelo com os mesmos
clientes, mas a decisão muda.

## Simulação de cenários

Com o modelo pronto, dá para perguntar "e se". O cliente de maior risco no
teste tem 94% de chance estimada de cancelar. Se as entregas dele não
tivessem atrasado, o modelo daria 83%. Aplicando a mesma mudança à base
inteira, 258 clientes sairiam da lista de risco.

O simulador mostra o que o *modelo* prevê. O modelo aprendeu associações,
e só um teste de verdade com clientes mostra o efeito real de uma ação.

## O projeto

O `projeto-aula-04/` organiza o mesmo sistema em arquivos:

| Arquivo | Papel |
|---|---|
| `modelos.py` | preparo, baselines, tuning, avaliação e explicabilidade |
| `treinar.py` | treina e grava o modelo em `artefatos/` |
| `app.py` | painel Streamlit com visão geral, modelo, explicabilidade, lista de contato e simulador |

Treinar e servir ficam separados: o treino grava o `Pipeline` inteiro, com
o preparo junto, e o painel só carrega o que foi gravado.

```bash
cd projeto-aula-04
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
python treinar.py
streamlit run app.py
```

No macOS ou Linux, ative com `source .venv/bin/activate`.

O [roteiro do projeto](https://github.com/klein-natan/nanodegree-AI-Atitus/blob/main/projeto-aula-04/README.md)
explica cada arquivo. O [notebook da aula](https://colab.research.google.com/github/klein-natan/nanodegree-AI-Atitus/blob/main/notebooks/aula-04-sistema-ponta-a-ponta.ipynb)
constrói o sistema passo a passo.

## Para ir além

- [ColumnTransformer no scikit-learn](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
- [SGDClassifier e o gradiente descendente estocástico](https://scikit-learn.org/stable/modules/sgd.html)
- [Optuna: tutorial de primeira otimização](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)
- [Importância por permutação](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Validação cruzada no scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
