---
description: Um sistema de risco de cancelamento, do dado bruto à decisão
---

# Aula 4 — Um Sistema de Ponta a Ponta

{% hint style="info" %}
**O que você leva desta aula**

Um modelo sozinho não resolve o problema de ninguém. Nesta aula, você
constrói um sistema inteiro em volta da regressão logística da Aula 3:
separar os dados, criar uma régua, preparar a tabela, treinar, ajustar,
testar, explicar e decidir. Em aula, fazemos tudo no notebook. O projeto
`projeto-aula-04/` mostra o mesmo sistema organizado em arquivos, com um
painel.
{% endhint %}

## Para que serve

O Clube do Café é uma assinatura mensal de café em grãos, com 2.000
assinantes. Todo mês, alguns cancelam: 29% já cancelaram. A equipe pode
ligar para um cliente e oferecer um desconto, mas cada ligação custa
dinheiro. A pergunta da empresa é:

> **Para quem vale a pena ligar esta semana?**

O sistema dá a cada cliente uma nota de risco, de 0 a 1, e transforma
essa nota numa lista de ligações.

## Os dados

Cada assinante tem dez informações: tempo de casa, plano, mensalidade,
forma de pagamento, se entrou com cupom, entregas atrasadas, reclamações
ao suporte, nota média dada aos cafés, dias sem entrar no app e idade. A
coluna `cancelou` é a resposta.

Dois problemas aparecem logo. 194 clientes nunca deram nota, então a
avaliação deles está vazia. E as colunas vivem em réguas muito
diferentes: a mensalidade vai de 30 a 150, e o cupom só vale 0 ou 1.

## A prova final fica trancada

Um bom professor não entrega a prova final como lista de exercícios. Com
o modelo é igual: 25% dos clientes vão para uma gaveta trancada, o
**teste**, que só é aberto no fim.

Para comparar modelos antes disso, usamos **provas simuladas**: o treino
é dividido em 5 pedaços, e cada pedaço serve de prova uma vez. A nota do
modelo é a média das 5 provas. É a **validação cruzada**.

## O baseline: a régua

O primeiro "modelo" é preguiçoso: responde sempre "fica", sem olhar para
o cliente. Ele acerta 71% das vezes e não encontra nenhum cancelamento.
O F1 dele é zero.

Essa é a lição: quando uma resposta é bem mais comum que a outra, a
acurácia engana. O sistema é julgado pelo F1.

## O preparo: uma receita para cada coluna

Na cozinha, cada ingrediente tem seu preparo. Com os dados é igual:

| Problema | Onde | Solução |
|---|---|---|
| células vazias | avaliação | preencher com a mediana e marcar quem não tinha nota |
| réguas diferentes | colunas de números | padronizar |
| texto em vez de número | plano, forma de pagamento | uma coluna de 0 e 1 por opção |
| poucos valores enormes | dias sem acessar | encolher com o logaritmo |

Padronizar põe todas as colunas na mesma régua:

$$
\text{valor padronizado} = \frac{\text{valor} - \text{média}}{\text{desvio padrão}}
$$

A mensalidade média é R$ 78,60, com desvio padrão de R$ 33,90. Quem paga
R$ 129 vira 1,5: um pouco acima do normal.

Cada receita é um `Pipeline`, uma lista de passos feitos em ordem. O
`ColumnTransformer` manda cada grupo de colunas para a sua receita e junta
tudo numa tabela só. O preparo fica na mesma peça que o modelo, para
aprender médias e medianas só com o treino, sem espiar a prova final.

## O modelo aprende um cliente de cada vez

Na regressão logística, cada característica soma ou tira pontos de risco,
e a sigmoide transforma o total num número entre 0 e 1. O que muda aqui é
o jeito de aprender os pesos.

O **SGD** (gradiente descendente estocástico) é um aprendiz que olha os
clientes um por um, em ordem sorteada. A cada palpite errado, ele ajusta
um pouquinho os pesos. Na Aula 1, o gradiente descendente olhava todos os
exemplos antes de cada ajuste; o SGD ajusta a cada exemplo.

Ele aprende rápido. Depois de ver 10 clientes, a AUC é 0,53, quase um
chute. Depois de 50, passa de 0,75. Depois de 300, está em 0,78, e cada
cliente novo ensina pouca coisa nova.

Sem nenhum ajuste, o modelo chega a F1 de 0,49 nas provas simuladas. O
baseline tinha zero.

## O ajuste fino com Optuna

Os pesos o modelo aprende sozinho. Os **hiperparâmetros** são escolhas
feitas antes do treino. Ajustamos dois:

- **a força do freio** (`alpha`): impede o modelo de decorar os clientes
  do treino. Freio de menos, ele decora; freio demais, fica simples demais;
- **os pesos das classes** (`class_weight`): com `"balanced"`, cada cliente
  que cancelou conta como 1,7 cliente, e o modelo presta mais atenção em
  quem é minoria.

O Optuna procura a melhor combinação como quem acerta o sal de uma
receita: prova, ajusta, prova de novo. A nota de cada tentativa é o F1
nas provas simuladas.

```python
def avaliar(tentativa):
    forca = tentativa.suggest_float("forca", 0.00001, 0.1, log=True)
    pesos = tentativa.suggest_categorical("pesos", [None, "balanced"])
    modelo = montar_modelo(forca, pesos)
    return cross_val_score(modelo, X_treino, y_treino, cv=dobras, scoring="f1").mean()
```

Em 30 tentativas, o F1 sobe de 0,49 para 0,58, com `"balanced"`.

## A prova final

O modelo ajustado treina com o treino inteiro e faz a prova uma única
vez. Dos 146 clientes do teste que cancelaram, ele encontra 113. O F1 é
0,62 e a AUC é 0,82, perto do que as provas simuladas prometiam: o modelo
aprendeu de verdade, não decorou.

## Em que o modelo presta atenção

Para saber quanto um jogador importa, tire ele de campo e veja quanto o
time piora. Aqui, tirar uma coluna é **embaralhá-la**: cada cliente recebe
o valor de outra pessoa. Quanto mais a AUC cai, mais importante era a
coluna. É a **importância por permutação**.

| Coluna | Queda na AUC |
|---|---|
| meses de casa | 0,125 |
| dias sem acessar | 0,048 |
| entregas atrasadas | 0,039 |
| avaliação média | 0,036 |
| idade | 0,000 |

O tempo de casa é o que mais pesa. A idade não ajuda em nada e pode sair
do sistema. E importância não é causa: ela mostra em que o *modelo*
presta atenção, não o que faz o cliente sair.

## A decisão: para quem ligar

O modelo dá o risco. Quem decide a partir de que risco vale ligar é a
empresa. Esse corte é o **limiar**.

Suponha que cada ligação custe R$ 10, que 30% de quem ia cancelar aceite
o desconto, e que cada cliente salvo pague mais 4 meses. Ligar para 200
clientes custa R$ 2.000. Se 110 deles iam cancelar pagando R$ 80, o
desconto segura 33, que pagam R$ 10.560 a mais. O lucro é R$ 8.560.

Escolher o limiar também é um ajuste, então ele sai das provas simuladas,
e não da prova final. Com esses números, o melhor limiar é 0,25, e ele
rende uns 20% a mais que o 0,50. Se cada ligação custasse R$ 60, o melhor
limiar subiria para 0,85, e o 0,50 daria prejuízo. Mesmo modelo, mesmos
clientes, decisão diferente.

## E se?

Com o modelo pronto, dá para simular. O cliente de maior risco no teste
tem 97% de risco. Se as entregas dele não tivessem atrasado, o modelo
daria 90%. Zerando os atrasos de todo mundo, uns 190 clientes sairiam da
lista de ligações.

O simulador mostra o que o *modelo* prevê. O modelo aprendeu associações,
e só um teste na vida real mostra o efeito de uma mudança.

## O projeto

O `projeto-aula-04/` organiza o mesmo sistema em arquivos, e vai um passo
além: o Optuna também testa tipos diferentes de freio.

| Arquivo | Papel |
|---|---|
| `modelos.py` | preparo, baseline, ajuste fino, avaliação e explicação |
| `treinar.py` | treina e guarda o modelo num arquivo |
| `app.py` | painel com visão geral, modelo, explicação, lista de ligações e simulador |

Treinar e usar ficam separados: o treino guarda o preparo e o modelo
juntos num arquivo, e o painel só carrega esse arquivo.

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
