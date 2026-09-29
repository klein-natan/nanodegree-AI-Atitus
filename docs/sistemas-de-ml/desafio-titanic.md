---
description: Um desafio de 45 minutos em equipe, com placar ao vivo da turma
---

# Desafio Titanic

{% hint style="info" %}
**Como funciona**

Em equipe, você tem 45 minutos para construir um classificador que diga
quem sobreviveu ao naufrágio. Você treina com 1.000 passageiros, prevê
outros 300 e envia as previsões para o placar da turma, que fica nesta
mesma página. Ganha o maior F1.
{% endhint %}

## O desafio

Em 1912, o Titanic afundou na primeira viagem. Os dados deste desafio
imitam os passageiros daquele navio: quem eram, em que classe viajavam,
com quantos parentes, quanto pagaram. Eles são **sintéticos**: nenhum
passageiro existe de verdade, e por isso a resposta não está em lugar
nenhum da internet.

A tarefa é a da Aula 3: uma classificação com duas respostas possíveis.
Sobreviveu (1) ou não sobreviveu (0).

## Como participar

1. Forme a equipe e escolha um nome. Use sempre o mesmo nome em todos os
   envios.
2. Abra o notebook do desafio no Colab (o botão está em **Materiais**, no
   fim da página) e salve uma cópia no seu Drive.
3. Treine o seu classificador com `titanic_treino.csv`.
4. Preveja os 300 passageiros de `titanic_teste.csv`.
5. Rode a última célula do notebook: ela baixa o arquivo `envio.csv`.
6. Envie o arquivo no placar, na seção **O placar**, mais abaixo.

## Os dados

| Coluna | O que é |
|---|---|
| `id_passageiro` | o número do passageiro. Não é característica: não use no modelo |
| `classe` | 1, 2 ou 3: a classe da passagem |
| `sexo` | `feminino` ou `masculino` |
| `idade` | em anos |
| `irmaos_conjuge` | quantos irmãos ou cônjuges viajavam junto |
| `pais_filhos` | quantos pais ou filhos viajavam junto |
| `tarifa` | quanto a passagem custou |
| `porto_embarque` | `Southampton`, `Cherbourg` ou `Queenstown` |
| `sobreviveu` | **a resposta**: 1 se sobreviveu. Só existe no treino |

No treino, 35% dos passageiros sobreviveram. O teste tem a mesma
proporção, mas chega sem a coluna `sobreviveu`: a resposta fica só com o
professor.

## O arquivo de envio

O placar aceita um arquivo CSV com três colunas e os 300 passageiros do
teste:

| Coluna | O que colocar |
|---|---|
| `id_passageiro` | os mesmos números de `titanic_teste.csv` |
| `sobreviveu` | a decisão do seu modelo: 0 ou 1 |
| `probabilidade` | a chance de sobreviver, de 0 a 1 (a saída do `predict_proba`) |

A última célula do notebook já monta o arquivo nesse formato:

```python
envio = pd.DataFrame({
    "id_passageiro": teste["id_passageiro"],
    "sobreviveu": previsoes,
    "probabilidade": probabilidades,
})
envio.to_csv("envio.csv", index=False)
```

Se o arquivo tiver algum problema (coluna faltando, passageiro a menos,
valor fora de 0 e 1), o placar recusa o envio e explica o que corrigir. O
envio recusado não conta no seu limite.

## Como o placar ordena

O placar mostra cinco métricas da Aula 3: acurácia, precisão, recall, F1
e AUC. A ordem é pelo **F1**, que junta precisão e recall num número só.

$$F_1 = \frac{2 \cdot \text{precisão} \cdot \text{recall}}{\text{precisão} + \text{recall}}$$

| Símbolo | Significado |
|---|---|
| precisão | dos passageiros que o modelo disse que sobreviveram, a fração que sobreviveu de verdade |
| recall | dos passageiros que sobreviveram de verdade, a fração que o modelo encontrou |
| `F₁` | a média harmônica das duas: só fica alta se as duas forem altas |

Exemplo: um modelo com precisão de 0,60 e recall de 0,70 tem F1 igual a
`2 × 0,60 × 0,70 / (0,60 + 0,70) = 0,646`.

O placar usa o F1 porque a acurácia engana aqui. Prever que **ninguém**
sobrevive acerta 65% dos passageiros, sem modelo nenhum, mas tem F1 zero:
não encontra um sobrevivente sequer. Essa previsão aparece no placar como
a régua. A sua equipe tem que ficar acima dela.

Em caso de empate no F1, desempata a AUC, que mede se o modelo dá
probabilidades maiores para quem sobreviveu.

## As regras

- Cada equipe pode enviar **até 5 vezes**. No placar, vale o melhor F1 da
  equipe.
- O nome da equipe não diferencia maiúsculas: "Iceberg" e "iceberg" são a
  mesma equipe.
- Use só os dados do desafio. O modelo é livre: o que vimos no curso ou
  qualquer outro do scikit-learn.
- O desafio dura 45 minutos. Quando o professor encerrar, vale o placar
  daquele momento.

## Dicas

- `sexo`, `porto_embarque` e até `classe` são categorias. O
  `pd.get_dummies` da Aula 2 transforma cada uma em colunas de 0 e 1.
  Aplique a mesma transformação no treino e no teste.
- A regressão logística da Aula 3 é um ótimo começo.
- O limiar de 0,5 não é obrigatório. Baixar o limiar aumenta o recall, e
  pode aumentar o F1.
- Com só 5 envios, meça antes de enviar. Separe um pedaço do treino para
  validação, ou use a validação cruzada da Aula 3. O placar não é lugar de
  tentativa e erro.

## O placar

<!-- placar:inicio -->
{% hint style="warning" %}
**O placar está desligado.** Ele só fica no ar durante o desafio, em sala.
Quando o professor ligar, o placar aparece aqui mesmo, nesta página.
Recarregue a página se ele não aparecer.
{% endhint %}
<!-- placar:fim -->

## Materiais

- **Notebook do desafio, no Google Colab:** [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klein-natan/nanodegree-AI-Atitus/blob/main/notebooks/desafio-titanic.ipynb)
- Dados de treino: [`titanic_treino.csv`](https://github.com/klein-natan/nanodegree-AI-Atitus/blob/main/data/titanic_treino.csv)
- Dados de teste: [`titanic_teste.csv`](https://github.com/klein-natan/nanodegree-AI-Atitus/blob/main/data/titanic_teste.csv)

Se ainda não sabe como abrir o notebook, veja
[Antes de começar](../antes-de-comecar.md) primeiro.
