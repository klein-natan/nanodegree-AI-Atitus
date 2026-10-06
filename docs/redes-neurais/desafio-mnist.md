---
description: Um desafio em dupla: treinar uma rede densa que lê dígitos escritos à mão
---

# Desafio MNIST

{% hint style="info" %}
**Como funciona**

Em dupla, no Colab, você treina uma rede neural densa que lê dígitos
escritos à mão. Você completa o código, escreve uma
previsão antes de cada treino e muda uma coisa por vez. Não tem placar:
vale entender o que muda.
{% endhint %}

## O desafio

O MNIST é um conjunto clássico: 70.000 imagens de 28 por 28 pixels, cada
uma com um dígito de 0 a 9. A tarefa é a da Aula 7, com dez respostas
possíveis em vez de duas.

| Entra | Sai |
|---|---|
| uma imagem de 28 × 28 pixels | o dígito, de 0 a 9 |

## Como participar

1. Abra o notebook no Colab (o botão está em **Materiais**) e salve uma
   cópia no seu Drive.
2. Preencha os `____` do código, de cima para baixo.
3. Antes de cada treino, escreva a sua previsão. Depois, explique o que
   viu.
4. Faça três experimentos, mudando uma coisa por vez, e preencha a tabela.
5. Rode o teste do embaralhamento e anote a acurácia da sua rede. Você vai
   usá-la na Aula 8.

## O que você vai ver

| Parte | O que você faz |
|---|---|
| 0 e 1 | olha as imagens e leva os pixels de 0 a 255 para 0 a 1 |
| 2 e 3 | monta e compila uma rede densa |
| 4 e 5 | treina, lê as curvas e olha onde a rede erra |
| 6 | três experimentos: neurônios, camadas e épocas |
| 7 | embaralha os pixels e descobre o que a rede densa não vê |

## Depois do desafio

Na [Aula 8](aula-08-redes-convolucionais.md) você treina uma rede
convolucional no mesmo MNIST e compara com a sua rede densa. Guarde a
acurácia e o número de parâmetros que você anotou.

## Materiais

- **Notebook do desafio, no Google Colab:** [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klein-natan/nanodegree-AI-Atitus/blob/main/notebooks/desafio-mnist.ipynb)
- Dados: o MNIST vem dentro do Keras, com `keras.datasets.mnist.load_data()`. Não há arquivo para baixar.
- Antes: a [Aula 7](aula-07-redes-neurais-densas.md), que ensina a rede densa.
