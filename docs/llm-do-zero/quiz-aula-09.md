---
description: Um quiz de 5 perguntas depois de cada microaula da Aula 9
---

# Quizzes da Aula 9

{% hint style="info" %}
**Como usar esta página**

Depois de cada microaula, e antes do exercício no Colab, responda o quiz
dela. As perguntas aparecem uma de cada vez, como uma pilha de cartas:
responda, leia a explicação e passe para a próxima. No fim aparece quantas
você acertou. Nada fica gravado: dá para refazer quantas vezes quiser.
{% endhint %}

## Microaula 1: o que um LLM faz

{% pilha titulo="Microaula 1 · O que um LLM faz" %}

```quiz
O que um modelo de linguagem devolve quando recebe um pedaço de texto?
- A próxima palavra certa, uma só.
- Um resumo do texto.
+ As chances de cada token que pode vir em seguida.
- A tradução do texto para números, e mais nada.
? Ele devolve uma distribuição de probabilidade sobre o próximo token. Conversar, resumir e traduzir vêm de repetir essa previsão muitas vezes.
```

```quiz
Nos livros de Machado, a continuação mais comum de "minha mãe" aparece em 9% das vezes. O que isso mostra?
+ Que não existe uma resposta certa: existe uma distribuição de chances.
- Que o modelo ainda não foi treinado.
- Que Machado escrevia de um jeito errado.
- Que 9% é a resposta certa, e o resto é erro.
? Existem 63 continuações diferentes, e nenhuma domina. Por isso a saída do modelo é uma distribuição, como a softmax da Aula 8.
```

```quiz
No modelo de contagem, como se calcula a chance de vir "u" depois de "q"?
- Contando quantas vezes o "u" aparece no livro inteiro.
- Dividindo 1 pelo número de letras do alfabeto.
- Somando as vezes que "q" e "u" aparecem.
+ Dividindo as vezes que "u" veio depois de "q" pelo total de letras que vieram depois de "q".
? É a contagem do par dividida pela contagem da letra atual: 33.842 / 33.847, ou 99,99%.
```

```quiz
O texto gerado pelo modelo de contagem de letras tem letras e espaços nos lugares certos, mas quase nenhuma palavra existe. Por quê?
- Porque o corpus de Machado é pequeno demais.
+ Porque ele só olha uma letra para trás.
- Porque o sorteio está errado.
- Porque ele não sabe o que é um espaço.
? Para formar palavras e frases, o modelo precisaria lembrar do que veio bem antes. Com uma letra só de contexto, cada escolha ignora todo o resto.
```

```quiz
Por que não dá para simplesmente contar com 10 letras de contexto em vez de uma?
- Porque o computador não aguenta contar até 10.
- Porque a contagem de 10 letras dá sempre o mesmo resultado.
- Porque a regra do "qu" deixa de valer.
+ Porque a tabela fica gigante e quase toda combinação de 10 letras nunca apareceu no corpus.
? Cada letra a mais multiplica a tabela por 118. Com 10 letras são 5 × 10²⁰ linhas, quase todas vazias. Uma rede neural resolve isso porque contextos parecidos dividem os mesmos pesos.
```

{% endpilha %}

## Microaula 2: a unidade certa

{% pilha titulo="Microaula 2 · A unidade certa" %}

```quiz
Qual é o problema de usar a palavra inteira como peça do modelo?
- O texto fica longo demais.
+ O vocabulário fica enorme, e uma palavra nova não tem entrada nenhuma.
- As palavras não cabem em bytes.
- O modelo não consegue contar palavras.
? Só Machado tem 50.390 palavras diferentes. E uma palavra que o modelo nunca viu (um nome, um erro de digitação) não tem número nenhum.
```

```quiz
E qual é o problema de usar a letra como peça?
- Existem letras demais no português.
- As letras não têm número.
- O modelo não sabe juntar letras.
+ O texto vira uma fila muito longa, e o custo cresce com o número de peças.
? Com 118 símbolos nada fica de fora, mas "O menino era pai do homem." vira 26 peças em vez de 7. O modelo paga quase quatro vezes mais para dizer a mesma coisa.
```

```quiz
Qual é a regra do BPE?
+ Encontre o par de peças vizinhas mais comum, junte as duas numa peça nova, e repita.
- Corte o texto sempre de três em três letras.
- Junte as palavras que um linguista marcou como importantes.
- Separe o texto nas sílabas de cada palavra.
? É a regra inteira, e ela cabe numa frase. Ninguém escolhe as peças: elas saem da contagem.
```

```quiz
Em "casa casada casamento", os pares "c + a", "a + s" e "s + a" aparecem 3 vezes cada. O que o BPE faz?
- Funde os três de uma vez.
- Para, porque não sabe qual escolher.
+ Usa uma regra fixa de desempate e funde um deles.
- Sorteia um par diferente a cada vez que roda.
? Empates acontecem em textos pequenos. O programa desempata por uma regra fixa (no nosso caso, o par que aparece primeiro), para sair sempre a mesma tabela.
```

```quiz
As últimas fusões do tokenizador de Machado são palavras inteiras, como "·fazer" e "·filha". O que isso diz?
- Que o algoritmo errou no fim.
- Que essas palavras são raras.
- Que o tokenizador foi feito à mão.
+ Que o vocabulário é um retrato do corpus: o que Machado usa muito vira peça.
? As primeiras fusões são pedaços comuns em qualquer português ("ra", "qu"). As últimas já são palavras frequentes nos livros. Treinado em outro texto, o vocabulário seria outro.
```

{% endpilha %}

## Microaula 3: do texto aos números, e de volta

{% pilha titulo="Microaula 3 · Codificar e decodificar" %}

```quiz
Por que o BPE começa dos bytes, e não das letras?
- Porque bytes são mais rápidos de digitar.
+ Porque qualquer texto, em qualquer língua, é uma fila de bytes: nenhum caractere fica de fora.
- Porque o português não tem letras suficientes.
- Porque os bytes já vêm agrupados em palavras.
? Um byte tem 256 valores possíveis. Um emoji ou um ideograma vira vários bytes, e o modelo nunca encontra um símbolo que não saiba escrever.
```

```quiz
O vocabulário do curso tem 1.024 tokens. De onde vem esse número?
- 1.024 palavras escolhidas de Machado.
- 1.024 letras diferentes do corpus.
+ 256 bytes mais 768 fusões.
- 512 sílabas mais 512 palavras.
? Os 256 bytes são a base. Cada fusão acrescenta uma peça. O professor decidiu parar em 1.024, um tamanho que cabe no Colab.
```

```quiz
Por que o tokenizador oficial separa o texto em palavras antes de fundir?
- Para o texto ficar mais longo.
- Porque o BPE só funciona com uma palavra de cada vez.
- Para tirar a pontuação do texto.
+ Para nenhuma fusão atravessar o espaço ou a pontuação, e "dele." e "dele," dividirem a mesma peça.
? Sem essa separação, o BPE aprende peças como "o,·", que juntam fim de palavra, vírgula e espaço, e desperdiçam vocabulário.
```

```quiz
Ao codificar uma palavra, em que ordem as fusões são aplicadas?
+ Na ordem em que o treino aprendeu: sempre a de número menor primeiro.
- Da maior peça para a menor.
- Em ordem alfabética.
- Na ordem em que as letras aparecem na palavra.
? É essa ordem que faz a mesma palavra ser cortada do mesmo jeito que o modelo viu no treino.
```

```quiz
O que acontece se você treinar o modelo com uma tabela de fusões e usar outra na hora de conversar com ele?
- Nada: os números são os mesmos.
- O modelo fica mais rápido.
+ Os números passam a significar outra coisa, e a saída vira lixo.
- O modelo aprende a tabela nova sozinho.
? É o erro do dia. O tokenizador é parte do modelo: os dois andam juntos, sempre. Por isso o notebook carrega a tabela oficial em vez de treinar uma nova.
```

{% endpilha %}

## Microaula 4: do texto ao lote de treino

{% pilha titulo="Microaula 4 · O lote de treino" %}

```quiz
Um lote de 32 janelas de 128 tokens é um tensor. Qual é o formato (shape) dele?
- [128, 32]
+ [32, 128]
- [4096]
- [1024, 128]
? Uma linha por janela (32) e uma coluna por token da janela (128). Ler o formato é metade de entender o código.
```

```quiz
Por que os últimos 10% do corpus ficam fora do treino?
- Para o treino ser mais rápido.
- Porque o fim dos livros é menos importante.
+ Para medir se o modelo aprendeu português ou decorou o texto que viu.
- Porque o tokenizador não consegue ler o fim.
? É o conjunto de validação, como o conjunto de teste das aulas anteriores. O modelo nunca treina nele.
```

```quiz
Na janela de treino, o que é o alvo?
- A última palavra da frase.
- Um token sorteado ao acaso.
- A mesma janela de trás para a frente.
+ A mesma sequência da entrada, andada uma casa.
? Cada posição tem que prever o token seguinte. Por isso nove tokens de texto dão oito exemplos de treino de uma vez.
```

```quiz
Quantas previsões o modelo corrige num passo de treino com 32 janelas de 128 tokens?
- 32
- 128
+ 4.096
- 1.024
? 32 × 128 = 4.096. Cada posição de cada janela é um exemplo, e todos entram no mesmo passo.
```

```quiz
Um modelo que chuta igual entre os 1.024 tokens tem perda 6,93. O contador de pares de tokens tem perda 4,48. O que isso quer dizer?
+ Que contar já é bem melhor que chutar, e o transformer vai ter que fazer melhor que 4,48.
- Que o contador é pior que o chute, porque 4,48 é menor.
- Que a perda não serve para comparar modelos.
- Que o transformer nunca passa de 4,48.
? Quanto menor a perda, menos o modelo se surpreende com o token certo. O 4,48 é o número a bater na Aula 11.
```

{% endpilha %}

## Para revisar

- [Aula 9 — Do Texto aos Números](aula-09-tokenizacao.md): a teoria completa.
- [Colas dos Exercícios](colas-dos-exercicios.md#aula-9-do-texto-aos-numeros): as dicas de cada exercício.
