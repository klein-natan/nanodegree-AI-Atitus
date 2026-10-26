---
description: O modelo do curso rodando aqui na página, passo a passo, e o treino dele na GPU do Colab
---

# Explorador do LLM

{% hint style="info" %}
**O que esta página faz**

Ela roda o modelo de linguagem do curso aqui mesmo, no seu navegador: os
809.728 pesos que o professor treinou por uma hora em Machado de Assis.
Escreva qualquer frase e acompanhe o caminho inteiro, do texto até as
chances da próxima palavra. Nada é enviado para lugar nenhum. No fim da
página, um notebook treina este mesmo modelo do zero, em 5 minutos.
{% endhint %}

A ideia é a mesma do [Transformer Explainer](https://poloclub.github.io/transformer-explainer/),
que mostra o GPT-2 por dentro, só que com o modelo das Aulas 10 e 11: quatro
blocos em vez de doze, 128 números por token em vez de 768, e um
vocabulário de 1.024 tokens tirado de Machado. Por ser pequeno, ele cabe
inteiro na página, e você vê cada número de verdade.

{% explorador %}

## O caminho inteiro, num exemplo

Esta página segue um texto curto, "A casa de", do começo ao fim. São oito
passos, e o explorador mostra cada um deles com o texto que você escrever.

| Passo | O que acontece | Formato depois | Pesos |
|---|---|---|---|
| 1 | o texto vira tokens | 3 números | nenhum |
| 2 | cada token vira um vetor | 3 × 128 | 147.456 |
| 3 | a normalização arruma os números | 3 × 128 | 256 por camada |
| 4 | a atenção mistura cada token com os anteriores | 3 × 128 | 66.048 por bloco |
| 5 | a camada densa pensa sobre cada posição | 3 × 128 | 65.920 por bloco |
| 6 | os passos 3 a 5 se repetem, quatro blocos | 3 × 128 | 529.920 nos quatro |
| 7 | a última posição vira chances | 1.024 | 132.352 |
| 8 | um token é sorteado, e tudo recomeça | 4 tokens | nenhum |

Repare na coluna do formato: do passo 2 ao 6, cada token é sempre uma lista
de 128 números. O modelo não muda o tamanho do vetor; muda o que está
escrito nele.

## Passo 1: o texto vira tokens

O tokenizador da [Aula 9](aula-09-tokenizacao.md) separa o texto em
palavras, transforma cada palavra em bytes e funde os pares de bytes na
ordem em que o BPE aprendeu. Cada peça que sobra tem um número, entre 0 e
1.023.

| Peça | `A` | `·casa` | `·de` |
|---|---|---|---|
| Número | 65 | 503 | 273 |

O ponto no começo de `·casa` é o espaço que vem antes da palavra: ele faz
parte do token. Palavras comuns em Machado viram um token só; nomes e
palavras raras viram vários (`Capitú` vira `·C`, `ap`, `it`, `ú`). No
explorador, é a etapa 1: clique num token para segui-lo pelas próximas.

O modelo olha no máximo 128 tokens para trás. É a janela de contexto: se o
texto passar disso, só os últimos 128 entram.

## Passo 2: cada token vira um vetor

O número do token é só um endereço. Ele escolhe uma linha da tabela de
embedding, que tem 1.024 linhas (uma por token) e 128 colunas. O token 503
vira a linha 503: 128 números que o treino ajustou.

A essa linha se soma uma segunda, a da **posição**: a tabela de posições
tem uma linha para cada lugar da janela, de 0 a 127. `·casa` está na
posição 1, então recebe a linha 1. É assim que o modelo sabe a ordem das
palavras.

$$\text{vetor de entrada} = \text{embedding do token} + \text{embedding da posição}$$

| Símbolo | Significado |
|---|---|
| embedding do token | a linha da tabela de tokens: o que a palavra é |
| embedding da posição | a linha da tabela de posições: onde ela está |

Exemplo: se a primeira coluna do token vale 0,12 e a da posição vale
−0,05, a primeira coluna do vetor de entrada vale 0,07. A soma é feita
coluna por coluna, nas 128.

No explorador, a etapa 2 mostra as três faixas, de vermelho (positivo) a
azul (negativo), e os tokens com vetor mais parecido. Tokens que fazem o
mesmo papel na frase ficaram parecidos durante o treino: o vizinho de
`·elle` é `·ella`; os de `·era` são `·fosse` e `·foi`.

## Passo 3: a normalização

Antes de cada etapa de conta, os 128 números de cada token passam por uma
**normalização**: subtrai-se a média deles e divide-se pelo desvio padrão.
Assim os números ficam sempre num tamanho parecido, e o treino não desanda
quando eles crescem.

$$\hat{x}_i = \frac{x_i - \mu}{\sigma}$$

| Símbolo | Significado |
|---|---|
| `xᵢ` | um dos 128 números do token |
| `μ` | a média dos 128 |
| `σ` | o desvio padrão dos 128 |

Exemplo com dois números, `(3, 5)`: a média é 4, o desvio padrão é 1, e o
resultado é `(−1, 1)`. Depois, pesos aprendidos esticam e deslocam cada
número.

## Passo 4: a atenção

É a única etapa em que um token vê os outros. Cada token faz uma
**pergunta** (Q), cada token anterior mostra uma **etiqueta** (K), e a
combinação das duas diz quanto ele olha para cada um. Ele então leva um
pouco do **conteúdo** (V) de cada token anterior, na proporção desses
pesos.

$$\text{Atenção}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}}\right) V$$

| Símbolo | Significado |
|---|---|
| `Q Kᵀ` | cada pergunta comparada com cada etiqueta |
| `√d` | a raiz do tamanho dos vetores, que segura o tamanho dos números |
| `softmax` | transforma as comparações em pesos que somam 100% |
| `V` | os conteúdos, misturados com esses pesos |

Em "A casa de", o token `·de` pode olhar para `A`, para `·casa` e para si
mesmo, mas não para o que vem depois: a **máscara causal** apaga o futuro.
O modelo faz isso quatro vezes em paralelo, com quatro **cabeças**, e cada
uma aprende a procurar uma coisa diferente.

No explorador, a etapa 3: escolha a camada e a cabeça, e clique num token.
As barras mostram para onde ele olha; o mapa embaixo mostra todos os
tokens de uma vez, com o triângulo de cima vazio por causa da máscara. Na
camada 4, os pedaços de "janela" olham para o "uma" que veio antes deles.

## Passo 5: o atalho e a camada densa

O resultado da atenção não substitui o vetor do token: ele é **somado** a
ele. É o atalho. Cada etapa acrescenta alguma coisa ao que já estava lá, em
vez de reescrever.

Depois vem a **camada densa**, aplicada a cada token separadamente: os 128
números viram 256, passam pela ativação GELU (uma ReLU com a dobra
arredondada) e voltam a ser 128. O resultado também é somado ao vetor.

> A atenção **junta** informação dos outros tokens. A camada densa
> **pensa** sobre essa mistura, um token de cada vez.

O explorador não desenha esta etapa: 256 números por token não dizem nada
a olho nu. Mas ela está lá, rodando, e tem metade dos pesos de cada bloco.

## Passo 6: quatro blocos

Os passos 3, 4 e 5 formam um **bloco transformer**, e o modelo empilha
quatro deles. A saída de um é a entrada do seguinte, sempre com 128
números por token. Cada bloco tem os seus próprios pesos e aprende a olhar
para outras coisas: no modelo do curso, as cabeças da camada 1 espalham a
atenção pela frase, e as da camada 3 olham quase só para os vizinhos.

## Passo 7: da última posição às chances

Depois dos quatro blocos, só interessa a **última** posição, a de `·de`:
foi ela que leu o texto inteiro. Os 128 números dela passam por uma última
normalização e por uma camada que os transforma em 1.024 pontuações, os
**logits**, uma para cada token do vocabulário. A softmax transforma as
pontuações em chances que somam 100%.

Depois de "A casa de", o modelo do curso dá 13% a `·D`, 8,5% a `·M` e
8,3% a `·S`: ele espera um nome próprio, como "A casa de D. Carmo".
Nenhum token passa de 15%. Não existe uma resposta certa, existe uma
distribuição.

## Passo 8: escolher, colar e repetir

Para escrever, um token é sorteado com essas chances e colado no fim do
texto. "A casa de" vira "A casa de D", e o caminho inteiro roda de novo,
agora com 4 tokens. É a **geração autorregressiva**: um token por vez, cada
um vira entrada do próximo.

Dois botões mudam o sorteio, como na [Aula 11](aula-11-treinar-e-usar.md):

- a **temperatura** divide as pontuações antes da softmax: baixa concentra
  a chance no favorito, alta espalha;
- o **top-k** deixa no sorteio só os `k` tokens mais prováveis.

No explorador, "Gerar 1 token" faz exatamente isso, e "Gerar 10" repete
dez vezes.

## Experimentos para fazer

| Faça | E repare |
|---|---|
| Escreva "Quincas Bor" | o favorito passa de 90%: só existe uma continuação razoável |
| Escreva "Não sei se a senhora" | o favorito mal passa de 8%: o modelo está em dúvida |
| Gere 10 tokens com temperatura 0,2 | o texto entra em loop |
| Gere 10 tokens com temperatura 2 e sem corte, depois com top-k 10 | o corte salva a temperatura alta |
| Clique em "a", o último pedaço de "janela", em "A casa de Capitú tinha uma janela", camada 4 | ele olha para "uma" |
| Clique em `·dous`, `·era` ou `·olhos` na etapa 2 | os vizinhos fazem o mesmo papel na frase |
| Troque a última palavra de uma frase | o mapa de atenção das posições anteriores não muda |

## O que o explorador não mostra

Ele mostra o modelo pronto, não o treino: os pesos são os do professor, e
não mudam enquanto você mexe. E ele escreve como Machado, com a grafia de
1899, porque foi só isso que leu. Escreva sobre celulares e ele vai
continuar falando de viscondes.

## Treine este modelo na GPU do Colab

O notebook abaixo treina este mesmo modelo do zero, do texto cru ao modelo
escrevendo, em Keras. Não tem exercício: é o código inteiro de um treino,
com um comentário curto em cada etapa, para ler de cima para baixo.

| Etapa | O que o código faz |
|---|---|
| 1. O ambiente | liga a placa de vídeo, com precisão mista e o compilador XLA |
| 2. O texto | baixa a obra de Machado, 3,7 milhões de caracteres |
| 3. O tokenizador | transforma o texto em 1,5 milhão de tokens e separa a validação |
| 4. Os lotes | sorteia janelas de 128 tokens, 128 por lote, dentro do TensorFlow |
| 5. O modelo | monta o GPT pequeno com as camadas prontas do Keras |
| 6. A perda e o otimizador | entropia cruzada, AdamW e o teste do passo zero |
| 7. O relógio | aquece, resfria e para o treino em 5 minutos, salvando o melhor |
| 8. O treino | o `fit`, com uma frase escrita pelo modelo a cada conferência |
| 9. A curva | a perda de treino e de validação ao longo do tempo |
| 10. O modelo escrevendo | texto gerado com três temperaturas |
| 11. O arquivo | os pesos, para baixar |

Antes de rodar, troque o ambiente para uma GPU: menu *Ambiente de
execução → Alterar o tipo de ambiente de execução → T4 GPU*. Depois,
*Executar tudo*. São uns 7 minutos: 5 de treino e o resto de preparação.
Compare a curva e o texto do fim com os 35 minutos de processador da Aula
11: é a mesma conta, numa máquina feita para ela.

- **Notebook do treino, no Google Colab:** [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klein-natan/nanodegree-AI-Atitus/blob/main/notebooks/treinar-um-llm-na-gpu.ipynb)
