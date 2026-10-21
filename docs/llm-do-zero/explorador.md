---
description: O modelo do curso rodando aqui na página, etapa por etapa
---

# Explorador do LLM

{% hint style="info" %}
**O que esta página faz**

Ela roda o modelo de linguagem do curso aqui mesmo, no seu navegador: os
809.728 pesos que o professor treinou por uma hora em Machado de Assis.
Escreva qualquer frase e acompanhe o caminho inteiro, do texto até as
chances da próxima palavra. Nada é enviado para lugar nenhum.
{% endhint %}

A ideia é a mesma do [Transformer Explainer](https://poloclub.github.io/transformer-explainer/),
que mostra o GPT-2 por dentro, só que com o modelo das Aulas 10 e 11: quatro
blocos em vez de doze, 128 números por token em vez de 768, e um
vocabulário de 1.024 tokens tirado de Machado. Por ser pequeno, ele cabe
inteiro na página, e você vê cada número de verdade.

{% explorador %}

## O caminho, em quatro etapas

O modelo faz sempre o mesmo caminho, da esquerda para a direita:

| Etapa | O que entra | O que sai | Pesos |
|---|---|---|---|
| 1. Tokens | o texto | uma fila de números, um por peça | nenhum: é o tokenizador da Aula 9 |
| 2. Vetores | os números | 128 números por token | 147.456 |
| 3. Quatro blocos | 128 números por token | 128 números por token, agora com contexto | 529.920 |
| 4. Saída | os 128 números da última posição | 1.024 chances, uma por token | 132.352 |

Quase todo o trabalho está na etapa 3. As outras só traduzem: texto em
números na entrada, números em chances na saída.

## Etapa 1: tokens

O tokenizador BPE da [Aula 9](aula-09-tokenizacao.md) corta o texto em
peças e troca cada peça pelo seu número. Palavras comuns em Machado viram
um token só (`·casa`, `·elle`); nomes e palavras raras viram vários
(`·C`, `ap`, `it`, `ú`). O ponto no começo de uma peça é o espaço que vem
antes dela.

O modelo olha no máximo 128 tokens para trás: é a janela de contexto. Se
o texto passar disso, só os últimos 128 entram.

## Etapa 2: o token vira vetor

Cada número escolhe uma linha da tabela de embedding: 128 números. A essa
linha se soma a linha da posição, que diz onde o token está na frase. É o
que a [Aula 10](aula-10-como-funciona-um-llm.md#o-token-vira-vetor)
explica, e o explorador mostra as três faixas: a do token, a da posição e
a soma, que é o que entra no primeiro bloco.

Embaixo aparecem os tokens com o vetor mais parecido, pela similaridade
do cosseno. Clique em `·elle` e veja `·ella` no topo: palavras que fazem o
mesmo papel na frase ficaram com vetores parecidos durante o treino.

## Etapa 3: a atenção, nos quatro blocos

Cada bloco tem duas metades. A atenção deixa cada token olhar para os
anteriores e misturar o que viu; a camada densa pensa sobre cada posição
sozinha. O explorador mostra a primeira metade, que é onde dá para ver
alguma coisa: para onde cada token olha.

Escolha uma camada e uma cabeça (ou a média das quatro) e clique num
token. As barras mostram quanto ele olha para cada token anterior. O mapa
embaixo mostra todos os tokens de uma vez: cada linha é um token, cada
coluna é para onde ele olha, e o triângulo de cima fica vazio, porque a
[máscara causal](aula-10-como-funciona-um-llm.md#a-mascara-nao-vale-olhar-a-resposta)
não deixa ninguém olhar o futuro.

## Etapa 4: as chances do próximo token

Depois dos quatro blocos, a última posição vira 1.024 pontuações, uma
para cada token do vocabulário, e a softmax transforma as pontuações em
chances. Os dois botões da [Aula 11](aula-11-treinar-e-usar.md) estão
acima do texto:

- a **temperatura** divide as pontuações antes da softmax: baixa
  concentra a chance no favorito, alta espalha;
- o **top-k** deixa no sorteio só os `k` tokens mais prováveis.

"Gerar 1 token" sorteia um token com essas chances e cola no fim do
texto. Tudo roda de novo, com o texto um token mais longo: é a geração
autorregressiva.

## Experimentos para fazer

| Faça | E repare |
|---|---|
| Escreva "Quincas Bor" | o favorito passa de 90%: só existe uma continuação razoável |
| Escreva "Não sei se a senhora" | o favorito mal passa de 8%: o modelo está em dúvida |
| Gere 10 tokens com temperatura 0,2 | o texto entra em loop |
| Gere 10 tokens com temperatura 2 e sem corte, depois com top-k 10 | o corte salva a temperatura alta |
| Clique em "janela", em "A casa de Capitú tinha uma janela", camada 4 | os pedaços de "janela" olham para "uma" |
| Clique em `·dous`, `·era` ou `·olhos` na etapa 2 | os vizinhos fazem o mesmo papel na frase |
| Troque a última palavra de uma frase | o mapa de atenção das posições anteriores não muda |

## O que o explorador não mostra

Ele mostra o modelo pronto, não o treino: os pesos são os do professor, e
não mudam enquanto você mexe. Para ver o treino acontecer, use o notebook
da [Aula 11](aula-11-treinar-e-usar.md). A camada densa de cada bloco e as
normalizações também estão lá, rodando, só que sem desenho: 128 números
virando 256 e voltando a 128 não dizem nada a olho nu.

E ele escreve como Machado, com a grafia de 1899, porque foi só isso que
leu. Escreva sobre celulares e ele vai continuar falando de viscondes.
