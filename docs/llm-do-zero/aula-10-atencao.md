---
description: Como o modelo decide para onde olhar
---

# Aula 10 — Atenção

{% hint style="info" %}
**O que você leva desta aula**

Você vai escrever o seu primeiro modelo em PyTorch, calcular à mão o peso
de atenção de uma palavra, e montar uma cabeça de atenção com máscara. No
fim, vai medir que um modelo que olha para trás erra menos que um que não
olha. É a única peça realmente nova de um transformer.
{% endhint %}

## Onde estamos

<figure><img src="../assets/aula-10/mapa.png" alt="Quatro quadros lado a lado, um por aula: do texto para números, para onde olhar, montar e treinar, gerar e duvidar. O segundo está aceso"><figcaption>A Aula 9 entregou uma fila de números. Esta aula ensina o modelo a escolher para onde olhar dentro dela.</figcaption></figure>

A Aula 9 terminou com um problema: o modelo de contagem só olha um token
para trás, e olhar mais longe com uma tabela é impossível. Esta aula
resolve isso em dois passos. Primeiro, o modelo de contagem vira uma rede
neural. Depois, a rede ganha a atenção.

## O primeiro modelo em PyTorch

### O embutimento: o token vira vetor

Cada token da Aula 9 é um número inteiro, e número inteiro não serve para
fazer conta de parecença. Como a Aula 9 terminou mostrando, o número é só
um endereço. O primeiro passo do modelo é trocar cada token pela linha
dele numa tabela:

```python
embutir = nn.Embedding(1024, 128)   # 1.024 tokens, 128 números cada
```

Essa tabela se chama **embutimento** (*embedding*). São 131.072 pesos, que
o treino ajusta como quaisquer outros. Depois dela, o token 600 deixou de
ser "o número 600" e virou um ponto num espaço de 128 dimensões. Tokens
que aparecem em contextos parecidos acabam com linhas parecidas, sem
ninguém mandar, do mesmo jeito que os filtros da Aula 8 apareceram sozinhos.

### Uma camada que dá uma pontuação para cada token

O modelo mais simples possível pega o vetor do token atual e o transforma
em 1.024 pontuações, uma para cada token que pode vir em seguida. A
softmax transforma as pontuações em probabilidades. As pontuações cruas
têm nome: **logits**.

No Keras você empilhava camadas numa lista. No PyTorch você escreve uma
classe com duas partes: o `__init__` cria as camadas, e o `forward` diz o
caminho dos números.

```python
class ModeloBigrama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embutir = nn.Embedding(1024, 128)
        self.cabeca = nn.Linear(128, 1024)

    def forward(self, indices, alvos):
        logits = self.cabeca(self.embutir(indices))
        perda = F.cross_entropy(logits.view(-1, 1024), alvos.view(-1))
        return logits, perda
```

Esse modelo faz o mesmo que o contador da Aula 9: prevê o próximo token
olhando só o atual. A diferença é que ele **aprende** a tabela por
gradiente, em vez de contar.

### O laço de treino

O `model.fit` do Keras fazia cinco coisas escondido. No PyTorch você vê
todas:

```python
for passo in range(500):
    entrada, alvo = sortear_lote(dados_treino)   # 1. um lote
    logits, perda = modelo(entrada, alvo)        # 2. a perda
    otimizador.zero_grad()                       # 3. apaga o gradiente anterior
    perda.backward()                             # 4. calcula o gradiente
    otimizador.step()                            # 5. dá o passo
```

É o gradiente descendente da Aula 1. O `zero_grad` existe porque o PyTorch
soma os gradientes de uma chamada para outra. Sem ele, cada passo carrega
a soma de todos os anteriores, e o treino desanda.

Antes de treinar, confira o **passo zero**. Um modelo recém-criado não
sabe nada e chuta igual entre 1.024 tokens, então a perda dele tem que
dar perto de `ln(1024) = 6,93`. No exercício da aula, o modelo novo dá
7,1. Em 500 passos, a perda cai para 4,2, que já está abaixo dos 4,48 do
contador da Aula 9. O texto que ele gera tem palavras de verdade e frases
sem sentido, porque o modelo escolhe cada token olhando só o anterior.

## O problema

Leia esta frase e complete a última palavra:

<figure><img src="../assets/aula-10/olhar_para_tras.png" alt="Uma frase de 25 palavras em duas linhas, sobre a casa de Capitú e uma janela, com quatro palavras destacadas e setas ligando a lacuna no fim a cada uma delas"><figcaption>Para completar a última palavra, o modelo precisa voltar mais de vinte palavras.</figcaption></figure>

Você completou sem esforço, porque leu a frase inteira e reteve as quatro
palavras que importam. O modelo precisa fazer o mesmo, e tem duas saídas
ruins pela frente.

**Saída ruim 1: olhar só as últimas palavras.** Barato, e perde tudo o que
está longe. É o modelo que você acabou de treinar, e é como os modelos
funcionavam antes de 2017.

**Saída ruim 2: somar todas as palavras anteriores por igual.** Aí "de",
"uma" e "para" pesam tanto quanto "Capitú". A informação some no meio da
média.

A resposta é uma terceira coisa: deixar cada posição **escolher** para onde
olhar, com pesos que o treino ajusta. Isso se chama **atenção**
(*attention*).

## Primeiro, a versão sem peso nenhum

Antes de qualquer coisa treinável, vale ver que a ideia central da atenção
funciona **sem nenhum peso novo**. Só com os vetores que já temos.

Tome três palavras e quatro números por palavra, para caber na página:

| Palavra | Vetor |
|---|---|
| `o` | (2, 0, 0, 0) |
| `gato` | (0, 3, 1, 0) |
| `dorme` | (0, 1, 1, 0) |

A palavra `dorme` quer saber para onde olhar. São quatro passos, e você
pode fazer os quatro à mão.

**Passo 1: medir parecença com cada palavra.** Multiplique casa com casa e
some. Isso se chama **produto escalar**, e é a mesma conta da convolução da
Aula 8:

$$q \cdot k = \sum_{i=1}^{d} q_i k_i$$

| Símbolo | Significado |
|---|---|
| `q` | o vetor da palavra que está perguntando |
| `k` | o vetor de uma das palavras da frase |
| `d` | quantos números tem cada vetor (aqui, 4) |

Exemplo: `dorme` com `o` dá `0·2 + 1·0 + 1·0 + 0·0 = 0`. Com `gato` dá
`0 + 3 + 1 + 0 = 4`. Com ela mesma dá `0 + 1 + 1 + 0 = 2`. Vetores
apontando para o mesmo lado dão número alto; perpendiculares dão zero.

**Passo 2: dividir pela raiz de `d`.** Aqui, `√4 = 2`. Os três números
viram 0, 2 e 1. O motivo vem logo abaixo.

**Passo 3: passar pela softmax**, a mesma da Aula 8, que transforma
qualquer trio de números em pesos que somam 100%: **9%, 67% e 24%**.

**Passo 4: somar os vetores, pesados por esses números.**

$$z = 0{,}09 \cdot (2,0,0,0) + 0{,}67 \cdot (0,3,1,0) + 0{,}24 \cdot (0,1,1,0)$$

| Símbolo | Significado |
|---|---|
| `z` | o vetor de saída da posição `dorme` |
| os três pesos | quanto ela decidiu olhar para cada palavra |
| os três vetores | as próprias palavras, sem transformação nenhuma |

O resultado é `(0,18, 2,24, 0,91, 0)`. Repare: `dorme` saiu carregando dois
terços de `gato`. A palavra deixou de ser só ela mesma e virou ela mesma
**mais o contexto**.

<figure><img src="../assets/aula-10/conta.png" alt="Quatro painéis em sequência: os produtos 0, 4 e 2; divididos por 2 dando 0, 2 e 1; a softmax dando 9%, 67% e 24%; e a saída como mistura dos três vetores"><figcaption>Quatro passos, três palavras, números pequenos. É a atenção inteira, e ainda não há um peso treinável.</figcaption></figure>

Os mesmos quatro passos, em movimento:

<figure><video src="../assets/aula-10/atencao.mp4" poster="../assets/aula-10/atencao.jpg" controls preload="none" playsinline aria-label="Animação dos quatro passos da atenção sem pesos"></video><figcaption>Produto escalar, divisão por √d, softmax e mistura: "dorme" sai carregando dois terços de "gato".</figcaption></figure>

**Por que dividir pela raiz de `d`?** Quanto mais dimensões, maiores os
produtos escalares, só por haver mais parcelas na soma. Números grandes
fazem a softmax virar tudo-ou-nada: a softmax de (0, 20, 10) é
praticamente (0%, 100%, 0%), enquanto a de (0, 2, 1) é (9%, 67%, 24%).
Com os pesos colados no zero e no um, o gradiente some. A divisão mantém
a distribuição espalhada, e com isso o modelo continua aprendendo.

No PyTorch, os quatro passos cabem em quatro linhas:

```python
pontos = frase @ dorme                   # 1. produto com cada palavra
pontos = pontos / math.sqrt(4)           # 2. divide pela raiz de d
pesos = F.softmax(pontos, dim=0)         # 3. vira peso
saida = pesos @ frase                    # 4. mistura
```

## Os dois buracos da versão sem pesos

A versão de cima funciona, e tem dois defeitos. Os dois dá para medir, e o
resto da aula é tapar um de cada vez.

<figure><img src="../assets/aula-10/espelho.png" alt="À esquerda uma matriz 3 por 3 de pesos de atenção com a diagonal destacada em duas das três linhas; à direita duas barras idênticas comparando a saída da palavra dorme na frase normal e na embaralhada"><figcaption>Buraco 1: a palavra olha para si mesma. Buraco 2: embaralhar a frase não muda nada.</figcaption></figure>

**Buraco 1: vira espelho.** Faça a mesma conta para as três palavras:

| Quem pergunta | olha `o` | olha `gato` | olha `dorme` |
|---|---|---|---|
| `o` | **79%** | 11% | 11% |
| `gato` | 1% | **95%** | 5% |
| `dorme` | 9% | **67%** | 24% |

Em duas das três linhas, a palavra olha principalmente para si mesma. Não é
azar: o produto escalar de um vetor com ele mesmo é o comprimento dele ao
quadrado, o maior valor que ele costuma tirar de qualquer comparação.

E tem um problema mais fundo por trás. Parecença de embutimento é **uma**
relação: palavras de sentido próximo ficam perto. Mas `dorme` não precisa
procurar sinônimos de `dorme`. Precisa procurar o **sujeito** dela, que é
`gato`, e sujeito não é sinônimo de nada. Uma relação só, fixa, não cobre
isso.

**Buraco 2: a frase virou um saco de palavras.** Embaralhe "o gato dorme"
para "dorme gato o" e refaça a conta. A saída de cada palavra é idêntica.
Nada na fórmula diz onde cada palavra está, então "o gato mordeu o
cachorro" e "o cachorro mordeu o gato" são a mesma coisa para ela.

## Os três papéis

A atenção de verdade dá três vetores a cada palavra, obtidos multiplicando
o vetor da palavra por três tabelas de pesos diferentes:

<figure><img src="../assets/aula-10/qkv.png" alt="Três blocos lado a lado: Pergunta (Q), Etiqueta (K) e Conteúdo (V), cada um com uma frase explicando o papel"><figcaption>A mesma palavra, vista de três jeitos. O treino ajusta as três tabelas de pesos.</figcaption></figure>

Pense numa biblioteca. Você chega com uma **pergunta** (o Q da posição
atual). Cada livro tem uma **etiqueta** na lombada (o K de cada palavra
anterior). Você compara a sua pergunta com todas as etiquetas, escolhe os
livros mais parecidos, e lê o **conteúdo** deles (o V).

A diferença para a biblioteca de verdade: aqui você não escolhe um livro
só. Você lê todos, em proporções diferentes.

Cada papel sai de uma multiplicação por uma tabela própria:

$$q = x\,W_Q \qquad k = x\,W_K \qquad v = x\,W_V$$

| Símbolo | Significado |
|---|---|
| `x` | o vetor da palavra, que sai do embutimento |
| `W_Q`, `W_K`, `W_V` | as três tabelas de pesos que o treino ajusta |
| `q`, `k`, `v` | a pergunta, a etiqueta e o conteúdo da palavra |

Exemplo com dois números. A palavra é `x = (1, 2)`. Suponha que `W_Q`
troque os dois números de lugar e que `W_K` não mexa em nada. Então
`q = (2, 1)` e `k = (1, 2)`, e o produto `q · k` dá `2 + 2 = 4`. Sem as
tabelas, a palavra comparada com ela mesma dava `1 + 4 = 5`. Com tabelas
diferentes, o produto de uma palavra com ela mesma deixa de ser
automaticamente o maior. **É isso que tapa o buraco 1**: o treino decide
quem olha para quem.

No código, cada tabela é um `nn.Linear` sem viés:

```python
self.pergunta = nn.Linear(128, 32, bias=False)
self.etiqueta = nn.Linear(128, 32, bias=False)
self.conteudo = nn.Linear(128, 32, bias=False)
```

## A fórmula inteira

$$\text{Atenção}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}}\right) V$$

| Símbolo | Significado |
|---|---|
| `Q Kᵀ` | todos os produtos escalares de todas as perguntas com todas as etiquetas |
| `√d` | a raiz da dimensão, que segura o tamanho dos números |
| `softmax` | transforma os números em pesos que somam 1 |

São os mesmos quatro passos que você já fez à mão, para todas as posições
de uma vez. O `Kᵀ` (K transposta) é a tabela de etiquetas deitada, para que
cada pergunta encontre cada etiqueta. Numa frase de 6 tokens, `Q Kᵀ` é uma
tabela de 6 × 6: uma linha por posição que pergunta, uma coluna por
posição consultada.

## A máscara: não vale olhar a resposta

Existe um problema que o desenho acima esconde. Se cada posição olha todas
as outras, a posição 3 olha a posição 4, que é justamente a palavra que ela
deveria prever.

<figure><img src="../assets/aula-10/mascara.png" alt="Duas matrizes de atenção lado a lado, em porcentagem: à esquerda todos os pesos preenchidos, à direita só o triângulo inferior, com o resto em branco"><figcaption>À esquerda o modelo cola. À direita, cada linha só enxerga o que veio antes dela.</figcaption></figure>

A correção é a **máscara causal**: antes da softmax, troque por menos
infinito todos os produtos que apontam para o futuro. A softmax eleva `e`
a cada número, e `e` elevado a menos infinito é zero:

$$\text{softmax}(1,\; 2,\; -\infty) = (27\%,\; 73\%,\; 0\%)$$

| Símbolo | Significado |
|---|---|
| `1`, `2` | os produtos de duas posições do passado |
| `−∞` | o produto com uma posição do futuro, apagado pela máscara |

Exemplo: `e¹ = 2,72`, `e² = 7,39` e `e^(−∞) = 0`. A soma é 10,11. Divididos
por ela: 27%, 73% e 0%. A posição do futuro some da mistura, e as outras
duas dividem os 100% entre si.

<figure><video src="../assets/aula-10/mascara.mp4" poster="../assets/aula-10/mascara.jpg" controls preload="none" playsinline aria-label="Animação da máscara causal numa matriz de atenção"></video><figcaption>O futuro vira −∞, a softmax transforma em 0%, e cada linha só olha para trás.</figcaption></figure>

```python
futuro = torch.triu(torch.ones(n, n), diagonal=1).bool()
pontos = pontos.masked_fill(futuro, float("-inf"))
```

O nome "causal" vem daí: a informação só corre do passado para o futuro,
nunca ao contrário.

Dá para conferir que a máscara funciona com um teste simples, que o
exercício da aula usa: troque **só a última palavra** da frase e rode a
cabeça de novo. Se as saídas das posições anteriores mudarem, alguma delas
está olhando o futuro.

{% hint style="warning" %}
**Erro do dia**

Esquecer a máscara. O modelo treina lindamente, a perda cai para quase
zero, e você comemora. Na hora de gerar texto, ele não tem mais o futuro
para copiar, e a saída vira lixo. É o erro mais frustrante desta aula,
porque o sintoma aparece longe da causa. O teste da última palavra pega o
erro na hora.
{% endhint %}

## Quatro cabeças, quatro olhares

Uma atenção só tem que decidir tudo com um conjunto de pesos. Na prática,
o modelo roda várias em paralelo, cada uma com sua própria tabela de Q, K e
V, e junta os resultados no fim. Cada uma se chama **cabeça** (*head*).

O nosso modelo tem 4 cabeças por camada. Como o vetor tem 128 números,
cada cabeça trabalha com 32. As quatro saídas de 32, lado a lado, voltam a
ser 128, e uma última tabela (`saida`) mistura o que as quatro acharam:

```python
resultados = []
for cabeca in self.cabecas:
    resultados.append(cabeca(x))
juntos = torch.cat(resultados, dim=-1)    # 4 × 32 = 128
return self.saida(juntos)
```

A conta dos pesos: cada cabeça tem três tabelas de 128 × 32 (12.288
pesos), e as quatro somam 49.152. A tabela de saída tem 128 × 128 =
16.384. Uma camada de atenção inteira tem **65.536 pesos**, o mesmo custo
de uma cabeça grande de 128. O que se ganha é a variedade:

<figure><img src="../assets/aula-10/cabecas.png" alt="Oito mapas de atenção em duas linhas: quatro cabeças da camada 1 e quatro da camada 4, cada uma com um padrão diferente"><figcaption>As oito cabeças, na mesma frase. Nenhuma faz o que a outra faz.</figcaption></figure>

Agora olhe de perto duas delas, na frase "A casa de Capitú tinha uma janela
para a rua.":

<figure><img src="../assets/aula-10/mapa_atencao.png" alt="Dois mapas de atenção grandes, com os tokens da frase nos dois eixos, mostrando padrões diferentes e todo o canto superior direito em branco"><figcaption>Cada linha é uma posição prevendo. Cada coluna é para onde ela olhou.</figcaption></figure>

Duas coisas para reparar. Primeira: o canto de cima à direita é todo
branco. É a máscara funcionando, no modelo de verdade. Segunda: "Capitú"
aparece cortado em quatro tokens (`·C`, `ap`, `it`, `ú`), porque um
vocabulário de 1.024 não tem espaço para nomes próprios. Os pedaços de uma
palavra costumam se olhar entre si, e é isso que remonta a palavra.

Dá para medir o que cada camada faz. Numa frase de 34 tokens, a média de
quão longe cada cabeça olha:

| Camada | Distância média | Peso na palavra anterior |
|---|---|---|
| 1 | 7,4 tokens | 9% |
| 2 | 3,2 tokens | 36% |
| 3 | 4,3 tokens | 25% |
| 4 | 4,7 tokens | 26% |

A primeira camada espalha o peso pela frase e quase não olha a palavra
imediatamente anterior. A segunda é a mais local, com mais de um terço do
peso na anterior. As de cima ficam no meio do caminho.

Ninguém programou essa divisão de trabalho: ela sai do treino.

## O buraco 2, enfim: a posição

Ficou uma dívida lá atrás. Nem Q, nem K, nem V, nem a máscara, nem as
quatro cabeças resolveram o saco de palavras: embaralhe a frase e a conta
dá o mesmo. Todas essas peças mexem em *quem olha para quem*, e nenhuma
delas sabe *onde* cada palavra está.

A correção precisa entrar antes dos produtos escalares, mexendo nos
vetores. A solução moderna se chama **RoPE** (*rotary position embedding*,
embutimento rotativo de posição): gire a pergunta e a etiqueta de cada
palavra por um ângulo proporcional à posição dela.

<figure><img src="../assets/aula-10/rope.png" alt="À esquerda quatro setas saindo da origem, a mesma palavra nas posições 0 a 3, cada uma girada um pouco mais; à direita uma curva da parecença média entre duas posições, que começa em 1 e cai com a distância"><figcaption>Duas palavras próximas giram ângulos parecidos, e continuam parecidas. Longe, a parecença cai.</figcaption></figure>

O giro trabalha em pares de números. Cada par `(a, b)` do vetor gira por
um ângulo `θ`, que é a posição vezes uma velocidade:

$$(a',\; b') = (a\cos\theta - b\sin\theta,\;\; a\sin\theta + b\cos\theta), \qquad \theta = p \cdot \omega$$

| Símbolo | Significado |
|---|---|
| `a`, `b` | um par de números do vetor |
| `p` | a posição da palavra na janela (0, 1, 2...) |
| `ω` | a velocidade de giro daquele par; cada par tem a sua |
| `θ` | o ângulo, em radianos |

Exemplo: o par `(1, 0)`, com velocidade `ω = 0,5`, na posição 2. O ângulo
é `θ = 2 × 0,5 = 1`. O par girado é `(cos 1, sen 1) = (0,54, 0,84)`. Na
posição 0, não gira nada.

Girar tem uma propriedade que serve à perfeição aqui: quando você gira
dois vetores e calcula o produto entre eles, o que sobra é a **diferença**
entre os dois ângulos. Uma palavra na posição 3 comparada com outra na
posição 1 dá o mesmo produto que as mesmas palavras nas posições 13 e 11.
O produto passa a depender da distância entre as palavras, e não da
posição de cada uma. O modelo trata igual uma frase que aparece no começo
e no fim do texto.

Cada par gira numa velocidade diferente: os primeiros rápido, os últimos
devagar. Na média de todos os pares, palavras próximas continuam parecidas
e palavras distantes ficam menos parecidas, que é a curva da direita na
figura.

O RoPE entra em duas linhas da cabeça de atenção. O conteúdo (`v`) não
gira, porque ele é o que a palavra entrega, e não o que entra na comparação:

```python
q = girar(self.pergunta(x))
k = girar(self.etiqueta(x))
```

## A prova

Hora de medir se tudo isso vale a pena. Pegue o modelo do começo da aula,
que só olha o token atual, e acrescente **uma** camada de atenção:

```python
x = self.embutir(indices)
x = x + self.atencao(x)        # a palavra, mais o que ela viu para trás
logits = self.cabeca(x)
```

Treine os dois do zero, do mesmo jeito, por 600 passos cada, e meça a
perda na validação:

| Modelo | Perda na validação |
|---|---|
| Só o token atual (bigrama) | 4,38 |
| Com uma camada de atenção | **4,17** |

Mesmo treino, mesmos dados. A única diferença é olhar para trás, e ela
baixou a perda em 0,2. Na Aula 11 o modelo ganha mais três camadas e um
treino de meia hora, e essa diferença cresce muito.

## O preço da janela

Uma conta que explica muita notícia. Para uma janela de `n` tokens, a
atenção calcula `n × n` produtos escalares. Dobrar a janela quadruplica o
custo.

| Janela | Produtos por cabeça |
|---|---|
| 128 (o nosso) | 16.384 |
| 1.000 | 1 milhão |
| 100.000 | 10 bilhões |

É por isso que janela grande é caro, e por que cada modelo novo anuncia a
janela dele como se fosse uma conquista. É.

## Cola da aula

| Conceito | Em uma frase |
|---|---|
| Embutimento | troca cada token por uma linha de 128 números de uma tabela treinável |
| Logits | as pontuações cruas, uma por token, antes da softmax |
| `nn.Module` | `__init__` cria as camadas, `forward` diz o caminho |
| Laço de treino | lote, perda, `zero_grad`, `backward`, `step` |
| Passo zero | um modelo novo tem perda perto de `ln(1024) = 6,93` |
| Produto escalar | mede a parecença entre dois vetores |
| `√d` | segura o tamanho dos produtos para a softmax não saturar |
| Q, K, V | pergunta, etiqueta e conteúdo, de três tabelas de pesos |
| Máscara causal | o futuro vira `−∞` antes da softmax, e some |
| Cabeças | várias atenções em paralelo, juntadas e misturadas no fim |
| RoPE | gira Q e K pela posição; no produto sobra a distância |
| Custo da janela | `n × n` produtos: dobrar a janela quadruplica a conta |

## Materiais

- **Notebook desta aula, no Google Colab:** [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klein-natan/nanodegree-AI-Atitus/blob/main/notebooks/aula-10-atencao.ipynb)
- **Quiz de cada microaula:** [no fim desta página](#quizzes-das-microaulas), 5 perguntas depois de cada explicação, antes do exercício.
- **Travou num exercício?** [Colas dos exercícios](colas-dos-exercicios.md#aula-10-atencao), com dicas que você abre uma de cada vez.
- Slides desta aula: entregues em sala.
- Modelo treinado, usado nas figuras: [`mini_llm.pt`](https://github.com/klein-natan/nanodegree-AI-Atitus/blob/main/data/mini_llm.pt)

Os quatro exercícios do notebook:

| Exercício | O que você constrói |
|---|---|
| 10.1 | o primeiro modelo em PyTorch e o laço de treino |
| 10.2 | a conta da atenção, à mão, com os números desta página |
| 10.3 | uma cabeça de atenção com máscara, e o teste da cola |
| 10.4 | o giro, quatro cabeças, e a prova contra o bigrama |

## Para ir além

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): a explicação visual mais conhecida, com animações de cada passo.
- [Attention in transformers, step-by-step (3Blue1Brown)](https://www.youtube.com/watch?v=eMlx5fFNoYc): a conta desta aula, animada. Em inglês, com legendas.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): o artigo de 2017 que apresentou o transformer. Difícil, mas vale ver o desenho da página 3.

## Quizzes das microaulas

Depois de cada microaula, e antes do exercício no Colab, responda o quiz
dela. As perguntas aparecem uma de cada vez, como uma pilha de cartas:
responda, leia a explicação e passe para a próxima. No fim aparece quantas
você acertou. Nada fica gravado: dá para refazer quantas vezes quiser.

### Microaula 1: o primeiro modelo em PyTorch

{% pilha titulo="Microaula 1 · O primeiro modelo" %}

```quiz
O que faz a camada nn.Embedding(1024, 128)?
- Transforma cada palavra em uma letra.
+ Troca cada token por uma linha de 128 números de uma tabela que o treino ajusta.
- Divide o texto em 128 pedaços.
- Calcula a perda do modelo.
? É o embutimento: 1.024 linhas, uma por token, com 128 números cada. São 131.072 pesos, ajustados como quaisquer outros.
```

```quiz
Numa classe do PyTorch, para que servem o __init__ e o forward?
- O __init__ treina o modelo, e o forward mede a perda.
- Os dois fazem a mesma coisa.
- O __init__ desenha o gráfico, e o forward imprime o resultado.
+ O __init__ cria as camadas, e o forward diz o caminho que os números percorrem.
? É a troca do Keras para o PyTorch: em vez de uma lista de camadas, uma classe em que você escreve o caminho.
```

```quiz
No laço de treino, para que serve o zero_grad()?
- Para zerar os pesos do modelo.
- Para zerar a perda.
+ Para apagar o gradiente do passo anterior, que senão se soma ao novo.
- Para sortear um lote novo.
? O PyTorch acumula os gradientes de uma chamada para outra. Sem o zero_grad, cada passo carregaria a soma de todos os anteriores.
```

```quiz
Qual é a ordem certa das linhas do laço de treino?
- step, backward, perda, zero_grad
+ lote, perda, zero_grad, backward, step
- backward, lote, step, perda, zero_grad
- perda, step, lote, backward, zero_grad
? Sorteia o lote, calcula a perda, apaga o gradiente antigo, calcula o novo (backward) e dá o passo (step). É o que o model.fit do Keras fazia escondido.
```

```quiz
Um modelo recém-criado, ainda sem treino, deu perda 31. O que isso indica?
- Que ele já aprendeu muito.
- Que a perda dele é normal: todo modelo começa assim.
- Que o vocabulário tem 31 tokens.
+ Que há erro de montagem: um modelo novo devia dar perto de ln(1024) = 6,93.
? É o teste do passo zero. Um modelo que não sabe nada chuta igual entre os 1.024 tokens, e a perda de um chute é 6,93.
```

{% endpilha %}

### Microaula 2: a conta da atenção

{% pilha titulo="Microaula 2 · A conta" %}

```quiz
Por que "olhar só as últimas palavras" não basta para prever bem?
+ Porque a informação que importa pode estar longe, como "Capitú" 21 palavras antes da lacuna.
- Porque olhar as últimas palavras é caro demais.
- Porque as últimas palavras são sempre artigos.
- Porque o modelo não consegue olhar para trás.
? Era assim antes de 2017, e é o que o modelo do Exercício 1 faz. As palavras que decidem a próxima podem estar a dezenas de posições.
```

```quiz
E por que "somar todas as palavras anteriores por igual" também não serve?
- Porque a soma fica grande demais para o computador.
- Porque ela esquece o começo da frase.
+ Porque "de" e "uma" passam a pesar tanto quanto "Capitú", e a informação some na média.
- Porque a soma muda a ordem das palavras.
? A saída é deixar cada posição escolher quanto olhar para cada palavra, com pesos que o treino ajusta. Isso é a atenção.
```

```quiz
O produto escalar de "dorme" = (0, 1, 1, 0) com "gato" = (0, 3, 1, 0) é:
- 1
- 2
- 3
+ 4
? Multiplica casa com casa e soma: 0·0 + 1·3 + 1·1 + 0·0 = 4. Vetores que apontam para o mesmo lado dão produto alto.
```

```quiz
Por que a atenção divide os produtos pela raiz de d?
- Para os pesos somarem 100%.
+ Para os números não ficarem grandes demais e a softmax não virar tudo-ou-nada.
- Para trocar a ordem das palavras.
- Para o resultado ficar inteiro.
? Com muitas dimensões, os produtos crescem só por terem mais parcelas. Números grandes saturam a softmax, e aí o gradiente some.
```

```quiz
Na atenção sem pesos, "o gato dorme" e "dorme gato o" dão a mesma saída para "dorme". O que isso mostra?
- Que a conta está errada.
- Que "dorme" é a palavra mais importante.
- Que a atenção funciona perfeitamente.
+ Que nada na fórmula sabe onde cada palavra está: a frase virou um saco de palavras.
? É o buraco 2. Ele só é tapado na Microaula 4, com o RoPE, que gira os vetores conforme a posição.
```

{% endpilha %}

### Microaula 3: pergunta, etiqueta e conteúdo

{% pilha titulo="Microaula 3 · A cabeça de atenção" %}

```quiz
Na analogia da biblioteca, o que é a etiqueta (K)?
- O que a posição atual está procurando.
+ O que cada palavra anterior oferece, como a etiqueta na lombada de um livro.
- O texto que a palavra entrega quando é escolhida.
- O número da página onde a palavra aparece.
? A pergunta (Q) procura, a etiqueta (K) anuncia, e o conteúdo (V) é o que se lê do livro escolhido.
```

```quiz
De onde vêm a pergunta, a etiqueta e o conteúdo de uma palavra?
- Do dicionário de português.
- Da posição da palavra na frase.
- Do tokenizador.
+ Do vetor da palavra multiplicado por três tabelas de pesos diferentes (três nn.Linear).
? q = x·W_Q, k = x·W_K e v = x·W_V. O treino ajusta as três tabelas.
```

```quiz
Por que usar tabelas diferentes para Q e K tapa o buraco 1 (a palavra olhando para si mesma)?
+ Porque o produto da pergunta de uma palavra com a etiqueta dela mesma deixa de ser automaticamente o maior.
- Porque as tabelas apagam a palavra atual.
- Porque o modelo passa a ignorar a diagonal.
- Porque Q e K ficam sempre iguais a zero.
? Sem as tabelas, uma palavra comparada com ela mesma dá o comprimento ao quadrado, o maior valor possível. Com tabelas diferentes, o treino decide quem olha para quem.
```

```quiz
A softmax de (1, 2, −∞) é:
- (33%, 33%, 33%)
- (0%, 0%, 100%)
+ (27%, 73%, 0%)
- (50%, 50%, 0%)
? e elevado a menos infinito é zero. A posição do futuro some, e as outras duas dividem os 100% entre si.
```

```quiz
Um modelo sem máscara causal treina com perda perto de zero e depois gera lixo. Por quê?
- Porque a máscara deixa o treino mais lento.
- Porque o modelo decorou o tokenizador.
+ Porque no treino ele colava, olhando o token que devia prever, e na geração não existe futuro para copiar.
- Porque a perda perto de zero é sempre um bom sinal.
? É o erro do dia. O teste da cola pega esse erro: trocar só a última palavra não pode mudar a saída das posições anteriores.
```

{% endpilha %}

### Microaula 4: várias cabeças e a posição

{% pilha titulo="Microaula 4 · Várias cabeças" %}

```quiz
O nosso modelo usa 4 cabeças de 32 números em vez de uma cabeça de 128. O que se ganha?
- Velocidade: quatro cabeças custam quatro vezes menos.
+ Variedade: cada cabeça aprende a olhar para uma coisa diferente.
- Mais pesos: quatro cabeças têm quatro vezes mais.
- Nada: é só um jeito diferente de escrever.
? O custo é o mesmo de uma cabeça de 128. Uma cabeça pode olhar a palavra anterior, outra o sujeito, outra os pedaços da mesma palavra.
```

```quiz
Depois de juntar as 4 cabeças com torch.cat, para que serve a camada saida?
- Para descartar três das quatro cabeças.
- Para transformar os números em texto.
- Para aplicar a máscara.
+ Para misturar o que as quatro cabeças acharam.
? As quatro saídas de 32, lado a lado, voltam a ser 128. A camada saida, de 128 × 128, mistura essas quatro partes.
```

```quiz
O que o RoPE faz?
- Apaga as palavras distantes.
- Soma a posição ao número do token.
+ Gira a pergunta e a etiqueta por um ângulo que cresce com a posição.
- Ordena as palavras em ordem alfabética.
? No produto de dois vetores girados sobra só a diferença dos ângulos. A atenção passa a saber a distância entre as palavras, e a ordem volta a importar.
```

```quiz
Na prova da aula, o modelo só com o token atual teve perda 4,38, e o modelo com uma camada de atenção teve 4,17. O que mudou entre os dois?
+ Só a atenção: mesmo treino, mesmos dados, mesmo número de passos.
- O tokenizador.
- O tamanho do vocabulário.
- O texto de treino.
? Olhar para trás baixou a perda em 0,2. Na Aula 11, com quatro blocos e meia hora de treino, essa diferença cresce muito.
```

```quiz
Se a janela de contexto dobra de tamanho, o custo da atenção:
- Fica igual.
- Dobra.
- Cai pela metade.
+ Quadruplica.
? A atenção calcula n × n produtos. Com o dobro de tokens, são quatro vezes mais produtos. É por isso que janela grande custa caro.
```

{% endpilha %}
