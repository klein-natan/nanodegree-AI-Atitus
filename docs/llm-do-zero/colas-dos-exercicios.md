---
description: Dicas para destravar cada exercício das Aulas 9 a 11, uma de cada vez
---

# Colas dos Exercícios

{% hint style="info" %}
**Como usar esta página**

Cada passo de exercício tem até três dicas, e você abre uma de cada vez.
A primeira lembra a ideia. A segunda mostra o caminho no código. A última
é a resposta: abra só depois de tentar as outras duas.

Usar a cola não é trapaça. Travar dez minutos numa vírgula ensina menos
que ver a linha certa e entender por que ela é certa.
{% endhint %}

## Antes de abrir qualquer dica

Metade dos travamentos não tem nada a ver com o exercício. Confira estas
quatro coisas primeiro:

1. **Você rodou a célula de preparação?** Nas Aulas 10 e 11 a primeira
   célula de código traz tudo das aulas anteriores. Sem ela, nada funciona.
2. **Você rodou as células de cima, na ordem?** O Colab só conhece o que
   já rodou. Uma função definida numa célula que você pulou não existe.
3. **Você trocou todos os `____`?** Uma lacuna esquecida dá
   `NameError: name '____' is not defined`.
4. **Você leu a última linha do erro?** É lá que o Python diz o que
   aconteceu. As primeiras linhas só mostram o caminho até o problema.

## Os erros mais comuns no Colab

Procure a última linha do erro nesta lista:

- `NameError: name 'x' is not defined`: uma célula de cima não rodou, ou o
  nome está escrito diferente.
- `NameError: name '____' is not defined`: sobrou uma lacuna sem
  preencher.
- `... is incompatible with the layer: expected axis -1 ...`: uma camada
  `Dense` recebeu um vetor do tamanho errado. Confira os números das
  camadas.
- `Incompatible shapes` ou `operands could not be broadcast together`:
  você somou ou multiplicou dois arrays de formatos diferentes.
- `index ... is out of bounds`: o índice passou do fim da lista ou do
  array.
- `object is not callable`: você usou parênteses onde iam colchetes, ou
  deu a uma variável o nome de uma função (por exemplo, `gerar = ...`).
- **A célula roda para sempre**: se é treino, espere. Se não for, aperte o
  ■ e confira o loop.
- **"Your session crashed after using all available RAM"**: rode de novo a
  partir da preparação, sem carregar o corpus duas vezes.

Se o ❌ apareceu e você não sabe por quê, leia a pista que vem com ele:
ela foi escrita para o erro mais comum daquele passo.

## Aula 9 — Do Texto aos Números

As células marcadas com 🧰 **Ferramentas** trazem código pronto: rode e
siga. As lacunas desta aula são sempre uma palavra ou um número.

### Exercício 9.1

O modelo de linguagem mais simples: contar que letra vem depois de qual.

**1.b O que vem depois do "q"?**

<details><summary>Dica 1: a ideia</summary>

A letra atual está na posição `i`. A letra seguinte está uma posição
depois.

</details>

<details><summary>Resposta</summary>

```python
letra_seguinte = corpus[i + 1]
```

</details>

**1.c Da contagem para a chance**

<details><summary>Dica 1: a ideia</summary>

Chance é "quantas vezes aconteceu" dividido por "quantas vezes podia ter
acontecido". O segundo número já está numa variável, calculada na linha
de cima.

</details>

<details><summary>Resposta</summary>

```python
chance_do_u = depois_do_q["u"] / total
```

</details>

**1.d O modelo escreve**

<details><summary>Resposta</summary>

Qualquer letra entre as aspas serve. Por exemplo:

```python
texto_gerado = gerar_letras("A", 300)
```

</details>

### Exercício 9.2

O BPE no computador: contar pares, fundir o mais comum, repetir.

**2.b Conte os pares vizinhos**

<details><summary>Dica 1: a ideia</summary>

Cada vez que o loop encontra um par, a contagem desse par aumenta em uma
unidade.

</details>

<details><summary>Resposta</summary>

```python
pares[par] += 1
```

</details>

**2.c Uma fusão**

<details><summary>Dica 1: a ideia</summary>

Os números de 0 a 255 já são os bytes. O primeiro número livre para uma
peça nova vem logo depois.

</details>

<details><summary>Resposta</summary>

```python
ids = fundir(ids, par, 256)
```

Se a verificação falhar, rode de novo a célula 2.a antes: ela volta a
lista para o texto original.

</details>

**2.d Mais duas fusões**

<details><summary>Dica 1: a ideia</summary>

O par mais comum sai da contagem que a linha de cima acabou de fazer.
Como ela se chama?

</details>

<details><summary>Resposta</summary>

```python
par = par_mais_comum(pares)
```

</details>

### Exercício 9.3

Codificar e decodificar com o tokenizador oficial.

**3.a Uma frase vira números**

<details><summary>Resposta</summary>

A frase já está guardada na variável `frase`:

```python
numeros = codificar(frase)
```

</details>

**3.b Cada número é um pedaço de texto**

<details><summary>Resposta</summary>

O loop entrega cada número na variável `numero`:

```python
print(numero, "->", mostrar(numero))
```

</details>

**3.c E de volta**

<details><summary>Resposta</summary>

Decodifique a lista que saiu do 3.a:

```python
de_volta = decodificar(numeros)
```

</details>

**3.d O tokenizador tem sotaque**

<details><summary>Resposta</summary>

Qualquer palavra serve. Escolha uma moderna, para ver a diferença:

```python
minha_palavra = "smartphone"
```

</details>

**3.e O corpus inteiro vira números**

<details><summary>Dica 1: a ideia</summary>

Caracteres por token é o número de caracteres dividido pelo número de
tokens. A lista de tokens do corpus se chama `ids_corpus`.

</details>

<details><summary>Resposta</summary>

```python
caracteres_por_token = len(corpus) / len(ids_corpus)
```

</details>

## Aula 10 — Como Funciona um LLM

As células 🧰 trazem código pronto: o tokenizador, o lote e o modelo
treinado do professor (`professor`). As lacunas são uma palavra ou um
número.

### Exercício 10.1

Entrada e alvo, o lote passando pelo embedding, e os vizinhos de cada
token.

**1.a Entrada e alvo**

<details><summary>Dica 1: a ideia</summary>

O alvo é a mesma frase, andada uma casa: o primeiro alvo é o segundo
token.

</details>

<details><summary>Dica 2: no código</summary>

`ids[:-1]` é tudo menos o último. O alvo é tudo menos o **primeiro**:
começa no índice 1.

</details>

<details><summary>Resposta</summary>

```python
alvo = ids[1:]
```

</details>

**1.b O lote passa pelo embedding**

<details><summary>Dica 1: a ideia</summary>

A tabela tem uma linha para cada token que pode aparecer. Quantos tokens
tem o vocabulário da Aula 9?

</details>

<details><summary>Resposta</summary>

```python
embedding = keras.layers.Embedding(input_dim=TAMANHO_VOCABULARIO, output_dim=128)
```

`TAMANHO_VOCABULARIO` vale 1024. Escrever `1024` direto também dá certo.

</details>

**1.c e 1.d Os vizinhos e o espaço em 3D**

Não há lacuna. Se `vizinhos` avisar que a palavra vira mais de um token,
escolha outra: só palavras comuns ganharam um token próprio no BPE. E
lembre do espaço no começo (`" casa"`, e não `"casa"`): é assim que a
palavra aparece no meio do texto.

### Exercício 10.2

A atenção do Keras, a máscara, e para onde o modelo treinado olha.

**2.a Quatro cabeças**

<details><summary>Dica 1: a ideia</summary>

Cada vetor tem 128 números, e cada cabeça trabalha com 32 (`key_dim=32`).
Quantas cabeças de 32 cabem em 128?

</details>

<details><summary>Resposta</summary>

```python
atencao = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)
```

</details>

**2.b A máscara causal**

<details><summary>Dica 1: a ideia</summary>

O argumento liga ou desliga a máscara. Ele recebe um valor de verdadeiro
ou falso do Python.

</details>

<details><summary>Resposta</summary>

```python
saida, pesos = atencao(frase, frase, use_causal_mask=True, return_attention_scores=True)
```

</details>

**2.c Para onde o modelo treinado olha**

Não há lacuna. Mude o `camada=0` para 1, 2 ou 3 e compare os mapas. Se der
erro dizendo que `professor` não existe, rode a célula de preparação de
novo: é ela que carrega o modelo do professor.

### Exercício 10.3

O bloco, o modelo inteiro, o passo zero e os pesos do professor.

**3.a O bloco**

<details><summary>Dica 1: a ideia</summary>

Duas lacunas. A camada densa abre de 128 para 256 números e precisa
**voltar** para 128, senão não dá para somar com o `x`. E o bloco
devolve a entrada mais o que a segunda metade calculou: o atalho.

</details>

<details><summary>Dica 2: no código</summary>

A primeira metade do bloco já mostra o atalho: `x = x + h`. A segunda
metade faz o mesmo, só que no `return`.

</details>

<details><summary>Resposta</summary>

```python
h = keras.layers.Dense(DIMENSAO, name=f"volta{i}")(h)
return x + h
```

</details>

**3.b O modelo**

<details><summary>Dica 1: a ideia</summary>

Quantos blocos o modelo tem? E quantas pontuações a saída precisa dar,
uma para cada o quê?

</details>

<details><summary>Resposta</summary>

```python
for i in range(CAMADAS):
...
saida = keras.layers.Dense(TAMANHO_VOCABULARIO, name="saida")(x)
```

</details>

Se o total de pesos saiu diferente de 809.728, confira as duas lacunas do
passo 3.a: um `Dense` com tamanho trocado muda a conta inteira.

**3.c O passo zero**

<details><summary>Dica 1: a ideia</summary>

O modelo devolve pontuações cruas, sem softmax. A perda precisa saber
disso para fazer a softmax ela mesma.

</details>

<details><summary>Resposta</summary>

```python
modelo.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

</details>

**3.d Os pesos do professor no seu modelo**

Não há lacuna. Se a célula der erro ao carregar os pesos, o seu modelo
ficou diferente do modelo do professor: confira os nomes e os tamanhos das
camadas nos passos 3.a e 3.b, e rode as duas células de novo.

## Aula 11 — Treinar e Usar um LLM

A célula de preparação traz o tokenizador, o lote, o `montar_modelo` da
Aula 10 e o modelo treinado do professor (`professor`). Hoje tem um treino
de 35 minutos no fim do Exercício 2: não feche a aba enquanto ele roda.

### Exercício 11.1

A função de treino pelo relógio, testada por um minuto.

**1.b A função de treino**

<details><summary>Dica 1: a ideia</summary>

Duas lacunas. A validação é medida em lotes do texto guardado para isso,
que o modelo nunca vê no treino. E o "melhor" modelo é o de menor perda
na validação, que o Keras chama de `val_loss`.

</details>

<details><summary>Dica 2: no código</summary>

A preparação separou o texto em `dados_treino` e `dados_validacao`. O
`lotes(...)` recebe um dos dois.

</details>

<details><summary>Resposta</summary>

```python
salvar_o_melhor = keras.callbacks.ModelCheckpoint(
    "meu_llm.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
...
validation_data=lotes(dados_validacao), validation_steps=10,
```

</details>

Se o teste de 1 minuto deu ❌ com a perda acima de 6, confira se a célula
do `Relogio` (a 1.a) rodou: sem ela, a função de treino nem existe.

### Exercício 11.2

A função `gerar`, com temperatura e top-k, e o treino de verdade.

**2.a A última posição**

<details><summary>Dica 1: a ideia</summary>

`logits[0]` é a primeira (e única) frase. Dentro dela, você quer a última
posição. Em Python, o último item de uma lista tem índice `-1`.

</details>

<details><summary>Resposta</summary>

```python
ultimos = logits[0, -1]
```

</details>

**2.b A temperatura**

<details><summary>Dica 1: a ideia</summary>

A temperatura divide as pontuações antes do sorteio. O nome dela, dentro
da função, é o do parâmetro.

</details>

<details><summary>Resposta</summary>

```python
sorteado = keras.random.categorical(ultimos / temperatura, 1)
```

</details>

**2.c O top-k**

<details><summary>Dica 1: a ideia</summary>

`ops.top_k(pontuacoes, quantos)` devolve as maiores pontuações. Quantas?
O número que a função recebeu como parâmetro.

</details>

<details><summary>Resposta</summary>

```python
melhores, candidatos = ops.top_k(ultimos, k)
```

</details>

**2.d O treino de verdade**

Não há lacuna. Se a célula terminou em menos de 35 minutos sem você
apertar nada, olhe a última linha: um erro ali quer dizer que a função de
treino tem um problema, e o teste de 1 minuto do passo 1.b devia ter
mostrado. Se o Colab desconectou no meio, o `meu_llm.weights.h5` guarda o
melhor modelo até a última conferência.

### Exercício 11.3

O seu modelo treinado, escrevendo e depois ajustado.

**3.a A curva e o melhor modelo**

<details><summary>Dica 1: a ideia</summary>

O `ModelCheckpoint` do Exercício 1 salvou o melhor modelo num arquivo. O
nome do arquivo está na função `treinar`.

</details>

<details><summary>Resposta</summary>

```python
modelo.load_weights("meu_llm.weights.h5")
```

</details>

**3.c O ajuste fino**

<details><summary>Dica 1: a ideia</summary>

O treino é o mesmo. A única coisa que muda é o texto de onde saem os
lotes: o texto do curso, separado na célula de cima.

</details>

<details><summary>Resposta</summary>

```python
ajustado.fit(lotes(curso_treino), steps_per_epoch=50, epochs=4)
```

</details>

Se o Colab reiniciou e o `modelo` sumiu, rode a preparação, o
`modelo = montar_modelo()` e o `modelo.load_weights("meu_llm.weights.h5")`
de novo antes do ajuste. Sem o arquivo, use o modelo do professor:
`modelo = professor`.
