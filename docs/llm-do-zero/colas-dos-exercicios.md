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

Os exercícios desta aula são experimentos com o modelo treinado do
professor. Quase nenhum tem resposta única: onde aparece `____`, você
escreve uma frase, um palpite ou uma palavra sua. As funções prontas
(`vizinhos`, `mapa_3d`, `para_onde_olha`, `apostas`, `estranheza`,
`chances`, `gerar`) vêm da célula de preparação.

### Exercício 10.1

Uma frase sua vira exemplos de treino, e o sentido das palavras vira
geometria.

**1.a Uma frase sua vira exemplos de treino**

<details><summary>Dica 1: a ideia</summary>

Qualquer frase serve: escreva entre as aspas, no lugar do `____`. Repare
na tabela: em cada linha, o modelo lê um pedaço maior do começo e tem que
prever o token seguinte.

</details>

<details><summary>Exemplo</summary>

```python
frase = "Hoje eu vou ao cinema com meus amigos."
```

Se aparecer `�`, é uma letra acentuada que o BPE guardou em dois bytes
separados: cada byte sozinho não é uma letra.

</details>

**1.b Palpite: quem fica perto de quem?**

<details><summary>Dica 1: a ideia</summary>

Não existe palpite errado aqui: o ponto é comparar o que você achou com o
que o modelo achou. Pense em que palavra aparece nos mesmos lugares que
"elle" nos livros de Machado.

</details>

<details><summary>Se a palavra "vira mais de um token"</summary>

Só palavras comuns ganharam um token próprio no BPE. Escolha uma mais
comum (artigos, pronomes, verbos como "era" e "foi", nomes dos
personagens).

</details>

**1.c Um grupo seu no mapa 3D**

<details><summary>Dica 1: a ideia</summary>

Quatro palavras do mesmo tipo, cada uma entre aspas: tempo ("dia",
"noite"), lugares ("casa", "rua", "porta"), família ("mãe", "pae",
"filha", "marido").

</details>

<details><summary>Exemplo</summary>

```python
meu_grupo = ["dia", "noite", "dias", "annos"]
```

</details>

### Exercício 10.2

Para onde o modelo olha, e a máscara e a posição postas à prova.

**2.a Para onde um substantivo olha**

<details><summary>Dica 1: a ideia</summary>

Escreva uma frase com "um" ou "uma" antes de um substantivo, e repita o
substantivo, sozinho, na segunda linha. Olhe a camada 4.

</details>

<details><summary>Exemplo</summary>

```python
frase = "Ella trazia uma carta na mão."
palavra = "carta"
```

Se aparecer "Não achei a palavra na frase", confira se ela está escrita
igualzinho nas duas linhas.

</details>

**2.b A máscara: o futuro não muda o passado**

<details><summary>Dica 1: a ideia</summary>

Troque só o `____` por uma palavra qualquer ("cachorro", "rei", "outro").
Depois compare as duas tabelas linha por linha até o "do".

</details>

**2.c A ordem importa**

<details><summary>Dica 1: a ideia</summary>

Use as mesmas palavras da frase certa, com a mesma grafia e a mesma
pontuação grudada nelas ("hontem." com o ponto), só em outra ordem.

</details>

<details><summary>Exemplo</summary>

```python
embaralhada = "senhora sabe Não hontem. que se a não elle veio sei"
```

</details>

### Exercício 10.3

O modelo inteiro em Keras, e a diferença que o treino faz.

**3.b Do tamanho de um GPT-2**

<details><summary>Resposta</summary>

```python
grande = montar_modelo(camadas=12, dimensao=768, cabecas=12)
```

</details>

**3.d Complete a frase**

<details><summary>Dica 1: a ideia</summary>

Para o favorito passar de 50%, comece um nome conhecido pela metade
("Quincas Bor", "Dom Cas"). Para ninguém passar de 10%, termine o começo
num lugar em que muita coisa cabe ("Não sei se a senhora").

</details>

Se a célula 3.c ou a 3.d der erro ao carregar os pesos, rode a célula
3.a de novo: o `montar_modelo` precisa existir antes.

## Aula 11 — Treinar e Usar um LLM

Como na Aula 10, os exercícios são experimentos: onde aparece `____`, você
escreve um começo de frase, um número ou uma frase sua. As funções prontas
(`montar_modelo`, `treinar`, `desenhar_curva`, `gerar`, `ajuste_fino`,
`estranheza`) vêm da célula de preparação. Hoje tem um treino de 35
minutos no fim do Exercício 2: não feche a aba enquanto ele roda.

### Exercício 11.1

O treino acontecendo, e o que o tamanho do passo faz com ele.

**1.a Um minuto de treino**

<details><summary>Dica 1: a ideia</summary>

Escreva o começo de uma frase entre as aspas, como "A casa de" ou "Era uma
vez". A célula leva pouco mais de um minuto: na primeira vez, mais uns 30
segundos para transformar o livro em tokens.

</details>

**1.b O tamanho do passo**

<details><summary>Dica 1: a ideia</summary>

Escreva um número sem aspas, com ponto e não vírgula: `0.05` (bem maior que
o padrão) ou `0.00001` (bem menor). A célula treina dois modelos, um
minuto cada.

</details>

<details><summary>O que esperar</summary>

Com 0.05, a perda empaca perto de 5,8: o passo é tão grande que pula por
cima do lugar bom. Com 0.00001, ela quase não sai de 6,9. O padrão, 0.001,
chega perto de 5,1 no mesmo minuto.

</details>

### Exercício 11.2

Os dois botões da geração, e o treino de verdade.

**2.a A temperatura**

<details><summary>Dica 1: a ideia</summary>

Um número sem aspas, entre 0.2 e 1.5. Rode algumas vezes com valores
diferentes (0.5, 0.8, 1.0) e compare.

</details>

<details><summary>O que esperar</summary>

Com 0.2, o texto entra em loop ("e ahi a pouco, e ahi a pouco"). Com 1.5,
aparecem palavras que não existem. Entre 0.7 e 0.9 costuma ficar melhor:
é o padrão da maioria das ferramentas.

</details>

**2.b O top-k salva a temperatura alta**

<details><summary>Dica 1: a ideia</summary>

Um número inteiro, como 10 ou 40. Com k = 1, só o favorito entra no
sorteio: é a escolha gulosa, sempre o mesmo texto.

</details>

**2.c O treino de verdade**

Não há lacuna. Se a célula terminou em menos de 35 minutos sem você
apertar nada, olhe a última linha: um erro ali quer dizer que alguma
célula de cima não rodou. Se o Colab desconectou no meio, o
`meu_llm.weights.h5` guarda o melhor modelo até a última conferência.

### Exercício 11.3

O seu modelo treinado, e depois ajustado.

**3.b Ajuste fino**

<details><summary>Dica 1: a ideia</summary>

Uma frase em português de hoje, sobre o que você viu no curso, com
palavras como modelo, dados, treino, perda, rede.

</details>

<details><summary>Exemplo</summary>

```python
minha_frase = "A rede neural aprende com os dados de treino."
```

</details>

Se o Colab reiniciou e o `meu_modelo` sumiu, rode a preparação e
`meu_modelo = montar_modelo()` e `meu_modelo.load_weights("meu_llm.weights.h5")`
antes do ajuste. Sem o arquivo, use o modelo do professor:
`meu_modelo = professor`.
