---
description: Dicas para destravar cada exercício das Aulas 9 a 12, uma de cada vez
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

1. **Você rodou a célula de preparação?** Nas Aulas 10, 11 e 12 a primeira
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
- `mat1 and mat2 shapes cannot be multiplied`: um `nn.Linear` recebeu um
  vetor do tamanho errado. Confira os dois números da camada.
- `The size of tensor a must match the size of tensor b`: você somou ou
  multiplicou dois tensores de formatos diferentes.
- `index out of range`: o índice passou do fim da lista ou do tensor.
- `'Tensor' object is not callable`: faltou o `self.` antes do nome da
  camada, ou você usou parênteses onde iam colchetes.
- **A célula roda para sempre**: se é treino, espere. Se não for, aperte o
  ■ e confira o laço.
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

Cada vez que o laço encontra um par, a contagem desse par aumenta em uma
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

Codificar e decodificar com o tokenizador oficial de 1.024 tokens.

**3.a Uma frase vira números**

<details><summary>Resposta</summary>

A frase já está guardada na variável `frase`:

```python
numeros = codificar(frase)
```

</details>

**3.b Cada número é um pedaço de texto**

<details><summary>Resposta</summary>

O laço entrega cada número na variável `numero`:

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

### Exercício 9.4

Do texto ao lote de treino, e o número a bater.

**4.a Treino e validação**

<details><summary>Dica 1: a ideia</summary>

O treino vai do começo até `corte`. A validação começa exatamente onde o
treino parou e vai até o fim.

</details>

<details><summary>Resposta</summary>

```python
dados_validacao = tokens[corte:]
```

</details>

**4.b Entrada e alvo**

<details><summary>Dica 1: a ideia</summary>

A entrada vai da posição 5000 até a 5008. O alvo é igual, andado uma
casa: some 1 nos dois números.

</details>

<details><summary>Resposta</summary>

```python
alvo = dados_treino[5001:5009]
```

</details>

**4.c O lote**

<details><summary>Resposta</summary>

Os números do slide: 32 janelas de 128 tokens.

```python
LOTE = 32
BLOCO = 128
```

</details>

**4.d A perda**

<details><summary>Dica 1: a ideia</summary>

As três chances são 50%, 25% e 10%. No Python, porcentagem se escreve
como fração: 50% é 0.50.

</details>

<details><summary>Resposta</summary>

```python
perda = (-math.log(0.50) - math.log(0.25) - math.log(0.10)) / 3
```

</details>

## Aula 10 — Atenção

### Exercício 10.1

O primeiro modelo em PyTorch, e o laço de treino.

**1.a Montar o modelo**

<details><summary>Dica 1: a ideia</summary>

A cabeça dá uma pontuação para cada token que pode vir em seguida. Quantos
tokens existem no vocabulário?

</details>

<details><summary>Resposta</summary>

```python
self.cabeca = nn.Linear(DIMENSAO, TAMANHO_VOCABULARIO)
```

</details>

**1.b O teste do passo zero**

<details><summary>Resposta</summary>

O chute ao acaso entre 1.024 tokens tem perda `ln(1024)`:

```python
perda_esperada = math.log(1024)
```

</details>

**1.c O laço de treino**

<details><summary>Dica 1: a ideia</summary>

O modelo só calcula a perda se souber a resposta certa, então ele recebe a
entrada e o alvo. O gradiente sai da perda, andando "para trás".

</details>

<details><summary>Dica 2: no código</summary>

"Para trás", em inglês, é *backward*. É um método da perda, então leva
parênteses.

</details>

<details><summary>Resposta</summary>

```python
logits, perda = bigrama(entrada, alvo)
otimizador.zero_grad()
perda.backward()
otimizador.step()
```

</details>

### Exercício 10.2

A conta da atenção, à mão, em PyTorch.

**2.a Parecença**

<details><summary>Dica 1: a ideia</summary>

Quem está perguntando é "dorme". Compare a frase inteira com o vetor
dela.

</details>

<details><summary>Resposta</summary>

```python
pontos = frase @ dorme
```

</details>

**2.b Raiz de d e softmax**

<details><summary>Resposta</summary>

Cada vetor tem quatro números, então `d = 4`:

```python
pontos_divididos = pontos / math.sqrt(4)
```

</details>

**2.c A mistura**

<details><summary>Dica 1: a ideia</summary>

A saída é uma mistura das próprias palavras, cada uma multiplicada pelo
seu peso. Onde estão as palavras?

</details>

<details><summary>Resposta</summary>

```python
saida_dorme = pesos @ frase
```

</details>

### Exercício 10.3

Uma cabeça de atenção de verdade, com máscara.

**3.a A máscara**

<details><summary>Dica 1: a ideia</summary>

A softmax de um número muito, muito negativo é zero. O mais negativo de
todos é menos infinito.

</details>

<details><summary>Resposta</summary>

```python
pontos_teste = pontos_teste.masked_fill(futuro, float("-inf"))
```

</details>

**3.b A cabeça inteira**

<details><summary>Dica 1: a ideia</summary>

Duas lacunas. A primeira: quais posições a máscara apaga? Você criou essa
tabela de verdadeiro e falso na linha de cima. A segunda: depois dos pesos,
o que entra na mistura: a pergunta, a etiqueta ou o conteúdo?

</details>

<details><summary>Dica 2: no código</summary>

A tabela da máscara se chama `futuro`. O conteúdo se chama `v`.

</details>

<details><summary>Resposta</summary>

```python
pontos = pontos.masked_fill(futuro, float("-inf"))
pesos = F.softmax(pontos, dim=-1)
return pesos @ v
```

</details>

**3.c O teste da cola**

Se apareceu "as posições anteriores mudaram", a máscara não está agindo.
Confira se a linha do `masked_fill` guarda o resultado de volta em
`pontos` (o `masked_fill` não altera o tensor original: ele devolve um
novo).

### Exercício 10.4

Posição, várias cabeças, e a prova de que a atenção ajuda.

**4.b A posição entra na cabeça**

<details><summary>Resposta</summary>

A etiqueta gira igual à pergunta. O conteúdo não gira:

```python
k = girar(self.etiqueta(x))
```

</details>

**4.c Quatro cabeças**

<details><summary>Dica 1: a ideia</summary>

Cada cabeça trabalha com uma fatia de 32 números, que já está calculada na
variável `tamanho_cabeca`. No `forward`, cada cabeça recebe o mesmo `x` e
devolve a sua parte.

</details>

<details><summary>Resposta</summary>

```python
lista.append(CabecaDeAtencao(tamanho_cabeca))
...
resultados.append(cabeca(x))
```

</details>

**4.d Atenção contra bigramas**

<details><summary>Resposta</summary>

O modelo soma ao vetor de cada token o que a atenção trouxe:

```python
x = x + self.atencao(x)
```

</details>

## Aula 11 — O Transformer e o Treino

### Exercício 11.1

O bloco transformer.

**1.a O MLP**

<details><summary>Dica 1: a ideia</summary>

A porta decide quanto passa, número a número. O que passa é o conteúdo.
Os dois se multiplicam.

</details>

<details><summary>Resposta</summary>

```python
return self.descida(porta * conteudo)
```

</details>

**1.b O bloco**

<details><summary>Dica 1: a ideia</summary>

A segunda linha tem o mesmo desenho da primeira: normaliza, faz alguma
coisa, soma de volta. Só trocam as peças.

</details>

<details><summary>Resposta</summary>

```python
x = x + self.mlp(self.norma2(x))
```

</details>

### Exercício 11.2

O modelo inteiro.

**2.a Montar o MiniLLM**

<details><summary>Dica 1: a ideia</summary>

Duas lacunas. Na primeira, a lista recebe um bloco novo a cada volta. Na
segunda, cada bloco transforma o `x` e devolve o `x` novo.

</details>

<details><summary>Dica 2: no código</summary>

Criar um bloco é chamar a classe: `Bloco()`. Usar um bloco é chamar o
objeto: `bloco(x)`.

</details>

<details><summary>Resposta</summary>

```python
lista.append(Bloco())
...
x = bloco(x)
```

</details>

Se o total de pesos saiu diferente de 787.584, a causa mais comum é uma
camada com tamanho trocado em alguma classe da preparação que você editou.
Rode a preparação de novo.

**2.b Perplexidade**

<details><summary>Resposta</summary>

A perplexidade é a exponencial da perda. O `.item()` tira o número de
dentro do tensor:

```python
perplexidade = math.exp(perda.item())
```

</details>

### Exercício 11.3

O laço de treino e o treino de 35 minutos.

**3.b A função de treino**

<details><summary>Dica 1: a ideia</summary>

São as três linhas do laço da Aula 10: o modelo recebe a entrada e o
alvo, a perda calcula o gradiente, o otimizador dá o passo.

</details>

<details><summary>Resposta</summary>

```python
logits, perda = modelo(entrada, alvo)
otimizador.zero_grad()
perda.backward()
torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
otimizador.step()
```

</details>

**3.c O treino de verdade**

Não há lacuna. Se a célula terminou em menos de 35 minutos sem você
apertar nada, olhe a última linha: um erro ali quer dizer que o laço tem um
problema, e o teste de 1 minuto do passo 3.b devia ter mostrado. Se o
Colab desconectou no meio, o `meu_llm.pt` guarda o melhor modelo até a
última conferência.

### Exercício 11.4

O modelo treinado.

**4.b Carregar o melhor modelo**

<details><summary>Resposta</summary>

O treino salvou o arquivo com o nome `meu_llm.pt`:

```python
modelo.load_state_dict(torch.load("meu_llm.pt"))
```

</details>

**4.d Levar o modelo para casa**

Se o download não começou, o navegador pode ter bloqueado: procure um
aviso na barra de endereço e permita. Outro caminho: no menu da esquerda
do Colab, clique no ícone de pasta, ache o `meu_llm.pt`, clique nos três
pontos e escolha **Fazer download**.

## Aula 12 — Geração e Limites

### Exercício 12.1

Do número ao texto.

**Preparação: carregar o modelo**

Se a janela de envio não abriu, rode a célula de novo. Se você não tem o
`meu_llm.pt`, clique em Cancelar: o notebook carrega o modelo do
professor, que tem o mesmo formato, e todos os exercícios funcionam igual.

**1.a O que o modelo devolve**

<details><summary>Dica 1: a ideia</summary>

O modelo devolve pontuações para todas as posições. Você quer só a da
última, a que viu o texto inteiro.

</details>

<details><summary>Resposta</summary>

```python
ultimos = logits[0, -1, :]
```

</details>

**1.b Sempre o mais provável**

<details><summary>Resposta</summary>

O token escolhido entra no fim do texto, para a próxima volta:

```python
indices.append(proximo)
```

</details>

**1.c Sortear**

<details><summary>Resposta</summary>

```python
proximo = torch.multinomial(probabilidades, 1).item()
```

</details>

### Exercício 12.2

Temperatura e top-k.

**2.a A temperatura**

<details><summary>Resposta</summary>

O laço já entrega cada temperatura na variável `temperatura`:

```python
p = F.softmax(ultimos / temperatura, dim=-1)
```

</details>

**2.b O top-k**

<details><summary>Dica 1: a ideia</summary>

A nota de corte já está calculada. Tudo o que tem pontuação menor que ela
sai do sorteio.

</details>

<details><summary>Resposta</summary>

```python
return pontuacoes.masked_fill(pontuacoes < corte, float("-inf"))
```

</details>

**2.c A função completa**

<details><summary>Resposta</summary>

Os dois botões chegam como argumentos da função:

```python
pontuacoes = logits[0, -1] / temperatura
pontuacoes = cortar_top_k(pontuacoes, top_k)
```

</details>

**2.e Desafio: top-p**

<details><summary>Resposta</summary>

Conta quantos tokens ficam abaixo de `p` na soma acumulada, mais o que faz
a soma passar:

```python
return (acumulada < p).sum().item() + 1
```

</details>

### Exercício 12.3

O modelo é o corpus.

**3.b O corpus novo**

<details><summary>Resposta</summary>

```python
dados_curso = torch.tensor(codificar(texto_curso))
```

</details>

**3.c O ajuste fino**

<details><summary>Dica 1: a ideia</summary>

É o laço da Aula 11. A única diferença é de onde vêm os lotes: do texto
novo, e não de Machado.

</details>

<details><summary>Resposta</summary>

```python
entrada, alvo = sortear_lote(dados_curso)
```

</details>

### Exercício 12.4

Gastar tempo em vez de peso.

**4.a Pontuar um texto**

<details><summary>Dica 1: a ideia</summary>

É a janela da Aula 9: o alvo é o texto andado uma casa. A entrada tira o
último token; o alvo tira o primeiro.

</details>

<details><summary>Resposta</summary>

```python
alvo = ids[:, 1:]
```

</details>

**4.b A melhor de cinco**

<details><summary>Resposta</summary>

Cada candidato entra na lista junto com a perda dele, e o `min` escolhe o de
menor perda:

```python
candidatos.append((pontuar(modelo, texto), texto))
```

</details>
