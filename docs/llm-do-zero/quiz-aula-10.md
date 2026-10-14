---
description: Um quiz de 5 perguntas depois de cada microaula da Aula 10
---

# Quizzes da Aula 10

{% hint style="info" %}
**Como usar esta página**

Depois de cada microaula, e antes do exercício no Colab, responda o quiz
dela. As perguntas aparecem uma de cada vez: responda, leia a explicação
e passe para a próxima. No fim aparece quantas você acertou.
{% endhint %}

## Microaula 1: o primeiro modelo em PyTorch

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

## Microaula 2: a conta da atenção

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

## Microaula 3: pergunta, etiqueta e conteúdo

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

## Microaula 4: várias cabeças e a posição

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

## Para revisar

- [Aula 10 — Atenção](aula-10-atencao.md): a teoria completa.
- [Colas dos Exercícios](colas-dos-exercicios.md#aula-10-atencao): as dicas de cada exercício.
