---
description: Um quiz de 5 perguntas depois de cada microaula da Aula 11
---

# Quizzes da Aula 11

{% hint style="info" %}
**Como usar esta página**

Depois de cada microaula, e antes do exercício no Colab, responda o quiz
dela. As perguntas aparecem uma de cada vez: responda, leia a explicação
e passe para a próxima. No fim aparece quantas você acertou.

Na Microaula 4 o seu treino de 35 minutos está rodando: responda o quiz
numa outra aba, sem fechar a do Colab.
{% endhint %}

## Microaula 1: o bloco transformer

{% pilha titulo="Microaula 1 · O bloco" %}

```quiz
Qual é o desenho das duas metades de um bloco transformer?
- Atenção, depois soma, depois normaliza.
- Só atenção, repetida duas vezes.
+ Normaliza, faz alguma coisa (atenção ou MLP), soma de volta à entrada.
- Embutimento, atenção, cabeça de saída.
? Em código: x = x + atencao(norma1(x)) e x = x + mlp(norma2(x)). O "x +" das duas linhas é o atalho.
```

```quiz
Qual frase descreve a divisão de trabalho dentro do bloco?
+ A atenção junta informação de outras posições; a camada densa pensa sobre essa mistura, uma posição de cada vez.
- A atenção pensa sobre cada posição; a camada densa junta as posições.
- As duas fazem a mesma coisa, só que em tamanhos diferentes.
- A atenção gera texto; a camada densa calcula a perda.
? A atenção é a única parte do modelo em que uma palavra vê outra. O MLP trabalha cada posição sozinha.
```

```quiz
Sem o atalho, o que acontece com o gradiente que chega ao primeiro bloco?
- Ele fica maior que o do último bloco.
- Ele fica exatamente igual.
- Ele vira negativo.
+ Ele chega quase mil vezes menor, e o primeiro bloco quase não aprende.
? É o gradiente que desaparece. Com o atalho, existe um caminho direto da saída até qualquer bloco, e todos recebem o mesmo tanto.
```

```quiz
O que a RMSNorm faz com um vetor?
- Soma 1 a cada número.
+ Divide cada número pelo tamanho típico do vetor (a raiz da média dos quadrados).
- Ordena os números do maior para o menor.
- Troca os números negativos por zero.
? Exemplo: (3, 4) tem média dos quadrados 12,5, cuja raiz é 3,54. O vetor vira (0,85, 1,13). É a normalização da Aula 7, sem subtrair a média.
```

```quiz
No SwiGLU, o que a "porta" faz?
- Fecha o modelo quando a perda sobe.
- Escolhe qual palavra vem depois.
+ Decide, número a número, quanto do conteúdo passa adiante.
- Separa o texto de treino do de validação.
? Um caminho carrega o conteúdo (subida); o outro, depois da SiLU, vira a porta. Os dois se multiplicam, e a descida volta para 128 números.
```

{% endpilha %}

## Microaula 2: o modelo inteiro

{% pilha titulo="Microaula 2 · O modelo inteiro" %}

```quiz
Onde está a maior parte dos 787.584 pesos do modelo?
- No embutimento.
- Na atenção.
+ No MLP, a camada densa de cada bloco.
- Na normalização.
? O MLP tem 393.216 pesos (50%), a atenção 262.144 (33%), o embutimento 131.072 (17%) e a normalização só 1.152.
```

```quiz
O que são os pesos amarrados?
+ Usar a mesma tabela no embutimento da entrada e na cabeça da saída.
- Pesos que o treino não pode mudar.
- Pesos que ficam iguais em todos os blocos.
- Os pesos da normalização.
? A entrada troca token por vetor; a saída troca vetor por pontuação de cada token. A mesma tabela faz os dois e economiza 131.072 pesos.
```

```quiz
Se a perda na validação é 3,40, qual é a perplexidade?
- 3,40
- 340
+ Cerca de 30
- 1.024
? Perplexidade é e elevado à perda: e^3,40 ≈ 30. O modelo está na dúvida entre uns 30 tokens, e não mais entre 1.024.
```

```quiz
O que quer dizer perplexidade 1?
- Que o modelo não sabe nada.
- Que o vocabulário tem um token só.
- Que o treino ainda não começou.
+ Que o modelo acerta sempre, o que só acontece se ele decorou.
? Perplexidade 1 é perda 0: certeza total no token certo. Em texto novo isso não acontece; se acontece no treino, é decoreba.
```

```quiz
Por que os pesos começam pequenos, sorteados perto de zero?
- Para o modelo ficar menor no disco.
- Para o tokenizador funcionar.
+ Para os números não crescerem bloco a bloco e a primeira perda não explodir.
- Porque o PyTorch não aceita pesos grandes.
? Cada bloco soma o que calculou ao atalho. Com pesos grandes, essa soma cresce a cada bloco. Com pesos pequenos, a perda do passo zero fica perto de 6,93.
```

{% endpilha %}

## Microaula 3: o laço de treino

{% pilha titulo="Microaula 3 · O treino" %}

```quiz
O que o AdamW tem a mais que o gradiente descendente da Aula 1?
+ Uma média dos gradientes recentes, passos ajustados por direção, e um empurrão leve dos pesos para o zero.
- Ele não usa gradiente nenhum.
- Ele treina sem precisar de perda.
- Ele só funciona em placa de vídeo.
? É o gradiente descendente com memória. O "W" (weight decay) segura o modelo contra decorar.
```

```quiz
Um lote estranho gerou um gradiente de tamanho 5. Com o corte em 1, o que acontece?
- O passo é cancelado.
- O gradiente vira 1 em todas as direções.
+ Cada número do gradiente é dividido por 5: o passo fica menor, na mesma direção.
- O lote é sorteado de novo.
? O corte limita o tamanho do passo sem mudar a direção. É um seguro barato contra um lote fora do normal.
```

```quiz
Por que o tamanho do passo começa pequeno (aquecimento)?
- Porque o computador ainda está esquentando.
+ Porque no começo os pesos são aleatórios, e um passo grande na direção errada só estraga.
- Porque a perda começa em zero.
- Porque o lote do começo é menor.
? No fim acontece o contrário: o modelo já está perto de um bom lugar, e passo grande passa do ponto. Por isso também existe o resfriamento.
```

```quiz
O treino da aula dura 35 minutos pelo relógio, e não um número fixo de passos. Qual é a vantagem?
- O modelo fica maior.
- A perda cai mais depressa.
- Todo mundo dá exatamente o mesmo número de passos.
+ O treino termina na hora certa em qualquer Colab, rápido ou lento.
? A taxa depende da fração do tempo que já passou. Quem tem uma máquina mais rápida simplesmente dá mais passos.
```

```quiz
A cada 200 passos, o treino salva o modelo em meu_llm.pt. Quando?
- Sempre, sobrescrevendo o anterior.
+ Só quando a perda na validação é a melhor até ali.
- Só quando a perda de treino sobe.
- Só no último passo.
? Assim o arquivo guarda o modelo que melhor generaliza. E se o Colab cair no meio, o melhor modelo já está salvo.
```

{% endpilha %}

## Microaula 4: enquanto o modelo treina

{% pilha titulo="Microaula 4 · Ler o treino" %}

```quiz
Olhando a curva de treino, onde a perda cai mais rápido?
+ Nos primeiros minutos; depois cada melhora custa mais.
- No meio do treino.
- Nos últimos minutos.
- A queda é igual o tempo todo.
? A perda despenca no começo e depois vai ficando cada vez mais difícil melhorar. É por isso que treinar modelos grandes é tão caro.
```

```quiz
Ao longo do treino, o que o modelo aprende primeiro?
- O sentido das frases.
- Os nomes das personagens.
+ As letras e a pontuação, depois as palavras curtas, depois frases gramaticais.
- A grafia moderna.
? A ordem é sempre a mesma: primeiro a forma, depois as palavras, e o sentido por último. É o que ele menos tem no fim.
```

```quiz
A perda de treino está bem abaixo da perda de validação. O que isso indica?
- Que o modelo está ótimo e pode treinar para sempre.
- Que a validação está com erro.
+ Que o modelo está decorando trechos do texto que viu.
- Que o tokenizador está errado.
? O número que vale é o da validação. Enquanto as duas caem juntas, tudo bem; quando a de validação para ou sobe, o resto é decoreba.
```

```quiz
Pela regra de bolso, quantas contas custa treinar um modelo?
- Uma conta por token.
- O número de pesos vezes 2.
- O número de tokens ao quadrado.
+ Umas 6 contas por peso, para cada token lido: 6 × N × T.
? 2 contas para ir (forward) e 4 para voltar com o gradiente (backward). O custo cresce com o tamanho do modelo e com o texto ao mesmo tempo.
```

```quiz
O que um modelo comercial tem de diferente do nosso, na arquitetura?
- Ele não usa atenção.
- Ele não usa tokenizador.
+ Nada de novo: mais camadas, vetores maiores, janela maior e muito mais texto.
- Ele usa uma perda diferente.
? A arquitetura é a mesma que você montou. É a mesma ideia, com mais dinheiro: meses de treino em milhares de placas.
```

{% endpilha %}

## Para revisar

- [Aula 11 — O Transformer e o Treino](aula-11-transformer-treino.md): a teoria completa.
- [Colas dos Exercícios](colas-dos-exercicios.md#aula-11-o-transformer-e-o-treino): as dicas de cada exercício.
