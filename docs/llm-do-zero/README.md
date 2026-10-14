---
description: Visão geral do módulo de LLM do Zero
---

# LLM do Zero

Todo mundo usa um modelo de linguagem grande (LLM). Poucos sabem como um
funciona por dentro. Ao longo destas quatro aulas, você constrói um do
zero em PyTorch, peça por peça, e treina no processador do Google Colab,
sem placa de vídeo. São 787.584 pesos, contra centenas de bilhões dos
modelos comerciais, e a arquitetura é a mesma.

O texto usado no treino é a obra de Machado de Assis, em domínio público.

## Como as aulas funcionam

Cada aula alterna uma explicação curta, de 20 a 25 minutos, e um exercício
que você faz sozinho no notebook, também de uns 20 minutos. Entre os dois,
um quiz de 5 perguntas no site confere se a explicação assentou. São quatro
rodadas por aula, e cada exercício acrescenta uma peça ao **mesmo** modelo:

| Aula | Os quatro exercícios |
|---|---|
| 9 | o modelo de contagem, o BPE, codificar e decodificar, o lote de treino |
| 10 | o primeiro modelo em PyTorch, a atenção à mão, a cabeça com máscara, várias cabeças |
| 11 | o bloco, o modelo inteiro, o laço de treino, e o treino de 35 minutos |
| 12 | gerar texto, temperatura e top-k, o ajuste fino, a melhor de cinco |

No fim da Aula 11, o modelo é seu: você escreveu cada parte e apertou o
botão do treino. Na Aula 12, você decide como ele escolhe as palavras e
descobre do que ele não é capaz.

Os exercícios têm lacunas (`____`) e uma célula de conferência depois de
cada passo. Se travar, a página
[Colas dos Exercícios](colas-dos-exercicios.md) tem dicas para cada passo,
que você abre uma de cada vez.

## Os três hábitos do módulo

As quatro aulas seguem a ordem de construção de *Build a Large Language
Model (From Scratch)*, de Sebastian Raschka, adaptada ao tempo de aula e a
quem está vendo isso pela primeira vez. Três hábitos vieram de lá, e valem
para muito além de LLM:

1. **Você sempre sabe onde está.** As quatro aulas abrem com o mesmo mapa,
   mudando só o quadro aceso.
2. **A versão simples primeiro.** Cada peça entra numa forma que funciona e
   que dá para fazer à mão, e só depois ganha o que falta. O modelo começa
   contando letras; a atenção começa sem peso nenhum.
3. **Medir em vez de afirmar.** Toda vez que caberia dizer "isto ajuda",
   mostramos o número: a perda dos bigramas que o transformer tem que
   bater, a atenção contra o modelo que não olha para trás, o gradiente com
   e sem atalho.

| Aula | Conteúdo | Status |
|---|---|---|
| [Aula 9 — Do Texto aos Números](aula-09-tokenizacao.md) | O que um LLM faz, o modelo de contagem e o tokenizador BPE | Disponível |
| [Aula 10 — Atenção](aula-10-atencao.md) | O primeiro modelo em PyTorch, e como ele decide para onde olhar | Disponível |
| [Aula 11 — O Transformer e o Treino](aula-11-transformer-treino.md) | Montar o modelo inteiro e treiná-lo no processador | Disponível |
| [Aula 12 — Geração e Limites](aula-12-geracao-limites.md) | Do logit ao texto, o ajuste fino, e o que o modelo não sabe fazer | Disponível |
| Quizzes das Aulas [9](quiz-aula-09.md), [10](quiz-aula-10.md), [11](quiz-aula-11.md) e [12](quiz-aula-12.md) | 5 perguntas depois de cada microaula, uma de cada vez | Disponível |
| [Colas dos Exercícios](colas-dos-exercicios.md) | Dicas para cada passo dos 16 exercícios | Disponível |
