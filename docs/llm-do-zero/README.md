---
description: Visão geral do módulo de LLM do Zero
---

# LLM do Zero

Todo mundo usa um modelo de linguagem grande (LLM). Poucos sabem como um
funciona por dentro. Ao longo destas três aulas, você monta um em Keras,
peça por peça, e treina no processador do Google Colab, sem placa de
vídeo. São 809.728 pesos, contra centenas de bilhões dos modelos
comerciais, e a arquitetura é a mesma.

O texto usado no treino é a obra de Machado de Assis, em domínio público.

## Como as aulas funcionam

Cada aula alterna uma explicação curta, de 20 a 30 minutos, e um exercício
que você faz sozinho no notebook, de 20 a 25 minutos. Entre os dois, um
quiz de 5 perguntas, no fim da página da aula, confere se a explicação
assentou. São três rodadas por aula:

| Aula | Os três exercícios |
|---|---|
| 9 | o modelo de contagem, o BPE, codificar e decodificar |
| 10 | o token vira vetor, a atenção com máscara, o modelo inteiro |
| 11 | o treino pelo relógio, gerar texto (e ligar o treino de 35 minutos), o ajuste fino |

No fim da Aula 10, você montou o modelo e viu ele escrever com os pesos do
professor. No fim da Aula 11, os pesos são seus: você apertou o botão do
treino, ouviu o seu modelo escrever e ensinou a ele outro assunto.

Os exercícios têm lacunas (`____`) e uma célula de conferência depois de
cada passo. Se travar, a página
[Colas dos Exercícios](colas-dos-exercicios.md) tem dicas para cada passo,
que você abre uma de cada vez.

## Os três hábitos do módulo

As três aulas seguem a ordem de construção de *Build a Large Language
Model (From Scratch)*, de Sebastian Raschka, adaptada ao tempo de aula e a
quem está vendo isso pela primeira vez. Três hábitos vieram de lá, e valem
para muito além de LLM:

1. **Você sempre sabe onde está.** As três aulas abrem com o mesmo mapa,
   mudando só o quadro aceso.
2. **Direto ao que importa.** Cada peça entra na forma que os modelos de
   verdade usam, com as camadas prontas do Keras. A matemática aparece
   para explicar o que a camada faz, não para ser refeita à mão.
3. **Medir em vez de afirmar.** Toda vez que caberia dizer "isto
   funciona", mostramos o número: o passo zero contra os 6,93 do chute, a
   máscara que não deixa o passado mudar, a frase que o ajuste fino deixou
   de estranhar.

| Aula | Conteúdo | Status |
|---|---|---|
| [Aula 9 — Do Texto aos Números](aula-09-tokenizacao.md) | O que um LLM faz, o modelo de contagem e o tokenizador BPE | Disponível |
| [Aula 10 — Como Funciona um LLM](aula-10-como-funciona-um-llm.md) | O embedding, a atenção, o bloco transformer e o modelo inteiro | Disponível |
| [Aula 11 — Treinar e Usar um LLM](aula-11-treinar-e-usar.md) | O treino no processador, a geração de texto, o ajuste fino e os limites | Disponível |
| [Explorador do LLM](explorador.md) | O modelo do curso rodando no navegador: tokens, vetores, atenção e chances, com qualquer frase | Disponível |
| [Colas dos Exercícios](colas-dos-exercicios.md) | Dicas para cada passo dos 9 exercícios | Disponível |
