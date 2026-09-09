---
description: Visão geral do módulo de LLM do Zero
---

# LLM do Zero

Todo mundo usa um modelo de linguagem grande (LLM). Poucos sabem como um
funciona por dentro. Ao longo destas quatro aulas, você constrói um do
zero em PyTorch, peça por peça, e treina no processador do seu próprio
computador. São 787.584 pesos, contra centenas de bilhões dos modelos
comerciais, e a arquitetura é a mesma.

O texto usado no treino é a obra de Machado de Assis, em domínio público.

As quatro aulas seguem a ordem de construção de *Build a Large Language
Model (From Scratch)*, de Sebastian Raschka, adaptada ao tempo de aula e a
quem está vendo isso pela primeira vez. Três hábitos vieram de lá, e valem
para muito além de LLM:

1. **Você sempre sabe onde está.** As quatro aulas abrem com o mesmo mapa,
   mudando só o quadro aceso.
2. **A versão simples primeiro.** Cada peça entra numa forma que funciona e
   que dá para fazer à mão, e só depois ganha o que falta. A atenção começa
   sem peso nenhum; o modelo começa como um esqueleto que roda e gera lixo.
3. **Medir em vez de afirmar.** Toda vez que caberia dizer "isto ajuda",
   mostramos o número.

| Aula | Conteúdo | Status |
|---|---|---|
| [Aula 9 — Do Texto aos Números](aula-09-tokenizacao.md) | O que um LLM faz, e o tokenizador BPE que ele usa | Disponível |
| [Aula 10 — Atenção](aula-10-atencao.md) | Como o modelo decide para onde olhar | Disponível |
| [Aula 11 — O Transformer e o Treino](aula-11-transformer-treino.md) | Montar o modelo inteiro e treiná-lo no processador | Disponível |
| [Aula 12 — Geração e Limites](aula-12-geracao-limites.md) | Do logit ao texto, e o que o modelo não sabe fazer | Disponível |
