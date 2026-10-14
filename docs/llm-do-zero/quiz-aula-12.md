---
description: Um quiz de 5 perguntas depois de cada microaula da Aula 12
---

# Quizzes da Aula 12

{% hint style="info" %}
**Como usar esta página**

Depois de cada microaula, e antes do exercício no Colab, responda o quiz
dela. As perguntas aparecem uma de cada vez: responda, leia a explicação
e passe para a próxima. No fim aparece quantas você acertou.
{% endhint %}

## Microaula 1: do número à palavra

{% pilha titulo="Microaula 1 · Do número à palavra" %}

```quiz
Depois de "Não sei se a senhora", o token favorito do modelo tem 7,6% de chance. O que isso mostra?
- Que o modelo está com defeito.
+ Que a saída é uma distribuição espalhada, e escolher o token é um problema à parte.
- Que o modelo vai sempre escolher esse token.
- Que faltou treino nessa frase.
? Uma passada do modelo devolve 1.024 chances. São precisos 116 tokens para somar 90%. Escolher qual vai para a tela se chama amostragem.
```

```quiz
Para gerar texto, de qual posição o modelo usa as pontuações?
- Da primeira.
- Da do meio.
+ Da última, a que viu o texto inteiro.
- De todas, somadas.
? As outras posições servem ao treino. Para gerar, só interessa a previsão depois do último token: logits[0, -1].
```

```quiz
O que acontece quando o modelo escolhe sempre o token mais provável (decodificação gulosa)?
- O texto fica criativo e variado.
- O texto fica cheio de palavras inventadas.
- O modelo para de gerar.
+ Sai sempre o mesmo texto, e ele costuma entrar em laço, repetindo um pedaço.
? Um texto que só escolhe o topo acaba num lugar de onde o topo aponta de volta para si. No exemplo da aula: "--Não, senhor. --Não, senhor."
```

```quiz
Qual é o problema de sortear direto da distribuição do modelo, sem corte nenhum?
+ A cauda: centenas de tokens improváveis, somados, ganham com frequência.
- O texto sai sempre igual.
- O sorteio é lento demais.
- O modelo deixa de usar a janela de contexto.
? No exemplo da aula, os tokens fora dos 40 melhores somam 32% da chance. Quase uma vez em três sai algo que o próprio modelo considera ruim.
```

```quiz
Por que um token ruim no começo da geração estraga o resto?
- Porque o modelo apaga o texto anterior.
- Porque a temperatura sobe sozinha.
+ Porque ele entra de volta como texto, e o modelo precisa continuar coerente a partir do erro.
- Porque o tokenizador não reconhece mais as palavras.
? É a geração autorregressiva: cada token escolhido vira entrada do passo seguinte. Não existe voltar atrás.
```

{% endpilha %}

## Microaula 2: os botões

{% pilha titulo="Microaula 2 · Os botões" %}

```quiz
Com logits 4, 2 e 0 e temperatura 2, as pontuações viram:
- 8, 4 e 0
+ 2, 1 e 0
- 6, 4 e 2
- 4, 2 e 0
? A temperatura divide todas as pontuações antes da softmax. Com T = 2, as chances ficam 67%, 24% e 9%.
```

```quiz
O que a temperatura muda nos candidatos?
- A ordem deles: o último passa a ser o primeiro.
- O número de candidatos.
+ Só a distância entre as chances, sem mudar a ordem.
- O texto de entrada.
? Temperatura baixa concentra a chance no favorito; alta aproxima todos e engorda a cauda. A ordem nunca muda.
```

```quiz
O que o top-k faz com k = 40?
- Gera 40 tokens de uma vez.
- Divide as pontuações por 40.
- Escolhe sempre o 40º token.
+ Fica com os 40 tokens de maior pontuação e zera a chance de todos os outros.
? Os outros 984 viram menos infinito, como na máscara da Aula 10, e saem do sorteio.
```

```quiz
Com top-p = 0,9, o começo aberto "Não sei se a senhora" deixa 116 candidatos, e "Quincas Bor" deixa 1. Por quê?
+ Porque o top-p acompanha a dúvida do modelo: quando ele tem certeza ("ba", 98,6%), sobra pouca opção.
- Porque "Quincas Bor" é uma frase menor.
- Porque o top-p é igual ao top-k.
- Porque o modelo errou no segundo caso.
? O top-p fica com os mais prováveis até a soma passar de p. Em dúvida, muitos candidatos; com certeza, um só.
```

```quiz
Subir a temperatura deixa o modelo mais criativo?
- Sim, temperatura alta é sinônimo de criatividade.
- Sim, mas só acima de 2.
+ Não: o que sobe é a chance de escolher um token que o próprio modelo acha improvável.
- Não, a temperatura não muda nada.
? É o erro do dia. Depois de um token improvável, o texto tem que se justificar a partir dele. Para outra resposta, mude o começo.
```

{% endpilha %}

## Microaula 3: o modelo é o corpus

{% pilha titulo="Microaula 3 · O modelo é o corpus" %}

```quiz
Medindo o texto gerado, o maior trecho igual a Machado tem 4 palavras. O que isso diz?
- Que o modelo copia frases inteiras dos livros.
+ Que o modelo guarda uma estatística do texto, e não o texto: ele monta frases novas.
- Que o corpus é pequeno demais.
- Que o tokenizador cortou as frases.
? Com menos de 800 mil números, não cabem 3,7 milhões de caracteres. Modelos grandes, com textos muito repetidos na internet, conseguem reproduzir trechos longos.
```

```quiz
Por que um modelo de linguagem escreve frases falsas com toda confiança?
- Porque foi treinado para mentir.
- Porque a temperatura estava alta.
- Porque faltou um passo de treino.
+ Porque ele só estima o que é provável: frase falsa e frase verdadeira, se fluentes, são iguais para ele.
? Ele não tem noção de verdadeiro e falso, nem um lugar para consultar fatos. "Alucinação" é um nome ruim, porque sugere defeito.
```

```quiz
Por que o nosso modelo escreve "elle" e "cousa"?
+ Porque leu só textos de 1899, e o modelo é o que o corpus diz.
- Porque o tokenizador troca as palavras modernas.
- Porque a temperatura estava baixa.
- Porque é um erro de treino.
? A mesma lição vale para qualquer modelo: treinado em fórum escreve como fórum; em textos com preconceito, reproduz o preconceito.
```

```quiz
O que é o ajuste fino (fine-tuning)?
- Mudar a arquitetura do modelo.
- Ajustar a temperatura na hora de gerar.
+ Continuar o treino de um modelo pronto, por poucos passos, com outro texto.
- Treinar um tokenizador novo.
? No exercício, 200 passos com o texto do curso fazem o modelo de Machado falar de modelos e variáveis. Nenhuma peça muda: só os pesos.
```

```quiz
Depois do ajuste fino no texto do curso, a perda da frase "O modelo de linguagem prevê o próximo token do texto." caiu de 4,42 para 1,91. O que isso significa?
- Que a frase ficou mais curta.
- Que o modelo ficou maior.
- Que a frase passou a ter mais tokens.
+ Que o modelo ajustado acha essa frase de hoje muito mais provável.
? Perda menor, menos surpresa. Cada modelo acha mais provável a frase do próprio corpus, e o ajustado começou a estranhar Machado.
```

{% endpilha %}

## Microaula 4: do modelo ao assistente

{% pilha titulo="Microaula 4 · Do modelo ao assistente" %}

```quiz
O Llama 3 8B (2024) é vinte vezes menor que o GPT-3 (2020) e leu cinquenta vezes mais texto. O que isso mostra?
- Que modelos pequenos são sempre piores.
- Que o texto de treino não importa.
+ Que o texto virou a alavanca principal: um modelo menor, treinado por mais tempo, responde tão bem e custa menos para usar.
- Que o GPT-3 foi treinado em português.
? Pela regra das 6 contas, o custo cresce com pesos e texto juntos. Nos últimos anos, a balança pendeu para o texto.
```

```quiz
Qual é a ordem das três etapas até um assistente?
+ Pré-treino, ajuste por instrução, ajuste por preferência.
- Ajuste por preferência, pré-treino, ajuste por instrução.
- Tokenizador, atenção, temperatura.
- Ajuste por instrução, pré-treino, top-k.
? O pré-treino é o que fizemos. A instrução ensina o formato de pergunta e resposta. A preferência ensina qual resposta as pessoas preferem.
```

```quiz
Como o ajuste por instrução ensina o modelo a responder perguntas?
- Mudando a arquitetura para ter uma camada de resposta.
- Desligando a máscara causal.
+ É ajuste fino com milhares de conversas marcadas (quem pergunta, quem responde).
- Aumentando a temperatura.
? É o Exercício 3 com outro texto. Depois de ler muitas conversas no mesmo formato, o modelo aprende que, depois do marcador do assistente, vem uma resposta.
```

```quiz
O que é "escalar no momento de responder" (inference-time scaling)?
- Treinar o modelo de novo a cada pergunta.
- Usar uma placa de vídeo maior.
- Aumentar o vocabulário.
+ Gastar mais tokens na hora de responder (raciocínio, várias tentativas, revisão), sem mexer em peso nenhum.
? É por isso que os modelos de raciocínio mostram um "pensando". É o mesmo transformer, gerando mais antes de mostrar o resultado.
```

```quiz
No exercício, a "melhor de cinco" escolheu a resposta com mais espaços repetidos. O que isso ensina?
- Que espaços repetidos são sinal de boa resposta.
+ Que escolher pela perda do próprio modelo premia o previsível, e por isso os sistemas de verdade usam um juiz melhor.
- Que o modelo deveria gerar só espaços.
- Que a melhor de cinco nunca funciona.
? Espaço repetido é fácil de prever, então dá perda baixa. Os sistemas reais usam outro modelo treinado para dar nota, ou um teste que confere a resposta.
```

{% endpilha %}

## Para revisar

- [Aula 12 — Geração e Limites](aula-12-geracao-limites.md): a teoria completa.
- [Colas dos Exercícios](colas-dos-exercicios.md#aula-12-geracao-e-limites): as dicas de cada exercício.
