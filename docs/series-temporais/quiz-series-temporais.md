---
description: Dez perguntas para conferir se você sabe ler uma série temporal e avaliar uma previsão
---

# Quiz — Séries Temporais

{% hint style="info" %}
**Como funciona**

São 10 perguntas sobre as Aulas 5 e 6. Nenhuma pede conta: todas pedem
para você **ler um resultado** e dizer o que ele significa. Clique na
opção que achar certa. A resposta e uma explicação aparecem na hora, e o
placar no fim da página soma os acertos. Não vale nota, e o botão
**Recomeçar** apaga tudo.
{% endhint %}

## Ler uma série

```quiz
Há três anos, a cafeteria vende cerca de R$ 330 a mais aos sábados do que às segundas, toda semana. Que peça da série é essa?
- Tendência.
+ Sazonalidade.
- Resto.
- Vazamento de dados.
? Um padrão que se repete num intervalo fixo, aqui a semana, é **sazonalidade**. A tendência é o rumo de longo prazo (a venda média subiu de R$ 938 em 2022 para R$ 1.267 em 2024). O resto é o que sobra depois de tirar as duas, como o ruído e os feriados.
```

```quiz
Você calcula a média móvel das vendas com uma **janela de 7 dias**. O que acontece com o sobe e desce da semana?
- Fica mais forte, porque a média destaca os picos.
- Não muda: a média móvel só apaga o ruído.
+ Some, porque cada janela contém uma semana inteira, com todos os dias.
- Some, junto com a tendência de alta.
? Uma janela do tamanho do ciclo apaga aquele ciclo: toda janela de 7 dias tem um sábado, uma segunda e todos os outros dias, então o zigue-zague se cancela. A tendência continua aparecendo. Uma janela de 30 dias apagaria também o efeito do mês.
```

```quiz
Um colega separou treino e teste com o `train_test_split`, sorteando os dias, e conseguiu um erro muito menor que o da régua. Qual é a primeira suspeita?
- Ele encontrou um modelo excelente, e deve colocá-lo em produção.
- O teste ficou pequeno demais.
+ Vazamento de dados: o modelo treinou com dias vizinhos, e até posteriores, aos dias do teste.
- O sorteio só atrapalharia se a série tivesse feriados.
? Em série temporal, o corte é **sempre no tempo**: treino até uma data, teste depois dela. Sorteando os dias, o modelo estuda dezembro para prever novembro, cercado de vizinhos que já viu. A métrica sai ótima no teste e ruim na vida real.
```

## A régua e os primeiros modelos

```quiz
A sazonal ingênua (copiar a semana passada) erra R$ 63,78 por dia no teste. Um modelo novo erra **R$ 70** nos mesmos dias. O que concluir?
- O modelo novo é bom: R$ 70 é um erro pequeno perto de vendas de R$ 1.200.
- Os dois são equivalentes, porque a diferença é pequena.
+ O modelo novo ainda não paga o próprio trabalho: perde para uma regra que não usa modelo nenhum.
- O certo é comparar o modelo novo com a previsão ingênua, que erra R$ 121,28.
? A régua é a melhor previsão de referência, e todo modelo tem que vencê-la. Um erro parece pequeno ou grande dependendo do que você compara, e a comparação honesta é com a melhor regra simples disponível, não com a pior.
```

```quiz
A regressão com **tendência e dia da semana** errou R$ 131,63 por dia em dezembro. Acrescentando as variáveis indicadoras do **mês**, o erro caiu para R$ 60,21. Por quê?
- Mais colunas sempre deixam a regressão melhor.
+ A primeira versão não sabia que dezembro é baixa temporada, e errava para cima o mês inteiro.
- O mês corrigiu o erro do Natal.
- O mês substituiu a tendência, que estava errada.
? Sem o mês, a regressão não conhecia a sazonalidade anual (o inverno vende mais que o verão), então previa para dezembro um nível alto demais. O Natal só foi corrigido depois, com a coluna de feriado, que levou o erro a R$ 44,11. Cada peça do calendário conserta um erro diferente.
```

```quiz
O **ARIMA(1,0,0)** previu, para os dias do teste, valores que caíam dia após dia em direção a R$ 1.097, a média dos três anos. Qual número está causando isso?
- O `p = 1`, porque um dia do passado é pouco.
+ O `d = 0`: o modelo prevê o valor e supõe que a série oscila em torno de um nível fixo, mas ela sobe.
- O `q = 0`, porque o modelo não corrige os próprios erros.
- Nenhum: o ARIMA sempre volta para a média.
? O `d` responde se o modelo prevê o valor (0) ou a mudança de um dia para o outro (1). Com `d = 0`, a previsão volta para o nível médio, que mistura 2022 com 2024. Com `d = 1`, ele prevê a mudança e soma ao último valor, e a tendência deixa de ser problema. Regra prática: se o gráfico sobe ou desce, use `d = 1`.
```

## Prophet e TimesFM

```quiz
O Prophet, direto da caixa, errou R$ 64,54: empatou com a régua. Com a **lista de feriados**, caiu para R$ 43,73. O que essa diferença ensina?
- Que o Prophet precisa de mais dados de treino para funcionar.
- Que a régua é um modelo melhor que o Prophet.
+ Que o modelo não adivinha o calendário: o conhecimento do negócio é que virou o jogo.
- Que feriados são a peça mais importante de qualquer série.
? O Prophet estima tendência e sazonalidades sozinho, mas não tem como saber que 25 de dezembro é diferente dos outros dias. Com a lista, ele chega quase ao mesmo erro da regressão com calendário da Aula 5 (R$ 44,11), porque os dois somam as mesmas peças.
```

```quiz
A faixa de incerteza do Prophet promete cobrir **80%** dos dias. No teste, **75%** dos dias caíram dentro dela. Qual é a leitura certa?
+ Um resultado honesto: a faixa cumpriu a promessa de perto, e um dia em cada quatro ou cinco fica de fora mesmo.
- A faixa está errada, porque deveria ter acertado todos os dias.
- O modelo é ruim, porque errou 25% dos dias.
- A faixa garante que nenhum dia vai ficar abaixo da borda de baixo.
? Uma faixa de 80% **espera** deixar cerca de um dia em cada cinco de fora. Com só 28 dias de teste, 75% está dentro do esperado. E a faixa só conhece o que já aconteceu: uma greve ou uma obra na rua podem jogar a venda para fora dela.
```

```quiz
O TimesFM previu as vendas da cafeteria **sem nenhum treino** e bateu a régua. Como isso é possível?
- Ele baixou os dados da cafeteria durante o `from_pretrained`.
- Ele treina escondido quando você pede a previsão.
+ Ele foi pré-treinado com uma quantidade enorme de outras séries, e aprendeu padrões como semanas, anos e tendências.
- Ele copia a semana passada, como a sazonal ingênua.
? O TimesFM é um **modelo de fundação**: foi treinado uma vez, pelo Google, com milhões de séries de todo tipo. Por isso ele prevê uma série nova só lendo o passado dela, sem mudar nenhum peso. Isso se chama previsão **zero-shot**.
```

```quiz
Nos 27 dias comuns do teste, o TimesFM foi o melhor de todos. No Natal, previu R$ 1.173, e a cafeteria vendeu R$ 780. Qual é a correção mais simples?
- Treinar o TimesFM de novo com os dados da cafeteria.
- Entregar menos dias de passado ao modelo, para ele se concentrar em dezembro.
- Trocar o TimesFM pela régua, que erra menos no Natal.
+ Somar à previsão dele o efeito médio de um feriado, medido no treino (cerca de −R$ 390), só nos dias de feriado.
? O modelo traz o padrão; o calendário é você quem traz. Com uma coluna de feriado e um peso tirado de uma média do treino, a previsão do Natal vai para R$ 783, e o erro médio cai para R$ 41,24, o melhor resultado do módulo. Entregar menos passado piora o TimesFM: com 365 dias, ele erra R$ 61,72.
```

## Onde revisar

| Se errou a questão | Releia |
|---|---|
| 1 | [Aula 5: as três peças de qualquer série](aula-05-conceitos-series-temporais.md#as-tres-pecas-de-qualquer-serie) |
| 2 | [Aula 5: a média móvel](aula-05-conceitos-series-temporais.md#a-media-movel) |
| 3 | [Aula 5: o corte é no tempo](aula-05-conceitos-series-temporais.md#a-regra-de-ouro-o-corte-e-no-tempo) |
| 4 | [Aula 5: três previsões sem modelo](aula-05-conceitos-series-temporais.md#tres-previsoes-sem-modelo-nenhum) |
| 5 | [Aula 5: a regressão com o calendário](aula-05-conceitos-series-temporais.md#o-primeiro-modelo-uma-regressao-com-o-calendario) |
| 6 | [Aula 5: d, prever o valor ou a mudança](aula-05-conceitos-series-temporais.md#d-prever-o-valor-ou-a-mudanca) |
| 7 | [Aula 6: o Prophet contra a régua](aula-06-prophet.md#o-prophet-contra-a-regua) |
| 8 | [Aula 6: o intervalo de incerteza](aula-06-prophet.md#o-intervalo-de-incerteza) |
| 9 | [Aula 6: o TimesFM](aula-06-prophet.md#um-modelo-que-nunca-viu-esta-cafeteria-o-timesfm) |
| 10 | [Aula 6: juntando os dois mundos](aula-06-prophet.md#juntando-os-dois-mundos) |
