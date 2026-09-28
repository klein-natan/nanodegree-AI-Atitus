---
description: Dez perguntas para conferir se você sabe ler e interpretar um modelo linear
---

# Quiz — Modelos Lineares

{% hint style="info" %}
**Como funciona**

São 10 perguntas sobre as Aulas 1, 2 e 3. Nenhuma pede conta: todas pedem
para você **ler um resultado** e dizer o que ele significa. Clique na
opção que achar certa. A resposta e uma explicação aparecem na hora, e o
placar no fim da página soma os acertos. Não vale nota, e o botão
**Recomeçar** apaga tudo.
{% endhint %}

## Regressão linear simples

```quiz
Um modelo prevê o preço de uma corrida por aplicativo com a reta **preço = 5 + 2,20 × distância**. O que o número 2,20 quer dizer?
- O preço de uma corrida de 0 km.
+ Cada quilômetro a mais soma R$ 2,20 ao preço previsto.
- O preço médio de todas as corridas.
- O erro médio da reta, em reais.
? O 2,20 é o peso `w₁`, o preço por quilômetro: ele diz quanto a previsão sobe a cada km a mais. O 5 é o `w₀`, a taxa fixa, que é o preço previsto para uma corrida de 0 km.
```

```quiz
Uma reta que prevê o preço das corridas tem **R² de 0,85**. Qual é a leitura correta?
- A reta acerta exatamente 85% das corridas.
- A reta erra, em média, 15% do preço de cada corrida.
- 85% das corridas ficam acima da reta.
+ A reta explica 85% da variação dos preços. O resto vem de coisas que ela não enxerga, como trânsito e horário.
? O R² compara a reta com o chute mais simples, que é prever o preço médio para todo mundo. Perto de 1, a reta explica quase toda a variação; perto de 0, ela não ajuda muito mais que a média. Ele não conta acertos exatos, e o erro em porcentagem é outra métrica (o MAPE).
```

```quiz
Dois modelos têm o **mesmo MAE**, de R$ 1,60 por corrida. O RMSE do modelo A é R$ 2,00, e o do modelo B é **R$ 6,50**. O que o RMSE alto do modelo B sugere?
+ O modelo B tem alguns erros bem grandes escondidos na média.
- O modelo B erra mais em todas as corridas.
- O modelo B é mais preciso, porque o RMSE dele é maior.
- Nada: com o mesmo MAE, os dois modelos são iguais.
? O RMSE eleva cada erro ao quadrado antes de tirar a média, então poucos erros grandes pesam muito nele. MAE igual e RMSE bem maior é o sinal de umas poucas previsões muito ruins. Vale investigar quais corridas são essas.
```

## Regressão múltipla

```quiz
Num modelo de aluguel com área, quartos, idade do prédio e bairro, o coeficiente de **quartos é +220,10**. O que esse número diz?
- Todo apartamento com mais quartos custa R$ 220,10 a mais, não importa o tamanho.
+ Um quarto a mais soma R$ 220,10 ao aluguel, comparando apartamentos de mesma área, idade e bairro.
- Cada quarto custa R$ 220,10 por mês para ser mantido.
- Quartos é a variável mais importante do modelo.
? Na regressão múltipla, cada coeficiente mede o efeito de uma unidade a mais daquela variável **com todas as outras paradas**. É por isso que o coeficiente de uma variável muda quando outra entra no modelo: a pergunta que ele responde muda.
```

```quiz
No mesmo modelo, o coeficiente do bairro **Jardins é +827,60**, e o da **área é +25,10** por metro quadrado. Dá para concluir que o bairro importa muito mais que a área?
- Sim: o coeficiente maior sempre indica a variável mais importante.
- Sim: variáveis de texto, como o bairro, sempre pesam mais.
- Não: os dois coeficientes são positivos, então importam igualmente.
+ Não: as unidades são diferentes. Vinte metros quadrados a mais já somam R$ 502, perto do efeito de trocar de bairro.
? Um coeficiente é o efeito de **uma unidade** da variável. O do bairro é o efeito de mudar de bairro; o da área é o efeito de um único metro quadrado. Coeficiente grande não quer dizer variável importante: pode querer dizer só que a unidade daquela variável é pequena.
```

```quiz
Depois de colocar dezenas de colunas novas no modelo de aluguel, o **R² no treino subiu** de 0,931 para 0,947, e o **R² no teste caiu** de 0,947 para 0,927. O que aconteceu?
- O modelo melhorou, porque o R² do treino subiu.
- O conjunto de teste está com problema e deveria ser trocado.
+ O modelo começou a decorar o ruído do treino e passou a prever pior os apartamentos novos.
- Nada importante: diferenças de R² tão pequenas não significam nada.
? Toda coluna nova faz o R² do treino subir ou ficar igual, mesmo que seja inútil. Quem mostra a verdade é o teste, com dados que o modelo nunca viu. Treino melhorando e teste piorando é o **sobreajuste**. A regularização, como o Lasso, é o freio contra ele.
```

## Regressão logística

```quiz
Num modelo de cancelamento de clientes, o coeficiente de **chamados ao suporte é +0,996**, e o de **tempo de casa é −0,079**. O que dá para concluir?
+ Mais chamados aumentam o risco de cancelar, e mais tempo de casa diminui.
- Cada chamado soma 99,6% de probabilidade de cancelar.
- O tempo de casa não importa, porque o coeficiente dele é quase zero.
- Os dois aumentam o risco, porque o modelo só soma.
? O sinal diz a direção: positivo empurra o risco para cima, negativo puxa para baixo. O tamanho não é uma porcentagem, porque a sigmoide não é uma reta. E −0,079 é o efeito de **um mês** de casa: ao longo de dois anos, ele pesa bastante.
```

```quiz
Numa base de clientes, **64% não cancelam**. Um modelo de cancelamento tem **acurácia de 75%**. O que dá para dizer?
- É um ótimo modelo: 75% é uma nota alta.
- É um modelo ruim: abaixo de 80%, nenhum modelo serve.
+ Ainda não dá para saber: um "modelo" que responde sempre "fica" já acerta 64%. É preciso olhar precisão, recall ou F1.
- O modelo encontra 75% dos clientes que cancelam.
? Quando uma resposta é bem mais comum que a outra, a acurácia engana. Compare sempre com a taxa da classe mais comum e olhe as métricas que contam os cancelamentos encontrados. Encontrar 75% dos que cancelam seria um recall de 0,75, que é outra coisa.
```

```quiz
A equipe de retenção reclama: "o modelo deixa passar muitos clientes que acabam cancelando". Qual métrica está baixa, e o que dá para fazer **sem treinar o modelo de novo**?
- A precisão. Dá para subir o limiar.
- A acurácia. Dá para trocar o modelo por um que responde sempre "fica".
- O R². Dá para colocar mais variáveis.
+ O recall. Dá para baixar o limiar, aceitando mais alarmes falsos.
? O recall responde "dos que cancelaram, quantos o modelo apontou?". Baixar o limiar põe mais gente na lista: o modelo encontra mais cancelamentos, mas também aponta mais gente que ficaria, e a precisão cai. Onde cortar é uma decisão de negócio.
```

```quiz
O modelo de cancelamento tem **AUC de 0,82**. Qual é a leitura correta?
+ Sorteando um cliente que cancelou e um que ficou, em 82% das vezes o modelo dá um risco maior ao que cancelou.
- O modelo acerta 82% dos clientes.
- 82% dos clientes vão cancelar.
- O melhor limiar para esse modelo é 0,82.
? A AUC mede se o modelo **ordena** bem os clientes: 0,5 é chute, 1 é perfeito. Ela não depende de limiar nenhum, e por isso não é uma taxa de acertos. Como ela só olha a ordem, um modelo com boa AUC ainda pode dar probabilidades exageradas.
```

## Onde revisar

| Se errou a questão | Releia |
|---|---|
| 1, 2 ou 3 | [Aula 1 — Regressão Linear Simples](aula-01-regressao-linear-simples.md) |
| 4, 5 ou 6 | [Aula 2 — Regressão Múltipla](aula-02-regressao-multipla.md) |
| 7, 8, 9 ou 10 | [Aula 3 — Regressão Logística](aula-03-regressao-logistica.md) |
