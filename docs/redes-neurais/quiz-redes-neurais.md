---
description: Dez perguntas para conferir se você sabe ler uma rede neural, uma curva de treino e uma convolução
---

# Quiz — Redes Neurais

{% hint style="info" %}
**Como funciona**

São 10 perguntas sobre as Aulas 7 e 8. Nenhuma pede conta: todas pedem
para você **ler um resultado** e dizer o que ele significa. Clique na
opção que achar certa. A resposta e uma explicação aparecem na hora, e o
placar no fim da página soma os acertos. Não vale nota, e o botão
**Recomeçar** apaga tudo.
{% endhint %}

## Redes densas

```quiz
Nas 600 torras de café, a regressão logística acertou **71,8%**. Em 71,8% das torras o café sai ruim. Qual é a leitura certa?
- A logística é um bom modelo: acertou quase três em cada quatro torras.
+ Ela não aprendeu nada: responder "ruim" para todo mundo daria exatamente o mesmo número.
- Ela só precisa de mais torras para passar disso.
- A rede neural vai acertar os mesmos 71,8%, porque os dados são os mesmos.
? A fronteira que a logística sabe desenhar é uma reta, e nenhuma reta separa o dentro do fora de uma ilha. Ela empatou com o modelo preguiçoso da Aula 3. A rede com duas camadas de 16 neurônios chega a 91,8% com os mesmos dados: a diferença está no formato da fronteira.
```

```quiz
Você empilha 10 camadas escondidas, todas **sem função de ativação**, e deixa só a sigmoide na saída. Como fica a fronteira nas torras?
- Uma ilha, porque dez camadas têm dobras de sobra.
- Uma ilha grosseira, que melhora se você acrescentar mais camadas.
+ Uma reta, igual à da regressão logística.
- A rede não treina: o Keras recusa camadas sem ativação.
? A dobra vem da ativação. Sem ela, cada camada só multiplica e soma, e uma soma de somas pode ser reescrita como uma única conta, `w₀ + w₁x₁ + w₂x₂`. Mil camadas resolveriam o mesmo problema que a Aula 2 resolvia. Cada neurônio com **ReLU** acrescenta uma dobra à fronteira.
```

```quiz
Numa rede grande, a perda de **treino** cai sem parar durante 300 épocas. A de **validação** cai até a época 13 e, dali em diante, sobe. Qual é o diagnóstico?
- A rede ainda está aprendendo: falta treinar mais épocas.
- O conjunto de validação está errado, porque deveria cair junto com o de treino.
- A rede é pequena demais para o problema.
+ A rede passou a decorar o ruído das torras de treino, e o melhor momento foi a época 13.
? Quando a rede decora, a perda de treino sempre melhora, e é a de validação que conta a verdade. Neste exemplo ela foi de 0,213 na época 13 para 0,374 na 300, e a acurácia ficou em 96,7% no treino contra 87,8% na validação. Treinar mais só pioraria. As defesas são parar a tempo, o L2, o dropout ou uma rede menor.
```

```quiz
A mesma rede, agora com **dropout de 0,3**: a acurácia de treino caiu de 97% para 91%, e a de validação subiu de 88% para 91%. O modelo piorou?
+ Não: o freio tirou a decoreba, e a que importa é a acurácia de validação.
- Sim: a acurácia de treino caiu, então ele aprendeu menos.
- Sim, e a saída é diminuir o dropout até o treino voltar a 97%.
- Depende: só valeria a pena se o treino continuasse acima de 95%.
? Decorar infla a acurácia de treino e deixa a de validação para trás (97% contra 88%). Com o freio, as duas se aproximam (91% e 91%). A validação é a que mede torras que a rede nunca viu. O sinal de alerta seria o oposto: freio demais derruba **as duas**.
```

```quiz
Um `EarlyStopping(patience=15)` parou o treino na época 28, e a melhor perda de validação foi na 13. Sem `restore_best_weights=True`, com que pesos a rede fica?
- Com os da época 13, porque o callback sempre guarda o melhor.
+ Com os da época 28, quinze épocas depois do melhor ponto, um pouco piores.
- Com os da época 1, porque o treino foi interrompido.
- Com pesos aleatórios, e é preciso treinar de novo.
? Sem esse argumento, o Keras deixa a rede onde o treino parou. É o erro mais comum com o callback. Com `restore_best_weights=True`, a rede volta aos pesos da época 13. Para guardar o melhor modelo em arquivo, entra o `ModelCheckpoint`.
```

```quiz
Com a taxa de aprendizado em 0,01, a perda de validação balança até o fim do treino. Você acrescenta `ReduceLROnPlateau(factor=0.5, patience=5)`. O que muda?
- O Keras troca o otimizador por outro, mais lento.
- A rede ganha mais neurônios quando a validação trava.
+ Cada vez que a validação passa 5 épocas sem melhorar, o passo cai pela metade, e a curva assenta.
- O treino para depois de 5 épocas sem melhora.
? A taxa vai a 0,005, depois a 0,0025, e assim por diante, até o piso. No teste, a perda de validação final foi de 0,253 com a taxa fixa e de 0,224 com a redução. A última opção descreve o `EarlyStopping`, que **para** o treino. O `ReduceLROnPlateau` só encolhe o passo e continua, e por isso deve esperar menos épocas que o `EarlyStopping`.
```

## Redes convolucionais

```quiz
Você embaralha os 784 pixels de todas as fotos, sempre na mesma ordem. Uma pessoa não reconhece mais nada, mas a rede densa aprende **exatamente igual**. O que isso revela?
+ Que ela nunca usou o fato de os pixels serem vizinhos: depois do `Flatten`, a vizinhança já tinha sumido.
- Que a rede densa enxerga melhor que uma pessoa.
- Que embaralhar os pixels é um bom jeito de melhorar a rede.
- Que a rede densa só funciona com fotos embaralhadas.
? Ao achatar a foto, o pixel que ficava logo acima de outro vai parar 28 casas adiante na fila. Para a rede, os dois viraram entradas quaisquer, tão relacionadas quanto duas colunas de uma planilha. A convolução devolve o valor da vizinhança, porque cada conta usa só pixels que estão juntos.
```

```quiz
Uma camada com 16 filtros 3×3 tem **160 pesos**. A primeira camada da rede densa, com 128 neurônios olhando as mesmas fotos, tem **100.480**. Por que a convolução precisa de tão poucos?
- Porque ela olha só um pedaço da foto e descarta o resto.
- Porque os pesos dos filtros são menores e ocupam menos memória.
- Porque ela aprende menos detalhes que a densa.
+ Porque os mesmos nove pesos de cada filtro servem para a foto inteira, em todas as posições.
? É o **peso compartilhado**: 16 filtros × (9 pesos + 1 viés) = 160. Na densa, cada um dos 784 pixels se liga a cada um dos 128 neurônios, e isso dá 784 × 128 + 128 = 100.480. Compartilhar também faz o filtro achar o mesmo detalhe em qualquer lugar da foto.
```

```quiz
Você desloca as fotos de teste em **2 pixels**. A acurácia da rede densa vai de 87,6% para 51,5%, e a da convolucional, de 90,2% para 76,0%. Qual é a leitura certa?
- As duas pioraram, então são igualmente frágeis.
+ A densa decorou posições (como "o pixel 350 é escuro"), e a convolucional aprendeu detalhes (como "há uma borda vertical em algum canto"), que sobrevivem ao deslocamento.
- A convolucional é imune a deslocamentos, porque caiu menos.
- O teste é injusto, porque favorece a convolucional.
? Nenhuma das duas viu fotos deslocadas no treino, e as duas pioram. A diferença está na **velocidade da queda**. A convolucional não é imune: com 4 pixels de deslocamento, ela cai para 33,6%. A densa, para 13,6%.
```

```quiz
O modelo convolucional acerta 90,2% das fotos. A matriz de confusão mostra calça com 98,0% e camisa com 65,7%, e a camisa virou camiseta 132 vezes. O que você diz ao brechó?
- "Confie nos 90% em qualquer categoria."
- "O modelo é ruim e deve ser descartado."
+ "Ele é confiável em oito categorias e precisa de revisão humana em quatro: camisa, camiseta, pulôver e casaco."
- "Troque as camisas por camisetas no catálogo, porque o modelo as confunde."
? Acerto médio esconde tudo. O erro se concentra num canto só: quatro peças de tronco, todas com dois braços e uma gola, em 28 por 28 pixels e sem cor. Uma pessoa erraria as mesmas. A frase "confiável em oito, revisão em quatro" é útil, e só a matriz de confusão permite dizê-la.
```

## Onde revisar

| Se errou a questão | Releia |
|---|---|
| 1 | [Aula 7: por que a reta não basta](aula-07-redes-neurais-densas.md#por-que-a-reta-nao-basta) |
| 2 | [Aula 7: a dobra, funções de ativação](aula-07-redes-neurais-densas.md#a-dobra-funcoes-de-ativacao) |
| 3 e 4 | [Aula 7: quando a rede decora](aula-07-redes-neurais-densas.md#quando-a-rede-decora) |
| 5 | [Aula 7: EarlyStopping](aula-07-redes-neurais-densas.md#earlystopping-parar-quando-nao-melhora-mais) |
| 6 | [Aula 7: ReduceLROnPlateau](aula-07-redes-neurais-densas.md#reducelronplateau-encolher-o-passo-quando-trava) |
| 7 | [Aula 8: por que a rede da Aula 7 é ruim nisso](aula-08-redes-convolucionais.md#por-que-a-rede-da-aula-7-e-ruim-nisso) |
| 8 | [Aula 8: três ideias em uma](aula-08-redes-convolucionais.md#tres-ideias-em-uma) |
| 9 | [Aula 8: o teste do deslocamento](aula-08-redes-convolucionais.md#por-que-ela-ganha-o-teste-do-deslocamento) |
| 10 | [Aula 8: onde ela erra](aula-08-redes-convolucionais.md#onde-ela-erra) |
