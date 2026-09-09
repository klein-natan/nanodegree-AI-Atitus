---
description: Por que o modelo é a parte fácil
cover: ../assets/fundamentos/Gemini_Generated_Image_y8w9b8y8w9b8y8w9.png
coverY: 0
---

# 1. Engenharia de Machine Learning

{% hint style="info" %}
**O que você leva desta página**

Um modelo que funciona no seu notebook não é um produto. Você vai ver o que
falta entre os dois, por que o código do algoritmo é só 5% do trabalho, e
por que sistemas de ML falham sem dar erro nenhum.
{% endhint %}

## Aprendizado de máquina, em uma frase

No **aprendizado de máquina**, o programa encontra padrões nos dados em vez
de seguir regras que alguém escreveu.

O exemplo é o da nossa Aula 1. Você quer prever o preço de um apartamento.
O caminho tradicional seria escrever centenas de regras sobre como o
bairro, a área e o número de quartos afetam o preço. Boa sorte.

O caminho do ML é outro: mostre ao programa alguns milhares de vendas
passadas e deixe ele achar a relação sozinho. Depois, aplique essa relação
a um apartamento novo, que ninguém precificou ainda.

## A diferença que muda tudo: probabilidade

Software tradicional é **determinístico**. Mesma entrada, mesma saída,
sempre. Se der errado, dá erro.

Sistema de ML é **probabilístico**. Ele não entrega uma verdade: entrega
uma estimativa. E aqui está a consequência que assusta:

{% hint style="warning" %}
**A falha silenciosa**

Um sistema de ML pode rodar sem um único erro técnico e entregar previsões
completamente erradas. O código está certo. Os dados é que mudaram.

Em software comum, quando quebra, você fica sabendo. Em ML, não. É por isso
que monitorar não é opcional.
{% endhint %}

## Programação tradicional contra ML

| | Programação tradicional | Aprendizado de máquina |
|---|---|---|
| Quem escreve a lógica | a pessoa, em `if` e `else` | o algoritmo, a partir de exemplos |
| O que você fornece | as regras | os dados e as respostas certas |
| O que você testa | o código | o código **e** os dados |
| O que você versiona | o código | o código, os dados **e** o modelo |
| Quando falha | dá erro | continua rodando |

A linha que mais custa a entrar na cabeça é a quarta. Em ML, o modelo é o
resultado de um algoritmo **mais** um conjunto específico de dados. Trocar
os dados dá outro modelo, mesmo sem tocar numa linha de código. Então os
dados também precisam de versão.

Essa mudança é indispensável para problemas que ninguém consegue codificar
a mão: reconhecer fala, detectar fraude, traduzir texto.

## Cientista de dados e engenheiro de ML

**Engenharia de Machine Learning** é aplicar prática de engenharia de
software a sistemas que são probabilísticos por natureza.

| | Ciência de dados | Engenharia de ML |
|---|---|---|
| Onde o trabalho vive | notebook, exploração | código de produção |
| O que entrega | um protótipo que responde a pergunta | um sistema que atende usuários |
| O que otimiza | descoberta | confiabilidade, escala, manutenção |

As duas são necessárias, e a Aula 6 deste curso é justamente a travessia de
uma para a outra.

{% hint style="info" %}
**O princípio dos 5%**

O choque de quem entra na área: o código do algoritmo é cerca de **5%** do
esforço de um sistema industrial.

Os outros 95% são coleta e limpeza de dados, infraestrutura, automação de
testes, pipelines e monitoramento.

O bom engenheiro de ML não é o que escreve o algoritmo mais complexo. É o
que constrói o sistema mais confiável em volta dele.
{% endhint %}

## O ciclo de vida, e por que ele não termina

Um programa comum você escreve uma vez, e ele executa a tarefa fixa. Um
sistema de ML se parece mais com uma planta: precisa de cuidado contínuo,
adaptação ao ambiente e alimentação constante com dados novos.

O ciclo não é uma linha reta que acaba na entrega. É um laço, e uma
descoberta na etapa 3 costuma mandar você de volta para a etapa 2.

<figure><img src="../assets/fundamentos/Gemini_Generated_Image_opoetpopoetpopoe (2).png" alt="Diagrama circular das quatro etapas do ciclo de vida de um projeto de machine learning"><figcaption>Quatro etapas, e uma seta que volta. O monitoramento reinicia o ciclo.</figcaption></figure>

**1. Escopo e objetivo.** Traduzir um desejo de negócio em um objetivo
técnico. "Aumentar o engajamento" não é objetivo de ML. Prever em que o
usuário vai clicar é classificação; prever quanto tempo ele vai ficar é
regressão. Escolher errado aqui custa meses.

**2. Engenharia de dados.** Coletar, limpar registros corrompidos e
transformar tudo num formato que o algoritmo engula. Essa transformação
tem nome: **engenharia de características** (*feature engineering*). É a
etapa mais difícil e mais decisiva. Sem dado bom, nenhum modelo salva.

**3. Desenvolvimento e avaliação.** Escolher o algoritmo, treinar, e então
testar contra dados que o modelo nunca viu. É o `train_test_split` da nossa
Aula 2. O objetivo é garantir que ele **generalize**, e não que tenha
decorado o passado.

**4. Implantação e monitoramento.** Colocar no ar, e continuar olhando. O
mundo muda, os dados mudam junto, e o desempenho cai. Quando cai o
bastante, o ciclo recomeça.

## Por que os modelos apodrecem

Software tradicional não se degrada sozinho: se ninguém mexer no código,
ele faz amanhã o que fazia ontem.

Modelo de ML se degrada. Ele costuma estar no melhor momento logo depois do
treino, e piora conforme o mundo se afasta dos dados com que aprendeu. Há
dois jeitos de isso acontecer, e vale saber diferenciar:

| Nome | O que muda | Exemplo |
|---|---|---|
| Desvio de dados (*data drift*) | as entradas mudam de perfil | um recomendador treinado com adolescentes começa a atender idosos |
| Desvio de conceito (*concept drift*) | a relação entre entrada e resposta muda | os padrões de consumo numa pandemia |

Nos dois casos o código continua rodando sem reclamar. Só as previsões
ficam erradas.

## Produção não é competição

Numa competição ou num artigo, o objetivo é um só: acertar mais que os
outros, num conjunto de dados parado.

Em produção você equilibra objetivos que brigam entre si:

| Requisito | A pergunta que ele faz |
|---|---|
| Precisão | o modelo acerta? |
| Latência | ele responde a tempo? |
| Throughput | ele aguenta o volume? |
| Custo | a conta da infraestrutura fecha? |
| Segurança | os dados dos usuários estão protegidos? |

Um modelo mais preciso e lento demais é inútil. Um modelo mais preciso e
caro demais também. Essa negociação é o trabalho.

## MLOps: se não está automatizado, está quebrado

**MLOps** é a prática de unificar o desenvolvimento de sistemas de ML com a
operação deles. O objetivo é padronizar e automatizar o ciclo inteiro.

A filosofia cabe numa frase dura: se um processo depende de alguém lembrar
de fazer, ele já está quebrado. Intervenção manual é lenta e erra.

| Pilar | O que automatiza |
|---|---|
| Integração contínua (CI) | testa o código **e** a validade dos dados |
| Entrega contínua (CD) | publica sozinho os modelos aprovados |
| Treinamento contínuo (CT) | retreina quando chegam dados novos ou o desempenho cai |

O terceiro é o que não existe em software comum, e é o que fecha o laço do
ciclo de vida.

{% hint style="info" %}
**A regra dos 25%**

O erro mais comum é achar que modelagem matemática é a competência que
importa. Não é. O resultado depende de quatro coisas, em partes iguais:

| Domínio | O que ele garante |
|---|---|
| Engenharia de software | código robusto e testável |
| Engenharia de dados | coleta, limpeza e validação |
| Modelagem | a escolha e o treino do algoritmo |
| Entendimento do negócio | que o problema resolvido seja o problema certo |

O quarto é o mais ignorado e o mais caro quando falta. Modelo perfeito para
a pergunta errada não vale nada.
{% endhint %}

## Cola

| Conceito | O que significa |
|---|---|
| Determinístico | mesma entrada, mesma saída, sempre |
| Probabilístico | a saída é uma estimativa, não uma verdade |
| Falha silenciosa | o código roda sem erro e a previsão está errada |
| Modelo | o resultado de um algoritmo **mais** um conjunto de dados |
| Engenharia de características | transformar dado bruto no formato que o algoritmo usa |
| Generalizar | acertar em dados novos, não só nos do treino |
| Desvio de dados | as entradas mudam de perfil |
| Desvio de conceito | a relação entre entrada e resposta muda |
| MLOps | automatizar o ciclo de vida do modelo, do dado ao monitoramento |
| Princípio dos 5% | o algoritmo é 5% do trabalho; o resto é sistema |
| Regra dos 25% | software, dados, modelagem e negócio, em partes iguais |

## Explique sem olhar

1. O que é uma falha silenciosa, e por que ela não acontece em software comum?
2. Por que em ML você precisa versionar os dados, e não só o código?
3. Qual a diferença entre desvio de dados e desvio de conceito? Dê um exemplo de cada.
4. Por que um modelo mais preciso pode ser a escolha errada em produção?
5. Dos quatro domínios da regra dos 25%, qual costuma faltar, e o que acontece quando falta?

## Para ir além

- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/), de Chip Huyen: o livro mais direto sobre o assunto desta página.
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml): 43 regras práticas de quem opera ML em escala. Comece pela regra 1.
- [Aula 6 — Um Sistema de Ponta a Ponta](../sistemas-de-ml/aula-06-sistema-ponta-a-ponta.md): a aula em que você constrói o sistema, não só o modelo.
- [Glossário](../glossario.md): para revisar qualquer termo novo desta página.

## Bibliografia

Babushkin, V., & Kravchenko, A. (2025). _Machine learning system design: With end-to-end examples_. Manning Publications Co.

Burkov, A. (2020). _Machine learning engineering_. True Positive Inc.

Crowe, R., Hapke, H., Caveness, E., & Zhu, D. (2025). _Machine learning production systems: Engineering machine learning models and pipelines_. O'Reilly Media.

Gift, N., & Deza, A. (2021). _Practical MLOps: Operationalizing machine learning models_. O'Reilly Media.

Huyen, C. (2022). _Designing machine learning systems: An iterative process for production-ready applications_. O'Reilly Media.

Kleppmann, M., & Riccomini, C. (2026). _Designing data-intensive applications: The big ideas behind reliable, scalable, and maintainable systems_ (2nd ed.). O'Reilly Media.
