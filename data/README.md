# Catálogo de dados

Todo dataset usado nas aulas mora nesta pasta. Cada um tem uma entrada aqui
com as colunas, a origem e quais aulas usam. Ver
[guias/estilo-notebooks.md](../guias/estilo-notebooks.md) para as regras de
como os notebooks devem carregar esses arquivos.

## corridas_app.csv

| | |
|---|---|
| **Origem** | Sintético, gerado por [`scripts/gerar_corridas_app.py`](scripts/gerar_corridas_app.py) (seed fixa = 42, reprodutível) |
| **Usado em** | Módulo Modelos Lineares — Aula 1 (Regressão Linear Simples) |
| **Linhas** | 200 corridas fictícias de aplicativo de transporte |

Colunas:

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `id_corrida` | inteiro | — | identificador sequencial da corrida (1 a 200) |
| `distancia_km` | decimal | quilômetros | distância percorrida na corrida |
| `preco` | decimal | reais (R$) | preço final cobrado pela corrida |

O preço foi gerado por `preço ≈ R$ 5,00 + R$ 2,20 × distância + ruído`, para
que exista uma relação linear real, mas imperfeita, entre as duas colunas —
exatamente o que a Aula 1 pede para os alunos redescobrirem. Os alunos nunca
veem essa fórmula antes do exercício de gradiente descendente.

Para regenerar (produz o arquivo byte a byte idêntico):

```bash
uv run python data/scripts/gerar_corridas_app.py
```

## alugueis.csv

| | |
|---|---|
| **Origem** | Sintético, gerado por [`scripts/gerar_alugueis.py`](scripts/gerar_alugueis.py) (seed fixa = 42, reprodutível) |
| **Usado em** | Módulo Modelos Lineares — Aula 2 (Regressão Múltipla) |
| **Linhas** | 300 apartamentos fictícios de aluguel urbano |

Colunas:

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `id_imovel` | inteiro | — | identificador sequencial do apartamento (1 a 300) |
| `area_m2` | decimal | metros quadrados | área útil do apartamento |
| `quartos` | inteiro | quartos | número de quartos (1 a 4) |
| `idade_anos` | inteiro | anos | idade do prédio |
| `bairro` | texto | — | `Centro`, `Jardins` ou `Vila Nova` |
| `aluguel` | decimal | reais (R$) | valor mensal do aluguel |

O aluguel foi gerado por `500 + 25 × área + 200 × quartos − 10 × idade +
efeito do bairro + ruído`, com o Centro valendo 0, o Jardins +800 e a Vila
Nova −300. Área e quartos são correlacionados de propósito (correlação de
0,82), porque a aula usa esse par para explicar por que um coeficiente muda
quando outra variável entra no modelo.

Para regenerar (produz o arquivo byte a byte idêntico):

```bash
uv run python data/scripts/gerar_alugueis.py
```

## assinaturas.csv

| | |
|---|---|
| **Origem** | Sintético, gerado por [`scripts/gerar_assinaturas.py`](scripts/gerar_assinaturas.py) (seed fixa = 42, reprodutível) |
| **Usado em** | Módulo Modelos Lineares — Aula 3 (Regressão Logística) |
| **Linhas** | 400 clientes fictícios de um serviço de streaming |

Colunas:

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `id_cliente` | inteiro | — | identificador sequencial do cliente (1 a 400) |
| `meses_de_casa` | inteiro | meses | há quanto tempo o cliente assina |
| `valor_mensal` | decimal | reais (R$) | quanto ele paga por mês |
| `chamados_suporte` | inteiro | chamados | chamados abertos nos últimos três meses |
| `plano` | texto | — | `Básico`, `Padrão` ou `Premium` |
| `cancelou` | inteiro | 0 ou 1 | 1 se o cliente cancelou, 0 se ficou |

O alvo foi gerado passando `z = −0,60 − 0,080 × meses + 0,015 × valor +
1,00 × chamados + efeito do plano` por uma sigmoide, e sorteando o
cancelamento com essa probabilidade. O resultado tem 36% de cancelamentos:
um desbalanceamento suave, de propósito, porque a aula usa a taxa da classe
maior (64%) para mostrar por que acurácia sozinha engana.

Para regenerar (produz o arquivo byte a byte idêntico):

```bash
uv run python data/scripts/gerar_assinaturas.py
```

## vendas_cafeteria.csv

| | |
|---|---|
| **Origem** | Sintético, gerado por [`scripts/gerar_vendas_cafeteria.py`](scripts/gerar_vendas_cafeteria.py) (seed fixa = 42, reprodutível) |
| **Usado em** | Módulo Séries Temporais — Aulas 4 e 5 |
| **Linhas** | 1.096 dias, de 2022-01-01 a 2024-12-31 |

Colunas:

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `data` | texto (AAAA-MM-DD) | dia | a data daquele dia de vendas |
| `vendas` | decimal | reais (R$) | total vendido no dia |
| `feriado` | inteiro | 0 ou 1 | 1 quando o dia é feriado nacional |

A série foi montada somando `tendência + sazonalidade semanal +
sazonalidade anual + efeito de feriado + ruído`. A tendência sobe R$ 0,45
por dia, sábado é o melhor dia da semana, julho é o melhor mês, e feriado
derruba a venda em cerca de R$ 380. Cada peça existe para ser reconhecida
na Aula 5 e reencontrada pelo Prophet na Aula 6.

Para regenerar (produz o arquivo byte a byte idêntico):

```bash
uv run python data/scripts/gerar_vendas_cafeteria.py
```

## clube_cafe_clientes.csv

| | |
|---|---|
| **Origem** | Sintético, com semente fixa |
| **Usado em** | Módulo Sistemas de ML — Aula 4 |
| **Linhas** | 400 assinantes: 146 cancelaram e 254 ficaram |

O Clube do Café é uma assinatura mensal fictícia. A Aula 4 usa esta tabela
para treinar um classificador de risco de cancelamento. O desbalanceamento
é moderado: 36,5% cancelaram.

`clube_cafe_clientes.csv`:

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `id_cliente` | inteiro | — | identificador do assinante (1 a 400) |
| `meses_de_casa` | inteiro | meses | há quanto tempo ele assina |
| `valor_mensal` | decimal | reais (R$) | quanto ele paga por mês |
| `entregas_atrasadas` | inteiro | entregas | atrasos nos últimos meses |
| `plano` | texto | — | `Degustação`, `Clássico` ou `Premium` |
| `cancelou` | inteiro | 0 ou 1 | 1 se o assinante cancelou |

O projeto em `projeto-aula-04/dados/` contém uma cópia para rodar sem
download.

## torra_cafe.csv

| | |
|---|---|
| **Origem** | Sintético, gerado por [`scripts/gerar_torra_cafe.py`](scripts/gerar_torra_cafe.py) (seed fixa = 42) |
| **Usado em** | Módulo Redes Neurais — Aula 7 |
| **Linhas** | 600 torras de café |

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `id_torra` | inteiro | — | identificador da torra (1 a 600) |
| `temperatura_c` | decimal | graus Celsius | temperatura do tambor |
| `tempo_min` | decimal | minutos | duração da torra |
| `boa` | inteiro | 0 ou 1 | 1 se o café saiu bom |

A torra dá certo dentro de uma elipse centrada em 205 °C e 11 minutos, com
ruído na borda. Essa fronteira fechada é o ponto da aula: uma regressão
logística só traça retas, e nenhuma reta separa o dentro do fora de uma
ilha. São 28,2% de torras boas, então chutar "ruim" para todo mundo acerta
71,8%, exatamente o que a logística consegue.

```bash
uv run python data/scripts/gerar_torra_cafe.py
```

## Fashion-MNIST (não há arquivo nesta pasta)

| | |
|---|---|
| **Origem** | Público, da Zalando Research. Vem dentro do Keras: `keras.datasets.fashion_mnist.load_data()` |
| **Usado em** | Módulo Redes Neurais, Aula 8 (Redes Convolucionais) |
| **Linhas** | 60.000 fotos de treino e 10.000 de teste |
| **Licença** | MIT, ver [o repositório oficial](https://github.com/zalandoresearch/fashion-mnist) |

Cada foto tem 28 por 28 pixels em tons de cinza, com valores inteiros de 0
a 255. O rótulo é um número de 0 a 9:

| Código | Categoria | Código | Categoria |
|---|---|---|---|
| 0 | camiseta | 5 | sandália |
| 1 | calça | 6 | camisa |
| 2 | pulôver | 7 | tênis |
| 3 | vestido | 8 | bolsa |
| 4 | casaco | 9 | bota |

Este é o primeiro dataset do curso que não mora em `data/`. O Keras baixa
e guarda em cache na própria máquina do aluno, então não há URL para
configurar nem arquivo para versionar. Nada a regenerar.

## machado.txt

| | |
|---|---|
| **Origem** | Project Gutenberg, baixado por [`scripts/baixar_machado.py`](scripts/baixar_machado.py) |
| **Usado em** | Módulo LLM do Zero, Aulas 9 a 12 |
| **Tamanho** | 3.676.878 caracteres, 623.120 palavras, 118 caracteres diferentes |
| **Licença** | Domínio público (Machado de Assis morreu em 1908) |

Toda a prosa de Machado que o Gutenberg tem em português, em um arquivo
só: *Memórias Pósthumas de Braz Cubas*, *Quincas Borba*, *Dom Casmurro*,
*Memorial de Ayres*, *Esaú e Jacob*, *A Mão e a Luva*, *Helena*, *Yayá
Garcia*, *Historias sem Data*, *Papeis Avulsos* e *Reliquias de Casa
Velha*. A poesia ficou de fora: a quebra de linha dos versos atrapalharia
um modelo com só 787.584 pesos.

O script tira o cabeçalho e o rodapé do Gutenberg, troca aspas e
travessões tipográficos pelos caracteres simples e junta tudo. A grafia
é a de 1899 ("elle", "cousa", "difficil"), e isso é proposital: a Aula 12
usa esse detalhe para mostrar que o modelo escreve como os dados que viu.

```bash
uv run python data/scripts/baixar_machado.py
```

## machado_bpe.json

| | |
|---|---|
| **Origem** | Gerado por [`scripts/treinar_tokenizador.py`](scripts/treinar_tokenizador.py) a partir de `machado.txt` |
| **Usado em** | Módulo LLM do Zero, Aulas 9 a 12 |
| **Conteúdo** | As 768 fusões do tokenizador BPE, na ordem em que foram aprendidas |

O vocabulário tem 1.024 tokens: 256 bytes mais 768 fusões. Com ele, o
corpus vira 1.484.827 tokens, ou 2,48 caracteres por token.

O tokenizador é parte do modelo. Retreiná-lo gera uma tabela diferente, e
o modelo da Aula 12 deixa de funcionar. Use sempre este arquivo.

```bash
uv run python data/scripts/treinar_tokenizador.py
```

## mini_llm.pt

| | |
|---|---|
| **Origem** | Gerado por [`scripts/treinar_mini_llm.py`](scripts/treinar_mini_llm.py) |
| **Usado em** | Módulo LLM do Zero, Aulas 11 e 12 |
| **Conteúdo** | Os 787.584 pesos do modelo, treinados até o fim |

O modelo de referência das aulas de LLM. Na Aula 11 a turma treina alguns
minutos e vê a perda cair; para gerar texto de verdade na Aula 12, todo
mundo carrega este arquivo. A arquitetura está em
[`scripts/mini_llm.py`](scripts/mini_llm.py), repetida célula por célula
dentro dos notebooks (no Colab não dá para importar um arquivo do
repositório).

```bash
uv run python data/scripts/treinar_mini_llm.py
```

## curso.txt

| | |
|---|---|
| **Origem** | As páginas de `docs/`, reunidas por `data/scripts/juntar_curso.py` |
| **Usado em** | Módulo IA Generativa, Aula 14 (Uma Aplicação de Verdade) |
| **Tamanho** | 187 pedaços, mediana de 763 caracteres |

O material do curso num arquivo só, para o assistente da Aula 14 responder
perguntas sobre ele. Cada pedaço é uma seção de uma página, começa com
`### ` e traz o título da aula e o da seção. O script tira as figuras, os
blocos de código e a marcação do GitBook.

A escolha do curso como documento é proposital: os alunos conhecem o
conteúdo e conseguem julgar se a resposta do assistente está certa.

Regere sempre que uma página mudar:

```bash
python data/scripts/juntar_curso.py
```

## Dataset novo

Todas as catorze aulas do curso já têm a entrada delas aqui. Dataset novo
entra com a mesma ficha: origem, aula que usa, colunas e como regerar.
Sintético precisa de gerador com semente fixa; real precisa de URL de
origem, data do download e licença.
