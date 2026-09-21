---
description: Termos técnicos do curso, explicados em uma frase cada
---

# Glossário

Toda vez que uma aula usa um termo técnico pela primeira vez, ele entra
aqui. Se você esqueceu o que uma palavra significa, é mais rápido procurar
nesta página do que voltar a reler uma aula inteira.

| Termo | Significado |
|---|---|
| Inteligência artificial (IA) | Programas que tomam decisões ou fazem previsões a partir de dados, em vez de seguir regras escritas à mão |
| Aprendizado de máquina (*machine learning*) | A parte da IA em que o programa aprende a regra sozinho, olhando exemplos |
| Modelo | O programa (com seus parâmetros ajustados) que faz a previsão |
| Dataset | A tabela de dados usada para treinar ou testar um modelo |
| Regressão linear | Um modelo que prevê um número desenhando uma reta (ou um plano, com mais variáveis) no meio dos dados |
| Coeficiente | Um dos números que a regressão ajusta para desenhar a melhor reta (`w₀`, `w₁`, ...); também chamado de peso |
| Resíduo (`εᵢ`) | A diferença entre o valor real e o valor previsto pelo modelo, para um exemplo |
| Função de perda (*loss function*) | A fórmula que resume o quanto um modelo erra, num único número |
| MSE (erro quadrático médio) | A média dos resíduos elevados ao quadrado. A função de perda mais comum em regressão |
| Derivada (inclinação) | O quanto uma curva sobe ou desce em um ponto: a subida dividida pelo avanço, com o avanço quase zero. Diz para que lado mexer um peso para o erro cair, e o quão rápido |
| Reta tangente | A reta que encosta na curva em um único ponto. A inclinação dela é a derivada naquele ponto |
| Gradiente | A inclinação da função de perda, uma para cada peso do modelo. Aponta para onde o erro cresce mais rápido |
| Gradiente descendente | A técnica de ajustar os coeficientes aos poucos, sempre na direção que reduz o erro |
| Taxa de aprendizado (`α`) | O tamanho do passo que o gradiente descendente dá a cada ajuste |
| MAE (erro absoluto médio) | A média do tamanho dos erros de um modelo, na mesma unidade da previsão |
| RMSE (raiz do erro quadrático médio) | A raiz quadrada do MSE. Penaliza erros grandes mais que o MAE |
| R² (coeficiente de determinação) | A fração da variação dos dados que o modelo consegue explicar, de 0 a 1 |
| Regressão múltipla | Uma regressão que usa várias variáveis ao mesmo tempo, cada uma com o seu peso |
| Variável indicadora (*dummy*) | Coluna de 0 ou 1 que representa uma categoria (por exemplo, um bairro) |
| Categoria de referência | A categoria que fica sem coluna própria e serve de comparação para as outras |
| Colinearidade | Duas variáveis que andam juntas e dividem entre si o crédito pelo mesmo efeito |
| R² ajustado | O R² descontando o custo de cada variável nova adicionada ao modelo |
| Conjunto de treino | A parte dos dados com que o modelo aprende |
| Conjunto de teste | A parte separada antes do treino, que o modelo não pode ver, e onde você mede de verdade |
| Sobreajuste (*overfitting*) | Quando o modelo melhora no treino e piora no teste: ele decorou em vez de aprender |
| Regularização | Somar à função de perda um pedágio por coeficiente grande: `custo = MSE + α · penalidade(w)` |
| Força da regularização (`α`) | O quanto o pedágio aperta: 0 desliga o freio, valores grandes empurram tudo para perto de zero. Não confundir com a taxa de aprendizado |
| Ridge (penalidade L2) | Regularização que cobra `Σw²`. Encolhe os coeficientes grandes e não zera nenhum |
| Lasso (penalidade L1) | Regularização que cobra a soma dos coeficientes sem o sinal. Zera os coeficientes inúteis e tira a coluna do modelo |
| ElasticNet | Mistura de Ridge e Lasso, com `l1_ratio` (ρ) decidindo a proporção entre as duas |
| Validação cruzada | Testar vários valores de `α` em pedaços diferentes do treino e ficar com o melhor. É o que as versões `CV` fazem sozinhas |
| Padronização (`StandardScaler`) | Deixar cada coluna com média 0 e desvio 1, para que o freio da regularização não dependa da unidade |
| Classificação | Prever uma categoria (cancela ou não cancela) em vez de um número |
| Regressão logística | O modelo que passa a soma dos pesos por uma sigmoide e devolve uma probabilidade |
| Sigmoide | Função em forma de S que transforma qualquer número em algo entre 0 e 1 |
| Chance (*odds*) | A razão entre a probabilidade de acontecer e a de não acontecer |
| Limiar (*threshold*) | O ponto de corte que transforma uma probabilidade em decisão |
| Matriz de confusão | Tabela com acertos e erros de um classificador: VP, FP, FN e VN |
| Acurácia | A fração de previsões corretas sobre o total |
| Precisão | Dos casos apontados pelo modelo, a fração que era de verdade |
| Recall | Dos casos que eram de verdade, a fração que o modelo apontou |
| F1 | A média harmônica entre precisão e recall, num número só |
| TPR (taxa de verdadeiros positivos) | Dos casos que eram de verdade, a fração que o modelo pegou. É o recall com outro nome |
| FPR (taxa de falsos positivos) | Dos casos que não eram, a fração que o modelo acusou à toa |
| Curva ROC | O gráfico de TPR contra FPR com todos os limiares de uma vez, um ponto para cada corte |
| AUC | A área embaixo da curva ROC. Sorteando um caso positivo e um negativo, é a chance de o modelo dar nota maior ao positivo |
| Calibração | Um modelo calibrado promete probabilidades que acontecem na frequência prometida |
| Brier | A média dos `(probabilidade − resultado)²`. É o MSE aplicado à probabilidade, e mede calibração |
| Limiar por custo | Escolher o corte pela razão entre o preço dos dois erros, e não pelo 0,50 de fábrica |
| Série temporal | Sequência de valores medidos ao longo do tempo, em que a ordem importa |
| Tendência | Para onde uma série caminha no longo prazo |
| Sazonalidade | Padrão que se repete em intervalo fixo, como a semana ou o ano |
| Média móvel | Média dos últimos `k` valores, usada para suavizar uma série |
| Vazamento de dados (*data leakage*) | Quando o modelo enxerga, no treino, informação que não teria na hora de prever |
| Previsão ingênua (*naïve*) | Prever que amanhã será igual a hoje |
| Sazonal ingênua | Prever que amanhã será igual ao mesmo dia do ciclo anterior |
| MAPE | O erro de cada exemplo vira porcentagem do valor daquele exemplo, e o MAPE é a média dessas porcentagens |
| Erro padrão residual (`s`) | O tamanho típico de um resíduo, em unidades do alvo |
| Intervalo de predição | A faixa em que cai **uma** observação nova: `ŷ ± 2s` |
| Intervalo de confiança | A faixa em que cai a **média**; encolhe com mais dados |
| Prophet | Biblioteca de previsão de séries temporais que estima tendência, sazonalidades e feriados |
| Intervalo de incerteza | A faixa em torno da previsão, que cobre uma porcentagem dos casos (80% no padrão do Prophet) |
| Ponto de mudança (*changepoint*) | Data em que a tendência tem permissão para mudar de inclinação |
| Ambiente virtual (`.venv`) | Caixa de bibliotecas isolada, criada para um projeto só |
| `requirements.txt` | Lista das bibliotecas de um projeto, para outra máquina repetir a instalação |
| Streamlit | Biblioteca que transforma um arquivo Python numa página web, sem escrever HTML |
| Protótipo | Sistema que funciona, mas ainda não retreina, registra nem monitora nada |
| Neurônio | A soma de entradas vezes pesos, seguida de uma função de ativação |
| Função de ativação | A dobra depois da soma, que permite fronteiras que não são retas |
| ReLU | Ativação que transforma número negativo em zero e deixa o positivo passar |
| Camada densa | Camada em que todo neurônio recebe todas as saídas da camada anterior |
| Retropropagação | O cálculo que distribui a culpa do erro entre todos os pesos da rede |
| Época | Uma passada completa por todos os exemplos de treino |
| Lote (*batch*) | Quantos exemplos entram antes de cada ajuste dos pesos |
| Normalização | Pôr as colunas na mesma escala antes de treinar uma rede |
| Achatar (*flatten*) | Enfileirar a tabela de pixels de uma imagem, perdendo a informação de quem era vizinho de quem |
| Convolução | Passar o mesmo filtro por toda a imagem, uma janela por vez |
| Filtro (*kernel*) | O quadradinho de nove pesos que a rede aprende e usa na imagem inteira |
| Mapa de ativação | A imagem que sai de um filtro, mostrando onde aquele detalhe apareceu |
| Pooling | Encolher a imagem guardando o maior número de cada quadrado |
| Canal | Quantos números tem cada pixel: 1 em tons de cinza, 3 em cor |
| Softmax | Função que transforma números soltos em probabilidades que somam 1 |
| Matriz de confusão | Tabela que mostra, para cada categoria real, o que o modelo respondeu |
| Modelo de linguagem | Modelo que devolve a probabilidade de cada continuação possível de um texto |
| Corpus | O conjunto de textos usado para treinar um modelo de linguagem |
| Token | A peça mínima de texto que o modelo manipula: um pedaço de palavra, em geral |
| Tokenizador | O programa que converte texto em tokens, e os tokens de volta em texto |
| BPE (*byte pair encoding*) | Treinar o tokenizador juntando repetidas vezes o par de peças mais comum |
| Vocabulário | Quantos tokens diferentes existem na tabela do tokenizador |
| Janela de contexto | Quantos tokens o modelo consegue enxergar de uma vez |
| Embutimento (*embedding*) | A tabela que troca cada token por um vetor de números |
| Atenção | O mecanismo em que cada posição escolhe para onde olhar, com pesos aprendidos |
| Produto escalar | Multiplicar casa com casa e somar: a medida de parecença entre dois vetores |
| Máscara causal | Zera, antes da softmax, tudo o que aponta para o futuro |
| Cabeça de atenção | Uma atenção independente; o modelo roda várias em paralelo |
| RoPE | Codifica a posição girando o vetor por um ângulo proporcional a ela |
| Transformer | A arquitetura feita de blocos com atenção e camada densa, repetidos |
| Conexão residual (atalho) | Somar a entrada de um bloco à saída dele |
| RMSNorm | Põe cada vetor num tamanho padrão, dividindo pela raiz da média dos quadrados |
| SwiGLU | Camada densa com uma porta que decide quanto de cada número passa |
| AdamW | O otimizador padrão para modelos de linguagem |
| Corte de gradiente | Limita o tamanho do passo quando um lote sai fora do normal |
| Perplexidade | `e` elevado à perda: entre quantos tokens o modelo está na dúvida |
| Logit | O número cru que o modelo dá a um token, antes da softmax |
| Temperatura | Divide os logits antes da softmax: baixa concentra, alta espalha |
| Top-k | Fica só com os k tokens mais prováveis antes de sortear |
| Top-p (núcleo) | Fica com os mais prováveis até a soma passar de p |
| Geração autorregressiva | Cada token gerado vira entrada do passo seguinte |
| Alucinação | Frase provável e falsa: o modelo não distingue uma da outra |
| Ajuste por instrução | O treino extra que transforma um continuador de texto em assistente |
| Modelo de fundação | Modelo grande, treinado uma vez, que serve de base para muitas tarefas |
| Hugging Face | O repositório público de onde vêm os modelos abertos |
| Molde de conversa | O texto com marcadores que transforma uma lista de mensagens num texto só |
| Prompt de sistema | O texto que define quem o modelo é, antes da conversa começar |
| Zero-shot | Pedir uma tarefa ao modelo sem exemplo nenhum |
| Few-shot | Colocar dois ou três exemplos resolvidos dentro do próprio prompt |
| RAG | Buscar trechos do seu documento e colá-los no prompt antes de perguntar |
| Pedaço (*chunk*) | Um trecho do documento, do tamanho de uma seção |
| Embutimento de frase | O vetor de um texto inteiro: a média dos vetores dos tokens |
| Similaridade do cosseno | O produto escalar de dois vetores de tamanho 1 |
| Base vetorial | Banco de dados que acha vizinhos rápido, para milhões de pedaços |
| Prompt aumentado | O prompt com os trechos buscados colados dentro |
| Busca híbrida | Buscar por palavra e por vetor, e juntar os resultados |
| Atenção sem pesos | A atenção na versão mais simples: compara os vetores das palavras com eles mesmos, sem tabela treinável. Funciona, e faz cada palavra olhar para si mesma |
| Vetor de contexto | A saída de uma posição na atenção: a própria palavra misturada com o que ela escolheu olhar |
| Modelo de mentira | O esqueleto do modelo, com os blocos devolvendo a entrada sem fazer nada. Roda, tem o formato certo e gera lixo: serve para você preencher uma peça de cada vez |
| Gradiente que desaparece | O encolhimento do gradiente conforme ele volta pelas camadas. Sem atalho, a primeira camada quase não aprende |
| Escalar no momento de responder | Melhorar a resposta gastando mais tokens na hora de responder (raciocínio, votação, revisão), sem treinar nada |
