# Guia do repositório para agentes de IA

Este repositório contém o material, em português do Brasil, de 14 aulas
de IA e ML. Foi criado originalmente com Claude Code. As fontes de autoria
e os geradores continuam no disco do professor, mas `.gitignore` os
exclui da publicação. Consulte essas fontes ao alterar uma aula.

## Onde está cada coisa

- `docs/`: fonte das páginas de leitura em Markdown, no formato GitBook.
  `docs/SUMMARY.md` define a ordem do livro; as imagens ficam em
  `docs/assets/`. O conteúdo pedagógico deve ser editado aqui primeiro.
- `index.html`: livro estático **gerado**, versionado e publicado. Depois
  de mudar `docs/`, rode `tools/build_site.py` e `tools/verificar_site.py`.
  O gerador usa `tools/site_template.html`; não edite o HTML gerado à mão.
- `notebooks/`: um `.ipynb` por aula, dividido em demonstração e
  exercícios com `# SEU CODIGO AQUI`. Na Aula 4 os exercícios ficam ao
  longo do código, em células "Sua vez". As Aulas 5 e 6 não têm
  exercício nenhum: são demonstração, com código direto, sem funções
  próprias e com as métricas do `sklearn.metrics`.
  Os links Colab nas páginas e no `README.md` precisam apontar para o
  nome real do notebook.
- `data/`: dados dos notebooks e artefatos do pequeno modelo de linguagem.
  `data/curso.txt` é gerado a partir de `docs/` por
  `data/scripts/juntar_curso.py`; rode o script quando mudar as páginas.
  Os demais scripts de dados ficam em `data/scripts/` (ignorados pelo Git).
- `aulas/`: fontes do professor, ignoradas pelo Git. Cada pasta contém
  `slides.md` em YAML, `plano.md`, `notebook_fonte.py`, figuras e, em geral,
  um `slides.pptx` gerado. As ferramentas estão em `.claude/skills/` e as
  regras editoriais em `guias/`. Edite `slides.md`, gere o PPTX com
  `.claude/skills/aula-slides/scripts/build_slides.py` e renderize para
  conferir. O projeto da Aula 4 era a antiga pasta `aulas/06-sistema-ml`;
  ela conserva o nome antigo, mas o código e o plano locais foram
  atualizados. O notebook publicado da Aula 4 é gerado do
  `notebook_fonte.py` dessa pasta e copiado para `notebooks/`.
- `projeto-aula-04/`: projeto local da Aula 4, exemplo completo do que o
  notebook constrói em aula. Regressão logística treinada por SGD
  (`SGDClassifier(loss="log_loss")`), preparo com `ColumnTransformer`,
  baselines com `DummyClassifier`, tuning com Optuna na validação cruzada
  estratificada, importância por permutação, limiar escolhido pelo lucro na
  validação e painel Streamlit com simulador. `modelos.py` contém a
  modelagem; `treinar.py` grava o artefato em `artefatos/` (ignorado pelo
  Git); `app.py` é o painel. O notebook da aula é gerado de
  `aulas/06-sistema-ml/notebook_fonte.py`: a versão dos alunos, com lacunas
  de code-along, vai para `notebooks/`; a do professor, executada, fica em
  `aulas/06-sistema-ml/aula-04-gabarito.ipynb`.

## Sequência atual

1. Aulas 1–3: modelos lineares, terminando em regressão logística.
2. Aula 4: sistema de classificação com Optuna.
3. Aulas 5–6: séries temporais (régua, regressão com calendário, ARIMA, Prophet e TimesFM).
4. Aulas 7–8: redes neurais; 9–12: LLM do zero; 13–14: IA generativa.

Os nomes dos arquivos publicados das Aulas 4–6 acompanham essa sequência.
Algumas fontes ignoradas em `aulas/` e pastas de imagens em `docs/assets/`
ainda preservam os números originais; confira as referências antes de mover
pastas ou figuras.

## Convenções de edição

- Escreva explicações curtas, concretas e em português do Brasil. Defina
  siglas e termos novos na primeira ocorrência e use exemplos numéricos.
- Preserve o teste separado do tuning. Ajuste hiperparâmetros só no
  treino com validação cruzada; deixe transformações aprendidas dentro
  do `Pipeline` para evitar vazamento entre dobras.
- Não volte a incluir seções finais `Explique sem olhar`, `Cola` ou `Cola
  da aula`, nem blocos de `Pergunta para a turma` nas páginas do livro.
  Os notebooks continuam com exercícios práticos.
- Ao mudar a ordem ou o nome de uma aula, revise `docs/SUMMARY.md`,
  `docs/README.md`, `README.md`, os links internos, os notebooks e as
  fontes ignoradas em `aulas/`. Depois regere o site.
- Notebooks devem continuar válidos como JSON e ter células de código
  sem saída gravada. Dados locais para Colab usam URLs brutas do
  repositório publicadas no próprio notebook.

## Verificação rápida

Rode `python data/scripts/juntar_curso.py`, depois
`.venv/Scripts/python.exe tools/build_site.py` e
`.venv/Scripts/python.exe tools/verificar_site.py` no Windows. Abra o
`index.html` em um navegador e confira menu, links e sumário. Valide os
notebooks com `json.load` e rode `python -m compileall projeto-aula-04`.
Para conferir o projeto da Aula 4 sem abrir Streamlit:

```bash
python -c "import sys; sys.path.insert(0, 'projeto-aula-04'); import modelos; d = modelos.carregar_dados(); s = modelos.treinar_sistema(d, 3); print(s['parametros'], s['metricas_teste'], len(modelos.prever_risco(s['modelo'], d)))"
```
