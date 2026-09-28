"""Painel do sistema de risco de cancelamento do Clube do Café.

Rode com:

    streamlit run app.py
"""

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import modelos
from treinar import ARQUIVO_ARTEFATO, treinar_e_salvar

st.set_page_config(page_title="Clube do Café: risco de cancelamento", layout="wide")

NOMES = {
    "meses_de_casa": "Meses de casa",
    "plano": "Plano",
    "valor_mensal": "Valor mensal (R$)",
    "forma_pagamento": "Forma de pagamento",
    "entrou_com_cupom": "Entrou com cupom",
    "entregas_atrasadas": "Entregas atrasadas",
    "chamados_suporte": "Chamados ao suporte",
    "avaliacao_media": "Avaliação média",
    "dias_sem_acessar": "Dias sem acessar",
    "idade": "Idade",
    "missingindicator_avaliacao_media": "Avaliação faltando",
}


def nome_bonito(coluna):
    for original, bonito in NOMES.items():
        if coluna == original:
            return bonito
    # Colunas criadas pelo OneHotEncoder, como "plano_Premium".
    for original, bonito in NOMES.items():
        if coluna.startswith(original + "_"):
            return f"{bonito}: {coluna[len(original) + 1:]}"
    return coluna


def br(numero, formato):
    # Números com vírgula decimal, como se escreve no Brasil.
    return format(numero, formato).replace(".", ",")


def formatar_reais(valor):
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


@st.cache_resource(show_spinner="Treinando o sistema pela primeira vez (uns 20 segundos)...")
def carregar_sistema():
    if not ARQUIVO_ARTEFATO.exists():
        treinar_e_salvar()
    return joblib.load(ARQUIVO_ARTEFATO)


@st.cache_data
def carregar_clientes():
    return modelos.carregar_dados()


sistema = carregar_sistema()
modelo = sistema["modelo"]
clientes = carregar_clientes()
clientes_com_risco = clientes.copy()
clientes_com_risco["risco"] = modelos.prever_risco(modelo, clientes)
y_teste = sistema["y_teste"]
risco_teste = sistema["risco_teste"]

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Decisão")
    limiar = st.slider(
        "Limiar para entrar na lista", 0.05, 0.95, 0.50, 0.05,
        help="Clientes com risco igual ou acima deste valor recebem contato.",
    )
    st.subheader("Economia da retenção")
    custo_contato = st.number_input("Custo de um contato (R$)", 0.0, 500.0, 10.0, 5.0)
    taxa_sucesso = st.slider("Chance de a oferta segurar quem ia sair", 0.0, 1.0, 0.30, 0.05)
    meses_retidos = st.slider("Meses a mais que um cliente salvo fica", 1, 12, 4)

    st.divider()
    st.caption(f"Modelo treinado em {sistema['treinado_em']} com {sistema['tentativas']} tentativas do Optuna.")
    tentativas = st.number_input("Tentativas para retreinar", 5, 200, 40, 5)
    if st.button("Retreinar o modelo"):
        with st.spinner("Retreinando..."):
            treinar_e_salvar(int(tentativas))
        carregar_sistema.clear()
        st.rerun()

st.title("Clube do Café: risco de cancelamento")
st.write(
    "Regressão logística treinada por gradiente descendente estocástico (SGD), "
    "com preparo dos dados num `ColumnTransformer` e hiperparâmetros escolhidos pelo Optuna."
)

aba_visao, aba_modelo, aba_explica, aba_lista, aba_simula = st.tabs(
    ["Visão geral", "Modelo", "Explicabilidade", "Lista de contato", "Simulador"]
)

# ---------------------------------------------------------------- visão geral
with aba_visao:
    metricas = sistema["metricas_teste"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assinantes", f"{len(clientes):,}".replace(",", "."))
    col2.metric("Cancelaram", br(clientes['cancelou'].mean(), ".1%"))
    col3.metric("AUC no teste", br(metricas['auc'], ".2f"))
    col4.metric("F1 no teste (limiar 0,50)", br(metricas['f1'], ".2f"))

    st.subheader("Onde estão os cancelamentos")
    st.caption("Cada barra é a fração de clientes daquele grupo que cancelou.")
    faixas = clientes.copy()
    faixas["faixa_de_meses"] = pd.cut(
        faixas["meses_de_casa"], [0, 6, 12, 24, 36, 60],
        labels=["até 6", "7 a 12", "13 a 24", "25 a 36", "37 a 60"],
    )
    faixas["faixa_sem_acessar"] = pd.cut(
        faixas["dias_sem_acessar"], [-1, 7, 14, 30, 200],
        labels=["até 7", "8 a 14", "15 a 30", "mais de 30"],
    )
    graficos = [
        ("faixa_de_meses", "Meses de casa"),
        ("entregas_atrasadas", "Entregas atrasadas"),
        ("faixa_sem_acessar", "Dias sem acessar"),
        ("plano", "Plano"),
        ("forma_pagamento", "Forma de pagamento"),
        ("entrou_com_cupom", "Entrou com cupom (1 = sim)"),
    ]
    colunas_graficos = st.columns(3)
    for posicao, (coluna, titulo) in enumerate(graficos):
        taxa = faixas.groupby(coluna, observed=True)["cancelou"].mean().reset_index()
        taxa[coluna] = taxa[coluna].astype(str)
        grafico = (
            alt.Chart(taxa, title=titulo)
            .mark_bar()
            .encode(
                x=alt.X(f"{coluna}:N", sort=None, title=None),
                y=alt.Y("cancelou:Q", title="Cancelaram", axis=alt.Axis(format="%")),
                tooltip=[coluna, alt.Tooltip("cancelou:Q", format=".1%")],
            )
            .properties(height=220)
        )
        colunas_graficos[posicao % 3].altair_chart(grafico, width="stretch")

    faltando = clientes["avaliacao_media"].isna().sum()
    st.info(
        f"{faltando} clientes nunca avaliaram um café. O preparo preenche a nota com a "
        "mediana e cria uma coluna avisando que ela faltava."
    )

# ---------------------------------------------------------------- modelo
with aba_modelo:
    st.subheader("1. O modelo ganha de quem não aprende nada?")
    st.caption("Validação cruzada com 5 dobras estratificadas, só no conjunto de treino.")
    comparacao = sistema["baselines"].copy()
    linha_tunada = pd.DataFrame([{"modelo": "SGD com tuning (Optuna)", "acuracia_cv": None, "f1_cv": sistema["f1_cv"]}])
    comparacao = pd.concat([comparacao, linha_tunada], ignore_index=True)
    st.dataframe(
        comparacao.rename(columns={"modelo": "Modelo", "acuracia_cv": "Acurácia", "f1_cv": "F1"}),
        hide_index=True, width="stretch",
        column_config={
            "Acurácia": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption("'Sempre fica' acerta 71% sem aprender nada e tem F1 zero: por isso a acurácia sozinha engana.")

    st.subheader("2. O que o Optuna testou")
    esquerda, direita = st.columns([2, 1])
    historico = sistema["historico"]
    pontos = (
        alt.Chart(historico)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("tentativa:Q", title="Tentativa"),
            y=alt.Y("f1_cv:Q", title="F1 médio nas dobras", scale=alt.Scale(zero=False)),
            color=alt.Color("penalty:N", title="Penalidade"),
            tooltip=list(historico.columns),
        )
    )
    linha = alt.Chart(historico).mark_line(color="black").encode(x="tentativa:Q", y="melhor_ate_agora:Q")
    esquerda.altair_chart(pontos + linha, width="stretch")
    direita.write("**Hiperparâmetros vencedores**")
    for nome, valor in sistema["parametros"].items():
        if isinstance(valor, float):
            valor = br(valor, ".5f")
        direita.write(f"`{nome}` = {valor}")
    direita.caption(
        "`alpha` é a força da regularização; `penalty` diz se ela é L2, L1 ou uma mistura; "
        "`class_weight='balanced'` dá mais peso a quem cancelou."
    )

    st.subheader(f"3. No teste separado, com limiar {br(limiar, '.2f')}")
    no_limiar = modelos.calcular_metricas(y_teste, risco_teste, limiar)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precisão", br(no_limiar['precisao'], ".2f"), help="Dos que o modelo apontou, quantos cancelariam.")
    col2.metric("Recall", br(no_limiar['recall'], ".2f"), help="Dos que cancelariam, quantos o modelo achou.")
    col3.metric("F1", br(no_limiar['f1'], ".2f"))
    col4.metric("Acurácia", br(no_limiar['acuracia'], ".2f"))

    esquerda, direita = st.columns(2)
    esquerda.write("**Matriz de confusão**")
    esquerda.dataframe(modelos.matriz_de_confusao(y_teste, risco_teste, limiar), width="stretch")
    distribuicao = pd.DataFrame({
        "risco": risco_teste,
        "real": y_teste.map({0: "Ficou", 1: "Cancelou"}).to_numpy(),
    })
    histograma = (
        alt.Chart(distribuicao, title="Risco previsto, separado pelo que aconteceu")
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("risco:Q", bin=alt.Bin(step=0.05), title="Risco previsto"),
            y=alt.Y("count():Q", stack=None, title="Clientes"),
            color=alt.Color("real:N", title="Real"),
        )
    )
    regua = alt.Chart(pd.DataFrame({"limiar": [limiar]})).mark_rule(color="red").encode(x="limiar:Q")
    direita.altair_chart(histograma + regua, width="stretch")

# ---------------------------------------------------------------- explicabilidade
with aba_explica:
    st.subheader("Importância por permutação")
    st.write(
        "Embaralhamos uma coluna do teste por vez, 20 vezes, e medimos quanto a AUC cai. "
        "Se o modelo depende daquela coluna, embaralhar destrói a informação e a AUC despenca."
    )
    importancia = sistema["importancia"].copy()
    importancia["variavel"] = importancia["variavel"].map(nome_bonito)
    importancia["minimo"] = importancia["queda_auc"] - importancia["desvio"]
    importancia["maximo"] = importancia["queda_auc"] + importancia["desvio"]
    barras = (
        alt.Chart(importancia)
        .mark_bar()
        .encode(
            x=alt.X("queda_auc:Q", title="Queda na AUC ao embaralhar"),
            y=alt.Y("variavel:N", sort="-x", title=None),
            tooltip=["variavel", alt.Tooltip("queda_auc:Q", format=".4f"), alt.Tooltip("desvio:Q", format=".4f")],
        )
    )
    erros = alt.Chart(importancia).mark_errorbar().encode(
        x=alt.X("minimo:Q", title="Queda na AUC ao embaralhar"), x2="maximo:Q", y=alt.Y("variavel:N", sort="-x")
    )
    st.altair_chart(barras + erros, width="stretch")
    st.caption(
        "Barras perto de zero (como a idade) quase não ajudam o modelo. "
        "Importância não é causa: ela mostra no que o modelo se apoia, não o que faz o cliente sair."
    )

    st.subheader("Os pesos w do modelo")
    st.write(
        "As colunas foram padronizadas, então os pesos são comparáveis: peso positivo "
        "empurra o risco para cima, negativo para baixo. Peso zero foi desligado pela regularização."
    )
    pesos = sistema["pesos"].copy()
    pesos["coluna"] = pesos["coluna"].map(nome_bonito)
    pesos["sentido"] = np.where(pesos["peso"] > 0, "aumenta o risco", "diminui o risco")
    grafico_pesos = (
        alt.Chart(pesos)
        .mark_bar()
        .encode(
            x=alt.X("peso:Q", title="Peso w"),
            y=alt.Y("coluna:N", sort="-x", title=None),
            color=alt.Color("sentido:N", title=None, scale=alt.Scale(range=["#2e7d32", "#c62828"])),
            tooltip=["coluna", alt.Tooltip("peso:Q", format=".3f")],
        )
    )
    st.altair_chart(grafico_pesos, width="stretch")

# ---------------------------------------------------------------- lista de contato
with aba_lista:
    st.subheader("Qual limiar dá mais lucro?")
    st.caption(
        "Todo contato custa; só quem ia cancelar pode ser salvo, e só uma parte deles aceita "
        "a oferta. A curva usa o risco da validação cruzada no treino, então escolher o "
        "limiar por ela não gasta o teste."
    )
    lucro = modelos.lucro_por_limiar(
        sistema["y_treino"], sistema["risco_validacao"], sistema["X_treino"]["valor_mensal"],
        custo_contato, taxa_sucesso, meses_retidos,
    )
    melhor = lucro.loc[lucro["lucro"].idxmax()]
    curva = (
        alt.Chart(lucro)
        .mark_line(point=True)
        .encode(
            x=alt.X("limiar:Q", title="Limiar"),
            y=alt.Y("lucro:Q", title="Lucro na validação, 1.500 clientes (R$)"),
            tooltip=["limiar", "contatos", "cancelariam", alt.Tooltip("lucro:Q", format=",.0f")],
        )
    )
    st.altair_chart(curva + regua, width="stretch")
    lucro_teste = modelos.lucro_por_limiar(
        y_teste, risco_teste, sistema["X_teste"]["valor_mensal"],
        custo_contato, taxa_sucesso, meses_retidos,
    )
    no_teste = lucro_teste[lucro_teste["limiar"] == melhor["limiar"]].iloc[0]
    melhor_limiar_texto = br(melhor["limiar"], ".2f")
    st.write(
        f"Com estes custos, o melhor limiar na validação é **{melhor_limiar_texto}**. "
        f"Conferido no teste separado (500 clientes), ele dá lucro de "
        f"{formatar_reais(no_teste['lucro'])} em {int(no_teste['contatos'])} contatos. "
        f"O limiar escolhido na barra lateral é {br(limiar, '.2f')}."
    )

    st.subheader("Lista de contato")
    lista = clientes_com_risco[clientes_com_risco["risco"] >= limiar]
    lista = lista.sort_values("risco", ascending=False)
    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes na lista", len(lista))
    col2.metric("Custo dos contatos", formatar_reais(len(lista) * custo_contato))
    col3.metric("Risco médio na lista", br(lista['risco'].mean(), ".0%") if len(lista) > 0 else "-")
    colunas_lista = ["id_cliente", "risco", "plano", "meses_de_casa", "entregas_atrasadas",
                     "chamados_suporte", "avaliacao_media", "dias_sem_acessar"]
    st.dataframe(
        lista[colunas_lista], hide_index=True, width="stretch",
        column_config={"risco": st.column_config.ProgressColumn("Risco", format="%.2f", min_value=0, max_value=1)},
    )
    st.download_button(
        "Baixar a lista (CSV)", lista[colunas_lista].to_csv(index=False).encode("utf-8"),
        file_name=f"lista_contato_limiar_{limiar:.2f}.csv", mime="text/csv",
    )
    st.caption(
        "A lista usa a base inteira, como a equipe faria com os assinantes ativos; o desempenho "
        "se mede no teste. Se o Optuna escolheu class_weight='balanced', o risco serve para "
        "ordenar os clientes, mas fica acima da chance real de cancelar."
    )

# ---------------------------------------------------------------- simulador
with aba_simula:
    st.subheader("E se este cliente fosse diferente?")
    mais_arriscados = clientes_com_risco.sort_values("risco", ascending=False)["id_cliente"].tolist()
    id_escolhido = st.selectbox("Cliente (ordenados do maior risco para o menor)", mais_arriscados)
    original = clientes[clientes["id_cliente"] == id_escolhido].iloc[0]

    esquerda, direita = st.columns([1, 1])
    with esquerda:
        st.write("**Mude o que quiser:**")
        planos = ["Degustação", "Clássico", "Premium"]
        pagamentos = ["Cartão", "Pix", "Boleto"]
        cenario = {
            "meses_de_casa": st.slider("Meses de casa", 1, 60, int(original["meses_de_casa"])),
            "plano": st.selectbox("Plano", planos, index=planos.index(original["plano"])),
            "valor_mensal": st.slider("Valor mensal (R$)", 30.0, 160.0, float(original["valor_mensal"]), 1.0),
            "forma_pagamento": st.selectbox("Forma de pagamento", pagamentos, index=pagamentos.index(original["forma_pagamento"])),
            "entrou_com_cupom": int(st.checkbox("Entrou com cupom", bool(original["entrou_com_cupom"]))),
            "entregas_atrasadas": st.slider("Entregas atrasadas", 0, 6, int(original["entregas_atrasadas"])),
            "chamados_suporte": st.slider("Chamados ao suporte", 0, 8, int(original["chamados_suporte"])),
            "dias_sem_acessar": st.slider("Dias sem acessar", 0, 120, int(original["dias_sem_acessar"])),
            "idade": st.slider("Idade", 18, 75, int(original["idade"])),
        }
        sem_nota = pd.isna(original["avaliacao_media"])
        if st.checkbox("Cliente nunca avaliou", sem_nota):
            cenario["avaliacao_media"] = None
        else:
            nota_inicial = 4.0 if sem_nota else float(original["avaliacao_media"])
            cenario["avaliacao_media"] = st.slider("Avaliação média", 1.0, 5.0, nota_inicial, 0.1)

    tabela_original = clientes[clientes["id_cliente"] == id_escolhido]
    tabela_cenario = pd.DataFrame([cenario]).astype({"avaliacao_media": float})
    risco_original = modelos.prever_risco(modelo, tabela_original)[0]
    risco_cenario = modelos.prever_risco(modelo, tabela_cenario)[0]

    with direita:
        col1, col2 = st.columns(2)
        col1.metric("Risco hoje", br(risco_original, ".0%"))
        col2.metric("Risco no cenário", br(risco_cenario, ".0%"), br((risco_cenario - risco_original) * 100, "+.0f") + " pontos",
                    delta_color="inverse")
        if risco_cenario >= limiar:
            st.warning(f"No cenário, o cliente continua na lista (limiar {br(limiar, '.2f')}).")
        else:
            st.success(f"No cenário, o cliente sai da lista (limiar {br(limiar, '.2f')}).")

        st.write("**O que empurra o risco deste cenário**")
        partes = modelos.contribuicoes(modelo, tabela_cenario)
        partes["coluna"] = partes["coluna"].map(nome_bonito)
        partes["sentido"] = np.where(partes["contribuicao"] > 0, "aumenta", "diminui")
        grafico_partes = (
            alt.Chart(partes)
            .mark_bar()
            .encode(
                x=alt.X("contribuicao:Q", title="Contribuição para z (w vezes x)"),
                y=alt.Y("coluna:N", sort="-x", title=None),
                color=alt.Color("sentido:N", title=None, scale=alt.Scale(domain=["aumenta", "diminui"], range=["#c62828", "#2e7d32"])),
                tooltip=["coluna", alt.Tooltip("contribuicao:Q", format=".3f")],
            )
            .properties(height=320)
        )
        st.altair_chart(grafico_partes, width="stretch")
        st.caption("Como o modelo é linear, z é a soma dessas barras mais w0; o risco é a sigmoide de z.")

    st.divider()
    st.subheader("E se mudássemos a operação inteira?")
    st.caption("Aplica a mudança a todos os clientes e compara o risco previsto antes e depois.")
    acoes = st.multiselect(
        "Ações",
        ["Zerar entregas atrasadas", "Reduzir pela metade os dias sem acessar", "Migrar boleto para cartão"],
        default=["Zerar entregas atrasadas"],
    )
    base_nova = clientes.copy()
    if "Zerar entregas atrasadas" in acoes:
        base_nova["entregas_atrasadas"] = 0
    if "Reduzir pela metade os dias sem acessar" in acoes:
        base_nova["dias_sem_acessar"] = (base_nova["dias_sem_acessar"] / 2).round()
    if "Migrar boleto para cartão" in acoes:
        base_nova["forma_pagamento"] = base_nova["forma_pagamento"].replace("Boleto", "Cartão")
    risco_novo = modelos.prever_risco(modelo, base_nova)

    col1, col2 = st.columns(2)
    col1.metric("Risco médio", br(risco_novo.mean(), ".1%"),
                br((risco_novo.mean() - clientes_com_risco['risco'].mean()) * 100, "+.1f") + " pontos", delta_color="inverse")
    antes = int((clientes_com_risco["risco"] >= limiar).sum())
    depois = int((risco_novo >= limiar).sum())
    col2.metric("Clientes acima do limiar", depois, depois - antes, delta_color="inverse")
    st.caption(
        "Cuidado: o modelo aprendeu associações, não causas. O simulador mostra o que o modelo "
        "prevê, e só um teste de verdade com clientes mostra o efeito real da ação."
    )
