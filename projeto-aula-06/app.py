"""O painel do Clube do Café.

Para rodar, no terminal, dentro desta pasta:

    streamlit run app.py

O navegador abre sozinho em http://localhost:8501.
"""

import pandas as pd
import streamlit as st

import modelos

st.set_page_config(page_title="Painel do Clube do Café", layout="wide")


def reais(valor):
    # Formata 12345.6 como "R$ 12.346"
    return "R$ " + f"{valor:,.0f}".replace(",", ".")


# O @st.cache_resource faz isso rodar uma vez só, e não a cada clique.
@st.cache_resource
def preparar_sistema():
    clientes = modelos.carregar_clientes()
    vendas = modelos.carregar_vendas()
    classificador = modelos.treinar_classificador(clientes)
    previsor = modelos.treinar_previsor(vendas)
    return clientes, vendas, classificador, previsor


clientes, vendas, classificador, previsor = preparar_sistema()

st.title("Painel do Clube do Café")
st.caption("Duas perguntas do mesmo negócio, respondidas por dois modelos.")

aba_risco, aba_receita = st.tabs(["Risco de cancelamento", "Previsão de receita"])


# --- Aba 1: quem corre risco de cancelar ----------------------------------

with aba_risco:
    st.subheader("A lista de risco de hoje")

    limiar = st.slider("A partir de que risco entrar na lista", 0.10, 0.90, 0.50, 0.05)

    # Trabalhamos numa cópia para não bagunçar a tabela original
    tabela = clientes.copy()
    tabela["risco"] = modelos.calcular_risco(classificador, clientes)
    em_risco = tabela[tabela["risco"] >= limiar]
    em_risco = em_risco.sort_values("risco", ascending=False)

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric("Clientes na base", len(tabela))
    coluna2.metric("Na lista de risco", len(em_risco))
    coluna3.metric("Receita mensal em risco", reais(em_risco["valor_mensal"].sum()))

    # Uma cópia só para mostrar na tela, com o risco em porcentagem
    mostrar = em_risco.head(20).copy()
    mostrar["risco"] = (mostrar["risco"] * 100).round().astype(int).astype(str) + "%"
    mostrar = mostrar[["id_cliente", "plano", "meses_de_casa", "valor_mensal",
                       "entregas_atrasadas", "risco"]]
    mostrar.columns = ["Cliente", "Plano", "Meses de casa", "Valor mensal",
                       "Entregas atrasadas", "Risco"]

    st.dataframe(mostrar, hide_index=True, use_container_width=True)
    st.caption("Os 20 clientes de maior risco. Baixe o limiar para a lista crescer.")

    st.divider()
    st.subheader("Simule um cliente")

    coluna1, coluna2, coluna3, coluna4 = st.columns(4)
    meses = coluna1.slider("Meses de casa", 1, 48, 6)
    valor = coluna2.slider("Valor mensal (R$)", 30, 200, 79)
    atrasos = coluna3.slider("Entregas atrasadas", 0, 6, 2)
    plano = coluna4.selectbox("Plano", ["Degustação", "Clássico", "Premium"])

    risco = modelos.risco_de_um_cliente(classificador, meses, valor, atrasos, plano)

    st.metric("Risco de cancelar", f"{risco:.0%}")
    if risco >= limiar:
        st.warning("Este cliente entraria na lista de risco de hoje.")
    else:
        st.success("Este cliente ficaria fora da lista de hoje.")


# --- Aba 2: quanto o clube vai faturar ------------------------------------

with aba_receita:
    st.subheader("Quanto o clube vai faturar")

    dias = st.slider("Quantos dias prever", 7, 90, 30, 7)

    previsao = modelos.prever(previsor, dias)
    futuro = previsao.tail(dias)

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric(f"Receita prevista em {dias} dias", reais(futuro["yhat"].sum()))
    coluna2.metric("Média por dia", reais(futuro["yhat"].mean()))
    coluna3.metric("Pior caso da faixa", reais(futuro["yhat_lower"].sum()))

    # Um gráfico com as duas coisas: o que já aconteceu e o que vem
    grafico = pd.DataFrame({
        "Receita realizada": vendas.set_index("data")["receita"].tail(120),
        "Previsão": futuro.set_index("ds")["yhat"],
    })
    st.line_chart(grafico)

    st.caption("Os últimos 120 dias de histórico e a previsão à frente.")

    st.divider()
    st.subheader("As peças que o modelo enxergou")
    st.pyplot(previsor.plot_components(previsao))
