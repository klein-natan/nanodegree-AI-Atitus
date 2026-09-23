"""Painel curto para explorar o classificador de cancelamento."""

import streamlit as st

import modelos

st.set_page_config(page_title="Risco de cancelamento", layout="wide")


@st.cache_resource
def preparar():
    clientes = modelos.carregar_dados()
    modelo, parametros, f1_cv, relatorio = modelos.treinar(clientes)
    return clientes, modelo, parametros, f1_cv, relatorio


clientes, modelo, parametros, f1_cv, relatorio = preparar()
st.title("Clube do Café: risco de cancelamento")
st.write("Um classificador de regressão logística, ajustado com Optuna e validação cruzada.")
st.caption(f"Cancelamentos na base: {clientes['cancelou'].mean():.1%} | Melhor F1 nas dobras: {f1_cv:.2f}")
st.code(f"Melhores hiperparâmetros: {parametros}")

with st.expander("Resultado no teste separado"):
    st.text(relatorio)

limiar = st.slider("Limiar para entrar na lista", 0.10, 0.90, 0.50, 0.05)
tabela = clientes.copy()
tabela["risco"] = modelos.calcular_risco(modelo, clientes)
lista = tabela[tabela["risco"] >= limiar].sort_values("risco", ascending=False)
st.metric("Clientes na lista", len(lista))
st.dataframe(lista[["id_cliente", "plano", "meses_de_casa", "entregas_atrasadas", "risco"]],
             hide_index=True, use_container_width=True)
