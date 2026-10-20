"""Ferramentas prontas para os notebooks das aulas de LLM (Aulas 10 e 11).

O notebook baixa este arquivo e faz `from llm_curso import *`. Aqui ficam o
tokenizador da Aula 9, o modelo do professor e as funções que desenham e
medem. O notebook fica só com o que interessa: as perguntas e os
experimentos.
"""
import collections
import json
import os
import re
import urllib.request

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops

URL_DADOS = "https://raw.githubusercontent.com/klein-natan/nanodegree-AI-Atitus/main/data/"
TAMANHO_VOCABULARIO = 1024
BLOCO = 128                      # janela de contexto


def baixar(nome, binario=False):
    local = os.path.join("..", "..", "data", nome)       # só existe no PC do professor
    if os.path.exists(local):
        with open(local, "rb") as arquivo:
            dados = arquivo.read()
    else:
        dados = urllib.request.urlopen(URL_DADOS + nome).read()
    return dados if binario else dados.decode("utf-8")


# --------------------------------------------------------------------------
# O tokenizador da Aula 9
# --------------------------------------------------------------------------

_oficial = json.loads(baixar("machado_bpe.json"))
_padrao = re.compile(_oficial["padrao"])
_fusoes = {(a, b): 256 + k for k, (a, b) in enumerate(_oficial["fusoes"])}
_tabela = {i: bytes([i]) for i in range(256)}
for (_a, _b), _novo in _fusoes.items():
    _tabela[_novo] = _tabela[_a] + _tabela[_b]
_memoria = {}


def _codificar_palavra(palavra):
    ids = list(palavra.encode("utf-8"))
    while len(ids) >= 2:
        pares = collections.Counter(zip(ids, ids[1:]))
        par = min(pares, key=lambda p: _fusoes.get(p, 10**9))
        if par not in _fusoes:
            break
        novo, saida, i = _fusoes[par], [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == par:
                saida.append(novo)
                i += 2
            else:
                saida.append(ids[i])
                i += 1
        ids = saida
    return ids


def codificar(texto):
    """Texto -> lista de números (tokens)."""
    saida = []
    for palavra in _padrao.findall(texto):
        if palavra not in _memoria:
            _memoria[palavra] = _codificar_palavra(palavra)
        saida.extend(_memoria[palavra])
    return saida


def decodificar(ids):
    """Lista de números -> texto."""
    return b"".join(_tabela[int(i)] for i in ids).decode("utf-8", errors="replace")


def mostrar(i):
    """Um token legível: o espaço aparece como ·"""
    return decodificar([i]).replace(" ", "·").replace("\n", "⏎")


def _um_token(palavra):
    """O token de uma palavra, se ela for um token só (tenta com espaço antes)."""
    for tentativa in (" " + palavra.strip(), palavra.strip()):
        ids = codificar(tentativa)
        if len(ids) == 1:
            return ids[0]
    return None


# --------------------------------------------------------------------------
# O modelo (o mesmo de data/scripts/mini_gpt.py)
# --------------------------------------------------------------------------

class TokenEPosicao(keras.layers.Layer):
    """Soma o embedding de cada token com o embedding da posição dele."""

    def __init__(self, dimensao=128, **kwargs):
        super().__init__(**kwargs)
        self.token = keras.layers.Embedding(TAMANHO_VOCABULARIO, dimensao, name="token")
        self.posicao = keras.layers.Embedding(BLOCO, dimensao, name="posicao")

    def call(self, ids):
        n = ops.shape(ids)[1]
        return self.token(ids) + self.posicao(ops.arange(n))


def _montar(camadas=4, dimensao=128, cabecas=4):
    entrada = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = TokenEPosicao(dimensao, name="embedding")(entrada)
    for i in range(camadas):
        h = keras.layers.LayerNormalization(name=f"norma_a{i}")(x)
        h = keras.layers.MultiHeadAttention(num_heads=cabecas, key_dim=dimensao // cabecas,
                                            name=f"atencao{i}")(h, h, use_causal_mask=True)
        x = x + h
        h = keras.layers.LayerNormalization(name=f"norma_b{i}")(x)
        h = keras.layers.Dense(2 * dimensao, activation="gelu", name=f"densa{i}")(h)
        x = x + keras.layers.Dense(dimensao, name=f"volta{i}")(h)
    x = keras.layers.LayerNormalization(name="norma_final")(x)
    return keras.Model(entrada, keras.layers.Dense(TAMANHO_VOCABULARIO, name="saida")(x))


def baixar_pesos_do_professor():
    if not os.path.exists("mini_llm.weights.h5"):
        with open("mini_llm.weights.h5", "wb") as arquivo:
            arquivo.write(baixar("mini_llm.weights.h5", binario=True))
    return "mini_llm.weights.h5"


def modelo_do_professor():
    """O modelo da aula, já treinado por uma hora em Machado de Assis."""
    modelo = _montar()
    modelo.load_weights(baixar_pesos_do_professor())
    return modelo


# --------------------------------------------------------------------------
# Exercício 1: o token vira vetor
# --------------------------------------------------------------------------

def exemplos_de_treino(frase):
    """Mostra os exemplos de treino que uma frase vira."""
    ids = codificar(frase)
    print(f"{len(ids)} tokens: " + " | ".join(mostrar(i) for i in ids) + "\n")
    print("O modelo lê...".ljust(48) + "e tem que prever")
    for posicao in range(1, len(ids)):
        contexto = decodificar(ids[:posicao]).replace("\n", " ")
        if len(contexto) > 44:
            contexto = "…" + contexto[-43:]
        print(f"{contexto:<48}{mostrar(ids[posicao])}")
    return len(ids) - 1


def _direcoes(modelo):
    tabela = np.asarray(modelo.get_layer("embedding").token.embeddings)
    return tabela / np.linalg.norm(tabela, axis=1, keepdims=True)


def vetor(modelo, palavra):
    """Os 128 números do embedding de uma palavra."""
    token = _um_token(palavra)
    if token is None:
        raise ValueError(f'"{palavra}" vira mais de um token: escolha uma palavra mais comum.')
    return np.asarray(modelo.get_layer("embedding").token.embeddings)[token]


def vizinhos(modelo, palavra, palpite=None, quantos=8):
    """Os tokens cujo embedding aponta para o mesmo lado (similaridade do cosseno)."""
    token = _um_token(palavra)
    if token is None:
        print(f'"{palavra}" vira mais de um token. Escolha uma palavra mais comum.')
        return []
    direcoes = _direcoes(modelo)
    notas = direcoes @ direcoes[token]
    ordem = [int(i) for i in np.argsort(notas)[::-1] if i != token][:quantos]
    print(f'Os mais parecidos com "{palavra.strip()}":')
    for i in ordem:
        print(f"   {mostrar(i):<12} {notas[i]:.2f}")
    if palpite is not None:
        alvo = _um_token(palpite)
        if alvo is None:
            print(f'\nSeu palpite "{palpite}" vira mais de um token, então não tem um vetor só.')
        elif alvo in ordem:
            print(f'\n✅ Seu palpite "{palpite.strip()}" está entre os {quantos} mais parecidos.')
        else:
            lugar = int((notas > notas[alvo]).sum())
            print(f'\n🤔 Seu palpite "{palpite.strip()}" ficou em {lugar}º lugar, com similaridade '
                  f"{notas[alvo]:.2f}. O modelo achou outros mais parecidos.")
    return [mostrar(i) for i in ordem]


GRUPOS_EXEMPLO = {
    "números": ["dous", "tres", "duas", "alguns", "outros", "todos"],
    "pessoas": ["elle", "ella", "Rubião", "Helena", "Jorge", "moça"],
    "verbos": ["era", "foi", "fosse", "eram", "estava", "tinha"],
    "corpo": ["olhos", "mão", "mãos", "cabeça", "coração"],
    "conectivos": ["quando", "porque", "onde", "quem", "como"],
}
_CORES = ["#B85042", "#2B6CB0", "#B7791F", "#2F855A", "#6B46C1", "#D53F8C"]


def mapa_3d(modelo, meu_grupo=None):
    """Desenha os embeddings dos grupos de exemplo (e o seu) em 3D, de três ângulos."""
    from sklearn.decomposition import PCA
    grupos = dict(GRUPOS_EXEMPLO)
    if meu_grupo:
        validas = [p for p in meu_grupo if _um_token(p) is not None]
        fora = [p for p in meu_grupo if _um_token(p) is None]
        if fora:
            print("Ficaram de fora (viram mais de um token): " + ", ".join(fora))
        if validas:
            grupos["seu grupo"] = validas
    palavras, ids, cores = [], [], []
    for (nome, lista), cor in zip(grupos.items(), _CORES):
        for palavra in lista:
            token = _um_token(palavra)
            if token is not None:
                palavras.append(palavra)
                ids.append(token)
                cores.append(cor)
    pontos = PCA(n_components=3).fit_transform(_direcoes(modelo)[ids])
    figura = plt.figure(figsize=(16, 5.5))
    for n, angulo in enumerate([20, 110, 200]):
        eixo = figura.add_subplot(1, 3, n + 1, projection="3d")
        eixo.scatter(pontos[:, 0], pontos[:, 1], pontos[:, 2], c=cores, s=40, depthshade=False)
        for (x, y, z), palavra, cor in zip(pontos, palavras, cores):
            peso = "bold" if cor == _CORES[5] else "normal"
            eixo.text(x, y, z, " " + palavra, color=cor, fontsize=9, fontweight=peso)
        eixo.view_init(elev=20, azim=angulo)
        eixo.set_xticks([]); eixo.set_yticks([]); eixo.set_zticks([])
    legenda = [plt.Line2D([], [], marker="o", linestyle="", color=cor, label=nome)
               for nome, cor in zip(grupos, _CORES)]
    figura.legend(handles=legenda, loc="lower center", ncol=len(grupos), frameon=False)
    plt.show()


# --------------------------------------------------------------------------
# Exercício 2: a atenção
# --------------------------------------------------------------------------

def _posicao_da_palavra(ids, frase, palavra):
    """A posição do último token da primeira ocorrência da palavra na frase."""
    alvo = palavra.strip()
    inicio = re.search(r"(?<!\w)" + re.escape(alvo) + r"(?!\w)", frase)
    if not inicio:
        return None
    fim = inicio.end()
    andado = 0
    for posicao in range(len(ids)):
        andado = len(decodificar(ids[:posicao + 1]))
        if andado >= fim:
            return posicao
    return None


def _pesos_de_atencao(modelo, ids):
    """Os pesos de atenção de cada camada: (camadas, cabeças, n, n)."""
    x = modelo.get_layer("embedding")(np.array([ids]))
    camadas, i = [], 0
    while True:
        try:
            norma = modelo.get_layer(f"norma_a{i}")
        except ValueError:
            break
        h = norma(x)
        h, pesos = modelo.get_layer(f"atencao{i}")(h, h, use_causal_mask=True,
                                                   return_attention_scores=True)
        x = x + h
        h = modelo.get_layer(f"densa{i}")(modelo.get_layer(f"norma_b{i}")(x))
        x = x + modelo.get_layer(f"volta{i}")(h)
        camadas.append(np.asarray(pesos[0]))
        i += 1
    return np.array(camadas)


def para_onde_olha(modelo, frase, palavra):
    """Para onde a palavra olha, em cada camada (a média das quatro cabeças)."""
    ids = codificar(frase)
    posicao = _posicao_da_palavra(ids, frase, palavra)
    if posicao is None:
        print(f'Não achei "{palavra}" na frase. Confira a grafia.')
        return
    pesos = _pesos_de_atencao(modelo, ids)[:, :, posicao, :posicao + 1].mean(axis=1)
    rotulos = [mostrar(i) for i in ids[:posicao + 1]]
    figura, eixos = plt.subplots(len(pesos), 1, figsize=(max(8, 0.55 * len(rotulos)), 2.0 * len(pesos)),
                                 sharex=True)
    for camada, (eixo, linha) in enumerate(zip(eixos, pesos)):
        cores = ["#B85042" if v == linha.max() else "#B8B8C0" for v in linha]
        eixo.bar(range(len(linha)), linha, color=cores)
        eixo.set_ylabel(f"camada {camada + 1}")
        eixo.set_ylim(0, 1)
        eixo.spines[["top", "right"]].set_visible(False)
    eixos[-1].set_xticks(range(len(rotulos)), rotulos, rotation=45, ha="right")
    titulo = f'Para onde "{palavra.strip()}" olha (o mais alto, em vermelho)'
    if mostrar(ids[posicao]).strip('·') != palavra.strip():
        titulo += ("\na palavra virou vários tokens; a atenção mostrada é a do último pedaço, "
                   f'"{mostrar(ids[posicao])}"')
    figura.suptitle(titulo)
    plt.tight_layout()
    plt.show()
    for camada, linha in enumerate(pesos):
        print(f"camada {camada + 1}: olha mais para {rotulos[int(linha.argmax())]!r} ({linha.max():.0%})")


def apostas(modelo, frase):
    """Em cada posição, o token mais provável para vir em seguida, e a chance dele."""
    ids = codificar(frase)
    chances = np.asarray(ops.softmax(modelo(np.array([ids]))[0], axis=-1))
    print(f"{'posição':<9}{'leu até':<14}{'aposta no próximo':<20}chance")
    resultado = []
    for posicao, token in enumerate(ids):
        melhor = int(chances[posicao].argmax())
        resultado.append((melhor, round(float(chances[posicao, melhor]), 4)))
        print(f"{posicao:<9}{mostrar(token):<14}{mostrar(melhor):<20}{chances[posicao, melhor]:.0%}")
    return resultado


def estranheza(modelo, texto):
    """A perda do modelo num texto: quanto ele se surpreende. Menor = mais provável."""
    ids = np.array([codificar(texto)])
    perda = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return float(perda(ids[:, 1:], modelo(ids[:, :-1])))


# --------------------------------------------------------------------------
# Exercício 3: o modelo inteiro
# --------------------------------------------------------------------------

def chances(modelo, texto, quantos=5):
    """As continuações mais prováveis de um texto, com as chances."""
    logits = modelo(np.array([codificar(texto)[-BLOCO:]]))[0, -1]
    p = np.asarray(ops.softmax(logits))
    print(f'Depois de "{texto}":')
    for i in np.argsort(p)[::-1][:quantos]:
        print(f"   {mostrar(i):<12} {p[i]:6.1%}")
    return float(p.max())


def gerar(modelo, texto, quantos=50, temperatura=0.8, k=None):
    """Continua o texto, sorteando um token de cada vez.

    temperatura: divide as pontuações antes do sorteio (baixa concentra, alta espalha).
    k: se der um número, só os k tokens mais prováveis entram no sorteio (top-k).
    """
    indices = codificar(texto)
    for _ in range(quantos):
        ultimos = modelo(np.array([indices[-BLOCO:]]))[0, -1]
        if k:
            melhores, candidatos = ops.top_k(ultimos, k)
            sorteado = keras.random.categorical(melhores[None] / temperatura, 1)
            indices.append(int(candidatos[int(sorteado[0, 0])]))
        else:
            sorteado = keras.random.categorical(ultimos[None] / temperatura, 1)
            indices.append(int(sorteado[0, 0]))
    return decodificar(indices)


# --------------------------------------------------------------------------
# Aula 11: treinar
# --------------------------------------------------------------------------

def montar_modelo(camadas=4, dimensao=128, cabecas=4):
    """O modelo da Aula 10, com pesos sorteados."""
    return _montar(camadas, dimensao, cabecas)


_dados = {}


def _tokens(nome):
    """O texto em tokens, separado em treino (90%) e validação (10%)."""
    if nome not in _dados:
        print(f"Transformando {nome} em tokens (uns 30 segundos, só na primeira vez)...")
        tokens = np.array(codificar(baixar(nome)), dtype="int32")
        corte = int(0.9 * len(tokens))
        _dados[nome] = (tokens[:corte], tokens[corte:])
    return _dados[nome]


def _lotes(dados, lote=32):
    while True:
        inicios = np.random.randint(0, len(dados) - BLOCO - 1, lote)
        yield (np.stack([dados[i:i + BLOCO] for i in inicios]),
               np.stack([dados[i + 1:i + BLOCO + 1] for i in inicios]))


def _taxa_no_momento(fracao, taxa_maxima):
    if fracao < 0.03:
        return taxa_maxima * fracao / 0.03                                         # aquecimento
    resto = (fracao - 0.03) / 0.97
    return taxa_maxima * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * resto)))         # resfriamento


class _Relogio(keras.callbacks.Callback):
    """Ajusta a taxa pela fração do tempo que passou, para na hora, e anota a curva."""

    def __init__(self, minutos, taxa_maxima):
        super().__init__()
        self.minutos, self.taxa_maxima = minutos, taxa_maxima
        self.historico = {"minuto": [], "treino": [], "validacao": []}

    def on_train_begin(self, logs=None):
        import time
        self.relogio, self.inicio = time, time.time()

    def on_train_batch_begin(self, passo, logs=None):
        fracao = (self.relogio.time() - self.inicio) / (self.minutos * 60)
        if fracao >= 1:
            self.model.stop_training = True
        self.model.optimizer.learning_rate.assign(_taxa_no_momento(min(fracao, 1), self.taxa_maxima))

    def on_epoch_end(self, conferencia, logs=None):
        minuto = (self.relogio.time() - self.inicio) / 60
        self.historico["minuto"].append(minuto)
        self.historico["treino"].append(float(logs["loss"]))
        self.historico["validacao"].append(float(logs["val_loss"]))
        print(f"   {minuto:4.1f} min   perda no treino {logs['loss']:.2f}   "
              f"na validação {logs['val_loss']:.2f}")


def treinar(modelo, minutos, taxa=1e-3, texto="machado.txt", arquivo=None, conferir_a_cada=200):
    """Treina o modelo pelo relógio, com AdamW e corte do gradiente.

    A cada `conferir_a_cada` passos, mede a perda na validação. Se `arquivo`
    tiver um nome, salva ali o modelo sempre que a validação bater o recorde.
    """
    treino, validacao = _tokens(texto)
    modelo.compile(optimizer=keras.optimizers.AdamW(learning_rate=taxa, weight_decay=0.1,
                                                    beta_2=0.95, global_clipnorm=1.0),
                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    relogio = _Relogio(minutos, taxa)
    callbacks = [relogio]
    if arquivo:
        callbacks.append(keras.callbacks.ModelCheckpoint(arquivo, monitor="val_loss", save_best_only=True,
                                                         save_weights_only=True))
    try:
        modelo.fit(_lotes(treino), steps_per_epoch=conferir_a_cada, epochs=100_000,
                   validation_data=_lotes(validacao), validation_steps=10,
                   callbacks=callbacks, verbose=0)
    except KeyboardInterrupt:
        print("Treino interrompido por você.")
    if relogio.historico["validacao"]:
        print(f"Fim: melhor perda na validação {min(relogio.historico['validacao']):.2f}")
    return relogio.historico


def desenhar_curva(*historicos, nomes=None):
    """A perda na validação ao longo do tempo, de um ou mais treinos."""
    plt.figure(figsize=(9, 4))
    for n, historico in enumerate(historicos):
        nome = nomes[n] if nomes else "validação"
        plt.plot(historico["minuto"], historico["validacao"], marker="o", label=nome)
    plt.axhline(np.log(1024), color="gray", linestyle=":", label="chute ao acaso (6,93)")
    plt.xlabel("minutos de treino")
    plt.ylabel("perda na validação")
    plt.legend()
    plt.show()


def ajuste_fino(modelo, passos=200, texto="curso.txt", taxa=3e-4):
    """Uma cópia do modelo, com o treino continuado em outro texto."""
    treino, _ = _tokens(texto)
    copia = montar_modelo()
    copia.set_weights(modelo.get_weights())
    copia.compile(optimizer=keras.optimizers.AdamW(learning_rate=taxa, global_clipnorm=1.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    historia = copia.fit(_lotes(treino), steps_per_epoch=passos // 4, epochs=4, verbose=0)
    print(f"Ajuste fino: {passos} passos em {texto}, perda de "
          f"{historia.history['loss'][0]:.2f} para {historia.history['loss'][-1]:.2f}")
    return copia


def baixar_para_o_computador(arquivo):
    """No Colab, baixa o arquivo para o seu computador."""
    try:
        from google.colab import files
        files.download(arquivo)
        print("Download iniciado: confira a pasta de downloads.")
    except ImportError:
        print(f"Fora do Colab: o arquivo {arquivo} está na pasta deste notebook.")


def contar_pesos(modelo):
    """Quantos pesos há em cada tipo de camada."""
    grupos = {"embedding": 0, "atenção": 0, "camada densa": 0, "normalização": 0, "saída": 0}
    for camada in modelo.layers:
        nome = camada.name
        if nome == "embedding":
            grupos["embedding"] += camada.count_params()
        elif nome.startswith("atencao"):
            grupos["atenção"] += camada.count_params()
        elif nome.startswith(("densa", "volta")):
            grupos["camada densa"] += camada.count_params()
        elif nome.startswith("norma"):
            grupos["normalização"] += camada.count_params()
        elif nome == "saida":
            grupos["saída"] += camada.count_params()
    total = sum(grupos.values())
    for grupo, quantos in grupos.items():
        print(f"   {grupo:<13} {quantos:>13,}  ({quantos / total:5.1%})".replace(",", "."))
    print(f"   {'total':<13} {total:>13,}".replace(",", "."))
    return total
