/* O motor do Explorador do LLM: o tokenizador BPE da Aula 9 e a passada do
   GPT pequeno das Aulas 10 e 11, em JavaScript puro, com os pesos do
   professor (window.MINI_GPT, de modelo.js). Roda no navegador e no Node
   (o teste em data/scripts/testar_explorador.js confere contra o Keras). */
(function (raiz) {
  "use strict";

  function criarMotor(M) {
    var C = M.config, D = C.dimensao, H = C.cabecas, K = D / H, O = C.oculta, V = C.vocabulario;

    /* ---------------------------------------------------------- pesos */
    var binario = typeof atob === "function"
      ? Uint8Array.from(atob(M.dados), function (c) { return c.charCodeAt(0); })
      : Uint8Array.from(Buffer.from(M.dados, "base64"));
    var meias = new Uint16Array(binario.buffer, binario.byteOffset, binario.byteLength / 2);
    function meiaParaFloat(h) {
      var s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
      if (e === 31) return f ? NaN : s * Infinity;
      return s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    var W = {};
    Object.keys(M.tensores).forEach(function (nome) {
      var t = M.tensores[nome], n = t.forma.reduce(function (a, b) { return a * b; }, 1);
      var arr = new Float32Array(n);
      for (var i = 0; i < n; i++) arr[i] = meiaParaFloat(meias[t.inicio + i]);
      W[nome] = arr;
    });

    /* ----------------------------------------------------- tokenizador */
    var padrao = new RegExp(M.padrao, "gu");
    var rank = {}, tabela = [];
    for (var b = 0; b < 256; b++) tabela.push([b]);
    M.fusoes.forEach(function (par, k) {
      rank[par[0] + "," + par[1]] = k;
      tabela.push(tabela[par[0]].concat(tabela[par[1]]));
    });
    var codificador = new TextEncoder(), decodificador = new TextDecoder("utf-8");

    function codificarPalavra(palavra) {
      var ids = Array.from(codificador.encode(palavra));
      while (ids.length >= 2) {
        var melhor = -1, melhorRank = Infinity;
        for (var i = 0; i < ids.length - 1; i++) {
          var r = rank[ids[i] + "," + ids[i + 1]];
          if (r !== undefined && r < melhorRank) { melhorRank = r; melhor = i; }
        }
        if (melhor < 0) break;
        var a = ids[melhor], c = ids[melhor + 1], novo = 256 + melhorRank, saida = [];
        for (var j = 0; j < ids.length; j++) {
          if (j < ids.length - 1 && ids[j] === a && ids[j + 1] === c) { saida.push(novo); j++; }
          else saida.push(ids[j]);
        }
        ids = saida;
      }
      return ids;
    }
    function codificar(texto) {
      var saida = [];
      (texto.match(padrao) || []).forEach(function (p) { saida.push.apply(saida, codificarPalavra(p)); });
      return saida;
    }
    function bytesDe(ids) {
      var bytes = [];
      ids.forEach(function (i) { bytes.push.apply(bytes, tabela[i]); });
      return new Uint8Array(bytes);
    }
    function decodificar(ids) { return decodificador.decode(bytesDe(ids)); }
    function mostrar(id) {
      var t = decodificador.decode(new Uint8Array(tabela[id]));
      if (t.indexOf("�") >= 0) return "<byte " + tabela[id].map(function (x) { return x.toString(16); }).join(" ") + ">";
      return t.replace(/ /g, "·").replace(/\n/g, "⏎");
    }

    /* --------------------------------------------------------- contas */
    function normaCamada(x, n, gama, beta) {
      var y = new Float32Array(n * D);
      for (var t = 0; t < n; t++) {
        var m = 0, v = 0, o = t * D;
        for (var i = 0; i < D; i++) m += x[o + i];
        m /= D;
        for (i = 0; i < D; i++) { var d = x[o + i] - m; v += d * d; }
        var s = 1 / Math.sqrt(v / D + C.epsilon);
        for (i = 0; i < D; i++) y[o + i] = (x[o + i] - m) * s * gama[i] + beta[i];
      }
      return y;
    }
    function densa(x, n, entrada, saida, kernel, vies) {   // x: (n, entrada) -> (n, saida)
      var y = new Float32Array(n * saida);
      for (var t = 0; t < n; t++) {
        var oy = t * saida, ox = t * entrada;
        for (var j = 0; j < saida; j++) y[oy + j] = vies[j];
        for (var i = 0; i < entrada; i++) {
          var xi = x[ox + i], ok = i * saida;
          if (xi === 0) continue;
          for (j = 0; j < saida; j++) y[oy + j] += xi * kernel[ok + j];
        }
      }
      return y;
    }
    function erf(z) {                                         // Abramowitz e Stegun 7.1.26
      var s = z < 0 ? -1 : 1; z = Math.abs(z);
      var t = 1 / (1 + 0.3275911 * z);
      var p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
      return s * (1 - p * Math.exp(-z * z));
    }
    function gelu(x) { for (var i = 0; i < x.length; i++) x[i] = 0.5 * x[i] * (1 + erf(x[i] / Math.SQRT2)); return x; }

    function atencao(h, n, camada) {
      var p = "atencao" + camada + ".";
      var q = densa(h, n, D, D, W[p + "q"], W[p + "bq"]);
      var k = densa(h, n, D, D, W[p + "k"], W[p + "bk"]);
      var v = densa(h, n, D, D, W[p + "v"], W[p + "bv"]);
      var mistura = new Float32Array(n * D), pesos = [], escala = 1 / Math.sqrt(K);
      for (var c = 0; c < H; c++) {
        var mapa = [];
        for (var t = 0; t < n; t++) {
          var linha = new Float32Array(n), maximo = -Infinity;
          for (var s = 0; s <= t; s++) {                      // a máscara causal: só s <= t
            var soma = 0;
            for (var i = 0; i < K; i++) soma += q[t * D + c * K + i] * k[s * D + c * K + i];
            linha[s] = soma * escala;
            if (linha[s] > maximo) maximo = linha[s];
          }
          var total = 0;
          for (s = 0; s <= t; s++) { linha[s] = Math.exp(linha[s] - maximo); total += linha[s]; }
          for (s = 0; s <= t; s++) {
            linha[s] /= total;
            for (i = 0; i < K; i++) mistura[t * D + c * K + i] += linha[s] * v[s * D + c * K + i];
          }
          mapa.push(linha);
        }
        pesos.push(mapa);
      }
      return { saida: densa(mistura, n, D, D, W[p + "o"], W[p + "bo"]), pesos: pesos };
    }

    function rodar(ids) {
      ids = ids.slice(-C.bloco);
      var n = ids.length, x = new Float32Array(n * D), etapas = { ids: ids };
      for (var t = 0; t < n; t++)
        for (var i = 0; i < D; i++)
          x[t * D + i] = W.token[ids[t] * D + i] + W.posicao[t * D + i];
      etapas.entrada = x.slice();
      etapas.atencao = [];
      for (var b = 0; b < C.camadas; b++) {
        var h = normaCamada(x, n, W["norma_a" + b + ".gama"], W["norma_a" + b + ".beta"]);
        var a = atencao(h, n, b);
        for (i = 0; i < x.length; i++) x[i] += a.saida[i];
        etapas.atencao.push(a.pesos);
        h = normaCamada(x, n, W["norma_b" + b + ".gama"], W["norma_b" + b + ".beta"]);
        h = gelu(densa(h, n, D, O, W["densa" + b + ".kernel"], W["densa" + b + ".vies"]));
        h = densa(h, n, O, D, W["volta" + b + ".kernel"], W["volta" + b + ".vies"]);
        for (i = 0; i < x.length; i++) x[i] += h[i];
      }
      var ultima = normaCamada(x.slice((n - 1) * D), 1, W["norma_final.gama"], W["norma_final.beta"]);
      etapas.logits = densa(ultima, 1, D, V, W["saida.kernel"], W["saida.vies"]);
      return etapas;
    }

    function chances(logits, temperatura, k) {
      var ordem = Array.from(logits.keys()).sort(function (a, b) { return logits[b] - logits[a]; });
      var corte = k ? ordem.slice(0, k) : ordem, maximo = logits[ordem[0]] / temperatura, total = 0;
      var p = new Float64Array(logits.length);
      corte.forEach(function (i) { p[i] = Math.exp(logits[i] / temperatura - maximo); total += p[i]; });
      corte.forEach(function (i) { p[i] /= total; });
      return { p: p, ordem: ordem };
    }
    function sortear(p) {
      var r = Math.random(), acumulado = 0;
      for (var i = 0; i < p.length; i++) { acumulado += p[i]; if (r < acumulado) return i; }
      return p.length - 1;
    }

    var direcoes = (function () {                            // para os vizinhos no embedding
      var d = new Float32Array(V * D);
      for (var i = 0; i < V; i++) {
        var norma = 0;
        for (var j = 0; j < D; j++) norma += W.token[i * D + j] * W.token[i * D + j];
        norma = Math.sqrt(norma) || 1;
        for (j = 0; j < D; j++) d[i * D + j] = W.token[i * D + j] / norma;
      }
      return d;
    })();
    function vizinhos(id, quantos) {
      var notas = [];
      for (var i = 0; i < V; i++) {
        if (i === id) continue;
        var s = 0;
        for (var j = 0; j < D; j++) s += direcoes[id * D + j] * direcoes[i * D + j];
        notas.push([i, s]);
      }
      notas.sort(function (a, b) { return b[1] - a[1]; });
      return notas.slice(0, quantos || 5);
    }

    return { config: C, pesos: W, codificar: codificar, decodificar: decodificar, mostrar: mostrar,
             rodar: rodar, chances: chances, sortear: sortear, vizinhos: vizinhos };
  }

  raiz.criarMotorDoLLM = criarMotor;
  if (typeof module !== "undefined") module.exports = criarMotor;
})(typeof window !== "undefined" ? window : globalThis);
