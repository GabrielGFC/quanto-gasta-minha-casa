/* Motor de previsão da versão estática (GitHub Pages).
   Reproduz no navegador a lógica de api/main.py usando os modelos exportados em modelos.json
   (gerado por `python -m api.exportar_estatico`). Nenhum dado sai do navegador, exceto a consulta do CEP ao ViaCEP. */
const Motor = (() => {
  let M = null;
  const ENTRADAS = 8;
  const NIVEIS_VIDRO = [0, 0.10, 0.25, 0.40];
  const NOME_DIR = { 2:'norte', 3:'leste', 4:'sul', 5:'oeste' };
  const ORDEM_LADOS = ['frente', 'direita', 'fundos', 'esquerda'];
  const NA = { frente:'na frente', direita:'na direita', fundos:'nos fundos', esquerda:'na esquerda' };
  const fmt = (v, c = 2) => Number(v).toFixed(c);
  const g = v => String(+Number(v).toFixed(4));   // equivalente ao :g do Python

  async function carregar(url = 'modelos.json') { M = await (await fetch(url)).json(); return M; }

  function padronizar(x) { return x.map((v, i) => (v - M.scaler.media[i]) / M.scaler.escala[i]); }
  function svr(mod, z) {
    let s = mod.intercept;
    for (let i = 0; i < mod.sv.length; i++) {
      const sv = mod.sv[i]; let d = 0;
      for (let k = 0; k < ENTRADAS; k++) { const t = z[k] - sv[k]; d += t * t; }
      s += mod.coef[i] * Math.exp(-mod.gamma * d);
    }
    return s;
  }
  function mlp(mod, z) {
    let a = z;
    mod.pesos.forEach((W, l) => {
      const out = mod.bias[l].slice();
      for (let j = 0; j < out.length; j++) for (let i = 0; i < a.length; i++) out[j] += a[i] * W[i][j];
      a = l < mod.pesos.length - 1 ? out.map(v => Math.max(0, v)) : out;   // ReLU nas ocultas, identidade na saída
    });
    return a[0];
  }
  const vetor = c => [c.compacidade_relativa, c.area_superficie, c.area_parede, c.area_telhado, c.altura_total, c.orientacao, c.area_janelas, c.distribuicao_janelas];
  const prever = (modelo, casa) => { const z = padronizar(vetor(casa)); return modelo === 'mlp' ? mlp(M.mlp, z) : svr(M[modelo], z); };

  function faixaDe(v) { const [l1, l2] = M.limites; return v <= l1 ? 'Baixo' : v <= l2 ? 'Médio' : 'Alto'; }
  function buscarNaBase(casa) {
    const x = vetor(casa);
    const i = M.base.findIndex(l => x.every((v, k) => Math.abs(l[k] - v) < 1e-6));
    return i < 0 ? null : { linha_planilha: i + 2, carga_aquecimento_real: +fmt(M.base[i][8]), carga_resfriamento_real: +fmt(M.base[i][9]), conjunto: M.base[i][10] ? 'teste' : 'treino' };
  }
  function formatosDaBase(andares) {
    const altura = andares === 2 ? 7 : 3.5, vistos = new Map();
    M.base.forEach(l => { if (Math.abs(l[4] - altura) < 1e-6 && !vistos.has(l[0])) vistos.set(l[0], l); });
    return [...vistos.values()].sort((a, b) => b[0] - a[0])
      .map(l => ({ compacidade_relativa:l[0], area_superficie:l[1], area_parede:l[2], area_telhado:l[3], altura_total:l[4] }));
  }

  function validar(casa) {
    const erro = m => { throw new Error(m); };
    if (!(casa.area_piso_m2 >= 20 && casa.area_piso_m2 <= 2000)) erro('A área precisa estar entre 20 e 2000 m²');
    if (casa.tipo === 'apartamento') {
      if (casa.andar_apartamento >= casa.andares_predio) erro('O andar do apartamento precisa ser menor que o total de andares do prédio (térreo = 0)');
      if (!casa.lados_externos.length) erro('Pelo menos um lado do apartamento precisa dar para fora');
      casa.janelas.forEach(j => { if (!casa.lados_externos.includes(j.lado)) erro(`Há janela ${NA[j.lado]}, mas esse lado não dá para fora do prédio`); });
    }
  }

  /* mesmo algoritmo de POST /analisar */
  function analisar(casa) {
    validar(casa);
    const apto = casa.tipo === 'apartamento';
    const forma = formatosDaBase(apto ? 1 : casa.andares)[casa.formato];
    const passos = [];
    const i0 = casa.orientacao - 2, cardeal = {};
    ORDEM_LADOS.forEach((l, k) => cardeal[l] = (i0 + k) % 4 + 2);
    const porLado = { frente:0, direita:0, fundos:0, esquerda:0 };
    casa.janelas.forEach(j => {
      const area = j.quantidade * j.largura_m * j.altura_m; porLado[j.lado] += area;
      passos.push(`${j.quantidade} janela(s) de ${g(j.largura_m)} m × ${g(j.altura_m)} m ${NA[j.lado]} (${NOME_DIR[cardeal[j.lado]]}) = ${fmt(area)} m² de vidro`);
    });
    casa.portas.forEach(p => {
      const area = p.quantidade * p.largura_m * p.altura_m;
      if (p.material === 'vidro') { porLado[p.lado] += area; passos.push(`${p.quantidade} porta(s) de vidro de ${g(p.largura_m)} m × ${g(p.altura_m)} m ${NA[p.lado]} = ${fmt(area)} m², que contam como vidro`); }
      else passos.push(`${p.quantidade} porta(s) de ${p.material} ${NA[p.lado]} (${NOME_DIR[cardeal[p.lado]]}): não deixam o calor passar como o vidro, então contam como parede`);
    });
    const vidro = Object.values(porLado).reduce((a, b) => a + b, 0), pctReal = vidro / casa.area_piso_m2;
    passos.push(`Vidro total: ${fmt(vidro)} m² em ${g(casa.area_piso_m2)} m² de piso = ${fmt(pctReal * 100, 1)}% do piso`);
    const pct = Math.min(pctReal, 0.4);
    if (pctReal > 0.4) passos.push('Isso passa de 40%, o máximo que existe na base. O cálculo usa 40%.');
    let dist = 0;
    if (vidro > 0) {
      const ladoMax = ORDEM_LADOS.reduce((a, b) => porLado[b] > porLado[a] ? b : a);
      dist = porLado[ladoMax] / vidro >= 0.4 ? cardeal[ladoMax] : 1;
      passos.push(dist !== 1 ? `O vidro está concentrado ${NA[ladoMax]} (${NOME_DIR[dist]})` : 'As janelas estão espalhadas pelos lados da casa');
    }
    const entradaPara = nivel => ({ ...forma, orientacao: casa.orientacao, area_janelas: nivel, distribuicao_janelas: nivel === 0 ? 0 : Math.max(dist, 1) });
    let k = 0; NIVEIS_VIDRO.forEach((n, i) => { if (n <= pct + 1e-9) k = i; });
    const baixo = NIVEIS_VIDRO[k], alto = NIVEIS_VIDRO[Math.min(k + 1, 3)];
    const peso = alto === baixo ? 0 : (pct - baixo) / (alto - baixo);
    const r = {};
    [['aquecimento', 'svr'], ['aquecimento_mlp', 'mlp'], ['resfriamento', 'svr_resf']].forEach(([nome, mod]) => {
      r[nome] = (1 - peso) * prever(mod, entradaPara(baixo)) + peso * prever(mod, entradaPara(alto));
    });
    if (peso > 0) passos.push(`A base só tem 0%, 10%, 25% e 40% de vidro. Como ${fmt(pct * 100, 1)}% fica entre ${g(baixo * 100)}% e ${g(alto * 100)}%, o modelo calculou os dois casos e fez a média ponderada (${fmt((1 - peso) * 100, 0)}% / ${fmt(peso * 100, 0)}%).`);
    let fator = 1;
    if (apto) {
      const parede = forma.area_parede, teto = forma.area_telhado, lados = new Set(casa.lados_externos).size;
      const ultimo = casa.andar_apartamento === casa.andares_predio - 1;
      const exposta = parede * lados / 4 + (ultimo ? teto : 0);
      fator = exposta / (parede + teto);
      passos.push(`Apartamento no ${casa.andar_apartamento === 0 ? 'térreo' : casa.andar_apartamento + 'º andar'} de um prédio de ${casa.andares_predio} andar(es), com ${lados} lado(s) para fora` + (ultimo ? ' e o teto encostando no telhado.' : ' e outro apartamento em cima.'));
      passos.push(`As 768 casas da base são isoladas. Para um apartamento, o sistema mantém só a parte do calor que passa pelas superfícies que dão para fora: ${fmt(exposta, 0)} m² de ${fmt(parede + teto, 0)} m² = ${fmt(fator * 100, 0)}% da casa equivalente. Esse ajuste é uma aproximação nossa, não vem da base.`);
      Object.keys(r).forEach(c => r[c] *= fator);
    }
    return {
      aquecimento_kwh_m2_ano: +fmt(r.aquecimento), aquecimento_mlp_kwh_m2_ano: +fmt(r.aquecimento_mlp),
      resfriamento_kwh_m2_ano: +fmt(r.resfriamento), faixa: faixaDe(r.aquecimento), tipo: casa.tipo,
      fator_apartamento: +fmt(fator, 3),
      vidro: { total_m2: +fmt(vidro), percentual_real: +fmt(pctReal * 100, 1), distribuicao_usada: dist, niveis_interpolados: [baixo, alto], peso: +fmt(peso, 3) },
      formato_usado: forma, casa_mais_parecida_na_base: buscarNaBase(entradaPara(peso < 0.5 ? baixo : alto)), passos,
    };
  }

  /* mesmo cálculo de POST /conta (sem a consulta do CEP, feita no front) */
  function conta(d, cidade) {
    const a = analisar(d.detalhada), cfg = M.tarifas;
    const resfAno = a.resfriamento_kwh_m2_ano * d.area_climatizada_m2 / d.eficiencia_ar;
    const climaMes = resfAno / 12, total = climaMes + d.outros_aparelhos_kwh_mes;
    const tarifa = cfg.tarifa_b1_sem_impostos.valor, band = cfg.bandeiras[d.bandeira] || 0;
    const icms = cfg.icms.valor, pis = cfg.pis_cofins.valor, final = (tarifa + band) / (1 - icms - pis);
    const energia = total * final;
    return {
      local: { cidade, uf: 'GO' },
      previsao_modelos: { aquecimento_kwh_m2_ano: a.aquecimento_kwh_m2_ano, resfriamento_kwh_m2_ano: a.resfriamento_kwh_m2_ano },
      consumo_kwh_mes: { ar_condicionado: +fmt(climaMes, 1), aquecimento: 0, outros_aparelhos: +fmt(d.outros_aparelhos_kwh_mes, 1), total: +fmt(total, 1) },
      tarifa: { sem_impostos: tarifa, bandeira: d.bandeira, adicional_bandeira: band, icms, pis_cofins: pis, final_com_impostos: +fmt(final, 4) },
      valor_mensal: { energia_com_impostos: +fmt(energia), iluminacao_publica: +fmt(d.iluminacao_publica), total: +fmt(energia + d.iluminacao_publica), climatizacao: +fmt(climaMes * final) },
      analise: a,
    };
  }

  function modelosInfo() {
    return [
      { nome:'SVR aquecimento', params:`RBF, C=1000, ε=0,1, γ=${fmt(M.svr.gamma, 3)} · ${M.svr.sv.length} vetores de suporte`, metrica:'MAE 0,34 · R² 0,998' },
      { nome:'SVR resfriamento', params:`RBF, C=1000, ε=0,1, γ=${fmt(M.svr_resf.gamma, 3)} · ${M.svr_resf.sv.length} vetores de suporte`, metrica:`MAE ${M.metricas_resf.mae_teste} · R² ${M.metricas_resf.r2_teste}` },
      { nome:'MLP', params:'camadas (64, 32), ReLU, Adam', metrica:'MAE 0,45 · R² 0,996' },
    ];
  }
  return { carregar, analisar, conta, modelosInfo, prever, padronizar, get dados() { return M; } };
})();
if (typeof module !== 'undefined') module.exports = Motor;
