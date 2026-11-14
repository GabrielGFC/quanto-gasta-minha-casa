"""Gera docs/index.html (versão estática para o GitHub Pages) a partir de front/index.html.

Troca as chamadas à API Python pelo motor em JavaScript (docs/motor.js + docs/modelos.json).
Uso, na raiz do projeto:
    python -m api.exportar_estatico
    python docs/gerar_site.py
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
t = (RAIZ / "front" / "index.html").read_text(encoding="utf-8")


def troca(antes, depois):
    global t
    assert antes in t, f"trecho não encontrado: {antes[:60]}"
    t = t.replace(antes, depois)


troca("<script>\nconst $ =", '<script src="motor.js"></script>\n<script>\nconst $ =')

troca("""fetch('/modelos').then(r => r.json()).then(d => {
  $('tab-modelos').innerHTML = d.modelos.map(m => `<tr><td>${m.nome}</td><td>${m.params}</td><td>${m.mae_teste !== undefined ? `MAE ${m.mae_teste} · R² ${m.r2_teste}` : `acurácia ${m.acuracia_teste}`}</td></tr>`).join('') +
    '<tr><td>SVR resfriamento</td><td>RBF, C=1000, ε=0,1, γ=scale</td><td>MAE 0,85 · R² 0,977</td></tr>';
}).catch(() => {});""",
"""const motorPronto = Motor.carregar().then(() => {
  $('tab-modelos').innerHTML = Motor.modelosInfo().map(m => `<tr><td>${m.nome}</td><td>${m.params}</td><td>${m.metrica}</td></tr>`).join('');
});""")

troca("""    const [rc, ra] = await Promise.all([
      fetch('/conta', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(corpo) }),
      fetch('/analisar', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(corpo.detalhada) }),
    ]);
    const c = await rc.json(), a = await ra.json();
    $('req').textContent = 'POST /conta\\n' + JSON.stringify(corpo, null, 2);
    $('res').textContent = `HTTP ${rc.status}\\n` + JSON.stringify(c, null, 2) + `\\n\\nPOST /analisar → HTTP ${ra.status}\\n` + JSON.stringify(a, null, 2);
    if (!rc.ok || !ra.ok) {
      const e = rc.ok ? a : c; const msg = typeof e.detail === 'string' ? e.detail : (e.detail || []).map(x => x.msg).join('; ');
      box.innerHTML = `<div class="erro-box">Algo não deu certo: ${msg}</div>`; return;
    }""",
"""    await motorPronto;
    let c, a;
    try { c = Motor.conta(corpo, estado.cidade); a = c.analise; }
    catch (erro) { box.innerHTML = `<div class="erro-box">Algo não deu certo: ${erro.message}</div>`; return; }
    $('req').textContent = 'Motor.conta(entrada)\\n' + JSON.stringify(corpo, null, 2);
    $('res').textContent = JSON.stringify({ ...c, analise: undefined }, null, 2) + '\\n\\nMotor.analisar(casa)\\n' + JSON.stringify(a, null, 2);""")

troca("<tr><td>1</td><td>O front envia a casa descrita pela pessoa (andares, formato, frente, área de piso e janelas medidas) para <code>POST /conta</code>.</td></tr>",
      "<tr><td>1</td><td>Versão estática: os modelos treinados em Python (scikit-learn) foram exportados para <code>modelos.json</code> e o cálculo roda no navegador em <code>motor.js</code>, com os mesmos resultados da API FastAPI.</td></tr>")
troca("<code>/analisar</code> converte", "A análise converte")
troca('Valores e fontes em <code>api/tarifas_goias.json</code>.', 'Valores e fontes em <code>modelos.json</code> (copiados de <code>api/tarifas_goias.json</code>).')
troca("""    <p style="font-size:.85rem;color:var(--muted)">Swagger com todos os endpoints: <a href="/docs" target="_blank">/docs</a>.</p>""",
      """    <p style="font-size:.85rem;color:var(--muted)">SVR: soma dos coeficientes duais × exp(−γ·‖x − vetor de suporte‖²) + intercepto. MLP: duas camadas ReLU e saída linear. As entradas passam pelo StandardScaler do treino (média e desvio exportados). A previsão é feita no seu navegador, e só o CEP é enviado ao ViaCEP.</p>""")

(RAIZ / "docs" / "index.html").write_text(t, encoding="utf-8")
print("docs/index.html gerado")
