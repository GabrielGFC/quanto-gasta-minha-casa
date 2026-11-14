"""API de previsão do consumo energético (FastAPI).

Executar na raiz do projeto:  uvicorn api.main:app --reload
Front:   http://127.0.0.1:8000/
Swagger: http://127.0.0.1:8000/docs
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split

import json
import re
import urllib.request

from api.schemas import CasaDetalhada, Conta, Edificacao
from api.treinamento import BASE, COLUNAS, ENTRADAS, PASTA_MODELOS, treinar

app = FastAPI(title="API – Previsão do consumo energético",
              description="SVR, Rede Neural (MLP) e SVC treinados na base ENB2012 (UCI).",
              version="1.0")
modelos = {}
base = {}
FRONT = Path(__file__).resolve().parent.parent / "front" / "index.html"


@app.on_event("startup")
def carregar_modelos():
    if not (PASTA_MODELOS / "svr_resf.pkl").exists():
        treinar()  # primeira execução (ou versão antiga sem o modelo de resfriamento): treina e salva
    # joblib/pickle só é seguro com arquivos confiáveis: aqui são gerados pelo próprio treinamento.py
    for nome in ["scaler", "svr", "mlp", "svc", "limites", "svr_resf", "metricas_resf"]:
        modelos[nome] = joblib.load(PASTA_MODELOS / f"{nome}.pkl")
    carregar_base()


def carregar_base():
    """Guarda a base original e marca quais casas foram usadas no treino e quais no teste."""
    dados = pd.read_excel(BASE)
    dados.columns = COLUNAS
    # mesmo sorteio do treinamento (768 linhas, test_size=0.2, random_state=42)
    _, idx_teste = train_test_split(dados.index, test_size=0.2, random_state=42)
    dados["conjunto"] = "treino"
    dados.loc[idx_teste, "conjunto"] = "teste"
    base["dados"] = dados


def faixa_de(valor: float) -> str:
    l1, l2 = modelos["limites"]
    return "Baixo" if valor <= l1 else ("Médio" if valor <= l2 else "Alto")


def buscar_na_base(casa: Edificacao):
    """Procura a casa com exatamente as mesmas 8 características na base ENB2012."""
    d = base["dados"]
    filtro = np.ones(len(d), dtype=bool)
    for c in ENTRADAS:
        filtro &= np.isclose(d[c].to_numpy(dtype=float), float(getattr(casa, c)))
    achadas = d[filtro]
    return None if achadas.empty else achadas.iloc[0]


def preparar(casa: Edificacao):
    if not modelos:
        raise HTTPException(503, "Modelos ainda não carregados")
    x = np.array([[getattr(casa, c) for c in ENTRADAS]], dtype=float)
    return modelos["scaler"].transform(x)  # mesmo padronizador do treino


@app.get("/", include_in_schema=False)
def front():
    return FileResponse(FRONT)


@app.get("/health", tags=["Sistema"])
def health():
    return {"status": "ok" if modelos else "carregando", "modelos_carregados": len(modelos) == 7}


@app.get("/modelos", tags=["Sistema"])
def listar_modelos():
    return {
        "modelos": [
            {"nome": "SVR", "tarefa": "regressão", "params": "RBF, C=1000, ε=0,1, γ=0,1", "mae_teste": 0.34, "r2_teste": 0.998},
            {"nome": "MLP", "tarefa": "regressão", "params": "camadas (64, 32), ReLU, Adam", "mae_teste": 0.45, "r2_teste": 0.996},
            {"nome": "SVC", "tarefa": "classificação", "params": "RBF, C=10", "acuracia_teste": 0.909},
        ],
        "faixas_kwh_m2": {"Baixo": f"até {modelos['limites'][0]:.2f}",
                          "Médio": f"{modelos['limites'][0]:.2f} a {modelos['limites'][1]:.2f}",
                          "Alto": f"acima de {modelos['limites'][1]:.2f}"},
    }


@app.post("/prever/svr", tags=["Previsão"])
def prever_svr(casa: Edificacao):
    valor = float(modelos["svr"].predict(preparar(casa))[0])
    return {"modelo": "SVR", "carga_aquecimento_prevista": round(valor, 2), "unidade": "kWh/m²"}


@app.post("/prever/mlp", tags=["Previsão"])
def prever_mlp(casa: Edificacao):
    valor = float(modelos["mlp"].predict(preparar(casa))[0])
    return {"modelo": "MLP", "carga_aquecimento_prevista": round(valor, 2), "unidade": "kWh/m²"}


@app.post("/classificar/svc", tags=["Previsão"])
def classificar_svc(casa: Edificacao):
    return {"modelo": "SVC", "faixa_consumo": str(modelos["svc"].predict(preparar(casa))[0])}


@app.post("/prever", tags=["Previsão"])
def prever_todos(casa: Edificacao):
    """Roda os três modelos de uma vez e compara com o valor registrado na base (usado pelo front)."""
    resposta = {"svr": prever_svr(casa), "mlp": prever_mlp(casa), "svc": classificar_svc(casa)}
    resposta["referencia_base"] = referencia(casa, resposta)
    return resposta


def referencia(casa: Edificacao, previsao: dict):
    linha = buscar_na_base(casa)
    if linha is None:
        return {"encontrada": False,
                "mensagem": "Combinação que não existe na base: só há previsão do modelo, sem valor real para comparar."}
    real = float(linha["carga_aquecimento"])
    svr = previsao["svr"]["carga_aquecimento_prevista"]
    mlp = previsao["mlp"]["carga_aquecimento_prevista"]
    return {
        "encontrada": True,
        "linha_planilha": int(linha.name) + 2,  # +1 do cabeçalho, +1 porque o Excel começa em 1
        "conjunto": str(linha["conjunto"]),
        "carga_aquecimento_real": round(real, 2),
        "carga_resfriamento_real": round(float(linha["carga_resfriamento"]), 2),
        "faixa_real": faixa_de(real),
        "erro_svr": round(svr - real, 2),
        "erro_mlp": round(mlp - real, 2),
        "erro_percentual_svr": round(abs(svr - real) / real * 100, 1),
    }


TARIFAS = Path(__file__).resolve().parent / "tarifas_goias.json"

NIVEIS_VIDRO = [0.0, 0.10, 0.25, 0.40]          # os únicos valores de vidro que existem na base
NOME_DIR = {2: "norte", 3: "leste", 4: "sul", 5: "oeste"}
ORDEM_LADOS = ["frente", "direita", "fundos", "esquerda"]
NA = {"frente": "na frente", "direita": "na direita", "fundos": "nos fundos", "esquerda": "na esquerda"}   # sentido horário, para quem sai pela porta


def formatos_da_base(andares: int):
    """Os 6 formatos reais da base para 1 ou 2 andares, do mais compacto ao mais alongado."""
    d = base["dados"]
    altura = 7.0 if andares == 2 else 3.5
    f = d[np.isclose(d["altura_total"], altura)].groupby("compacidade_relativa").first().reset_index()
    return f.sort_values("compacidade_relativa", ascending=False).to_dict("records")


@app.post("/analisar", tags=["Previsão"])
def analisar(casa: CasaDetalhada):
    """Converte as medidas reais das janelas para as variáveis da base e prevê aquecimento e resfriamento.

    1. Área de vidro de cada lado = quantidade × largura × altura.
    2. Percentual de vidro = vidro total ÷ área de piso (é assim que a base mede).
    3. A base só tem 0%, 10%, 25% e 40%. Para valores no meio, o modelo é rodado nos dois níveis vizinhos
       e o resultado é interpolado (média ponderada pela distância).
    4. Distribuição: se um lado tem 40% ou mais de todo o vidro, as janelas contam como concentradas nesse lado;
       senão, contam como espalhadas.
    """
    apto = casa.tipo == "apartamento"
    # apartamento = uma unidade de um pavimento só (3,5 m de pé-direito na base)
    forma = formatos_da_base(1 if apto else casa.andares)[casa.formato]
    passos = []

    # lados -> direção cardeal, a partir da direção da frente
    i0 = casa.orientacao - 2
    cardeal = {lado: (i0 + k) % 4 + 2 for k, lado in enumerate(ORDEM_LADOS)}
    por_lado = {lado: 0.0 for lado in ORDEM_LADOS}
    for j in casa.janelas:
        area = j.quantidade * j.largura_m * j.altura_m
        por_lado[j.lado] += area
        passos.append(f"{j.quantidade} janela(s) de {j.largura_m:g} m × {j.altura_m:g} m {NA[j.lado]} "
                      f"({NOME_DIR[cardeal[j.lado]]}) = {area:.2f} m² de vidro")
    for p in casa.portas:
        area = p.quantidade * p.largura_m * p.altura_m
        if p.material == "vidro":
            por_lado[p.lado] += area
            passos.append(f"{p.quantidade} porta(s) de vidro de {p.largura_m:g} m × {p.altura_m:g} m {NA[p.lado]} "
                          f"= {area:.2f} m², que contam como vidro")
        else:
            passos.append(f"{p.quantidade} porta(s) de {p.material} {NA[p.lado]} ({NOME_DIR[cardeal[p.lado]]}): "
                          f"não deixam o calor passar como o vidro, então contam como parede")
    vidro = sum(por_lado.values())
    pct_real = vidro / casa.area_piso_m2
    passos.append(f"Vidro total: {vidro:.2f} m² em {casa.area_piso_m2:g} m² de piso = {pct_real*100:.1f}% do piso")

    pct = min(pct_real, NIVEIS_VIDRO[-1])
    if pct_real > NIVEIS_VIDRO[-1]:
        passos.append("Isso passa de 40%, o máximo que existe na base. O cálculo usa 40%.")

    if vidro == 0:
        dist = 0
    else:
        lado_max = max(por_lado, key=por_lado.get)
        dist = cardeal[lado_max] if por_lado[lado_max] / vidro >= 0.4 else 1
        passos.append(f"O vidro está concentrado {NA[lado_max]} ({NOME_DIR[dist]})" if dist != 1
                      else "As janelas estão espalhadas pelos lados da casa")

    def entrada_para(nivel):
        return Edificacao(compacidade_relativa=forma["compacidade_relativa"], area_superficie=forma["area_superficie"],
                          area_parede=forma["area_parede"], area_telhado=forma["area_telhado"],
                          altura_total=forma["altura_total"], orientacao=casa.orientacao,
                          area_janelas=nivel, distribuicao_janelas=0 if nivel == 0 else max(dist, 1))

    k = max(i for i, n in enumerate(NIVEIS_VIDRO) if n <= pct + 1e-9)
    baixo = NIVEIS_VIDRO[k]
    alto = NIVEIS_VIDRO[min(k + 1, len(NIVEIS_VIDRO) - 1)]
    peso = 0.0 if alto == baixo else (pct - baixo) / (alto - baixo)

    def prever(modelo, nivel):
        return float(modelos[modelo].predict(preparar(entrada_para(nivel)))[0])

    resultado = {}
    for nome, modelo in [("aquecimento", "svr"), ("aquecimento_mlp", "mlp"), ("resfriamento", "svr_resf")]:
        resultado[nome] = (1 - peso) * prever(modelo, baixo) + peso * prever(modelo, alto)
    if peso > 0:
        passos.append(f"A base só tem 0%, 10%, 25% e 40% de vidro. Como {pct*100:.1f}% fica entre {baixo*100:g}% e {alto*100:g}%, "
                      f"o modelo calculou os dois casos e fez a média ponderada ({(1-peso)*100:.0f}% / {peso*100:.0f}%).")

    # Apartamento: a base só tem casas isoladas. Ajuste físico (fora da base): calor entra e sai pelas
    # superfícies em contato com o lado de fora. Paredes vizinhas a outros apartamentos e o teto com
    # outro apartamento em cima trocam pouco calor, então a carga é reduzida na proporção da área exposta.
    fator = 1.0
    if apto:
        parede, teto = forma["area_parede"], forma["area_telhado"]
        ultimo = casa.andar_apartamento == casa.andares_predio - 1
        exposta = parede * len(set(casa.lados_externos)) / 4 + (teto if ultimo else 0)
        fator = exposta / (parede + teto)
        passos.append(
            f"Apartamento no {'térreo' if casa.andar_apartamento == 0 else str(casa.andar_apartamento) + 'º andar'} de um prédio de "
            f"{casa.andares_predio} andar(es), com {len(set(casa.lados_externos))} lado(s) para fora"
            + (" e o teto encostando no telhado" if ultimo else " e outro apartamento em cima") + ".")
        passos.append(
            f"As 768 casas da base são isoladas. Para um apartamento, o sistema mantém só a parte do calor que passa "
            f"pelas superfícies que dão para fora: {exposta:.0f} m² de {parede + teto:.0f} m² = {fator*100:.0f}% da casa equivalente. "
            "Esse ajuste é uma aproximação nossa, não vem da base.")
        for chave in resultado:
            resultado[chave] *= fator

    faixa = faixa_de(resultado["aquecimento"])
    vizinho = buscar_na_base(entrada_para(baixo if peso < 0.5 else alto))
    return {
        "aquecimento_kwh_m2_ano": round(resultado["aquecimento"], 2),
        "aquecimento_mlp_kwh_m2_ano": round(resultado["aquecimento_mlp"], 2),
        "resfriamento_kwh_m2_ano": round(resultado["resfriamento"], 2),
        "faixa": faixa,
        "tipo": casa.tipo,
        "fator_apartamento": round(fator, 3),
        "vidro": {"total_m2": round(vidro, 2), "percentual_real": round(pct_real * 100, 1),
                  "por_lado_m2": {l: round(v, 2) for l, v in por_lado.items()},
                  "direcao_de_cada_lado": {l: NOME_DIR[c] for l, c in cardeal.items()},
                  "distribuicao_usada": dist, "niveis_interpolados": [baixo, alto], "peso": round(peso, 3)},
        "formato_usado": {k2: forma[k2] for k2 in ("compacidade_relativa", "area_superficie", "area_parede", "area_telhado", "altura_total")},
        "casa_mais_parecida_na_base": None if vizinho is None else {
            "linha_planilha": int(vizinho.name) + 2, "carga_aquecimento_real": round(float(vizinho["carga_aquecimento"]), 2),
            "carga_resfriamento_real": round(float(vizinho["carga_resfriamento"]), 2), "conjunto": str(vizinho["conjunto"])},
        "passos": passos,
    }


def consultar_cep(cep: str):
    """Confere o CEP no ViaCEP; se a consulta falhar, usa a faixa de CEPs de Goiás (72800-000 a 76799-999)."""
    cfg = json.loads(TARIFAS.read_text(encoding="utf-8"))
    ini, fim = cfg["faixa_cep_goias"]
    try:
        with urllib.request.urlopen(f"https://viacep.com.br/ws/{cep}/json/", timeout=4) as r:
            d = json.load(r)
        if d.get("erro"):
            raise HTTPException(422, f"CEP {cep} não encontrado")
        return {"cidade": d.get("localidade"), "uf": d.get("uf"), "fonte": "ViaCEP"}
    except HTTPException:
        raise
    except Exception:
        return {"cidade": None, "uf": "GO" if ini <= cep <= fim else "??", "fonte": "faixa de CEP (ViaCEP indisponível)"}


@app.post("/conta", tags=["Conta de luz"])
def estimar_conta(dados: Conta):
    """Estima a conta de luz mensal em Goiás a partir das cargas previstas pelos modelos."""
    cep = re.sub(r"\D", "", dados.cep)
    local = consultar_cep(cep)
    if local["uf"] != "GO":
        raise HTTPException(422, f"O CEP {cep} é de {local['uf']}. As tarifas configuradas são da Equatorial Goiás.")
    cfg = json.loads(TARIFAS.read_text(encoding="utf-8"))

    if dados.detalhada is not None:
        analise = analisar(dados.detalhada)
        aquec_m2, resf_m2 = analise["aquecimento_kwh_m2_ano"], analise["resfriamento_kwh_m2_ano"]
    elif dados.casa is not None:
        x = preparar(dados.casa)
        aquec_m2 = float(modelos["svr"].predict(x)[0])        # kWh térmicos por m² por ano
        resf_m2 = float(modelos["svr_resf"].predict(x)[0])
    else:
        raise HTTPException(422, "Envie 'casa' ou 'detalhada'")

    # carga térmica (energia de calor) -> energia elétrica, dividindo pela eficiência dos aparelhos
    resf_eletr_ano = resf_m2 * dados.area_climatizada_m2 / dados.eficiencia_ar
    aquec_eletr_ano = aquec_m2 * dados.area_climatizada_m2 / dados.eficiencia_aquecedor if dados.usa_aquecimento else 0.0
    clima_mes = (resf_eletr_ano + aquec_eletr_ano) / 12
    total_kwh = clima_mes + dados.outros_aparelhos_kwh_mes

    tarifa = cfg["tarifa_b1_sem_impostos"]["valor"]
    bandeira = cfg["bandeiras"][dados.bandeira]
    icms, pis = cfg["icms"]["valor"], cfg["pis_cofins"]["valor"]
    # cálculo "por dentro", como na conta: preço final = preço sem impostos / (1 - ICMS - PIS/COFINS)
    tarifa_final = (tarifa + bandeira) / (1 - icms - pis)
    energia = total_kwh * tarifa_final
    total = energia + dados.iluminacao_publica

    return {
        "local": {**local, "cep": cep},
        "previsao_modelos": {"aquecimento_kwh_m2_ano": round(aquec_m2, 2), "resfriamento_kwh_m2_ano": round(resf_m2, 2)},
        "consumo_kwh_mes": {
            "ar_condicionado": round(resf_eletr_ano / 12, 1),
            "aquecimento": round(aquec_eletr_ano / 12, 1),
            "outros_aparelhos": round(dados.outros_aparelhos_kwh_mes, 1),
            "total": round(total_kwh, 1),
        },
        "tarifa": {
            "sem_impostos": tarifa, "bandeira": dados.bandeira, "adicional_bandeira": bandeira,
            "icms": icms, "pis_cofins": pis, "final_com_impostos": round(tarifa_final, 4),
        },
        "valor_mensal": {
            "energia_com_impostos": round(energia, 2),
            "iluminacao_publica": round(dados.iluminacao_publica, 2),
            "total": round(total, 2),
            "climatizacao": round(clima_mes * tarifa_final, 2),
        },
        "fontes": {k: cfg[k]["fonte"] for k in ("tarifa_b1_sem_impostos", "icms", "pis_cofins")} | {"bandeiras": cfg["bandeiras"]["_fonte"]},
        "avisos": [
            "As cargas de aquecimento e resfriamento vêm de casas simuladas no clima de Atenas (base ENB2012). "
            "O clima de Goiás é mais quente, então o consumo real de ar-condicionado tende a ser maior e o de aquecimento, menor.",
            "Média mensal = consumo anual ÷ 12. Nos meses mais quentes a conta tende a ser maior.",
            "A tarifa sem impostos vem de fonte não oficial; confira o valor na sua conta.",
        ],
    }


@app.post("/simular", tags=["Previsão"])
def simular(casa: Edificacao):
    """Cenários 'e se': muda uma característica por vez, mantendo as outras, e mostra o efeito no SVR.
    Só gera combinações que existem na base, então todo cenário tem valor real para conferir."""
    atual = prever_svr(casa)["carga_aquecimento_prevista"]
    d = base["dados"]
    mesma_forma = d[np.isclose(d["compacidade_relativa"], casa.compacidade_relativa)]
    cenarios = []

    def add(rotulo, grupo, **mudancas):
        nova = casa.model_copy(update=mudancas)
        if (nova.area_janelas == 0) != (nova.distribuicao_janelas == 0):
            return
        if nova == casa:
            return
        valor = prever_svr(nova)["carga_aquecimento_prevista"]
        linha = buscar_na_base(nova)
        cenarios.append({
            "grupo": grupo, "cenario": rotulo, "svr": valor, "diferenca": round(valor - atual, 2),
            "real_base": None if linha is None else round(float(linha["carga_aquecimento"]), 2),
        })

    for v, n in [(0.0, "Sem janelas"), (0.1, "Com 10% de vidro"), (0.25, "Com 25% de vidro"), (0.4, "Com 40% de vidro")]:
        add(n, "Vidro", area_janelas=v,
            distribuicao_janelas=0 if v == 0 else (casa.distribuicao_janelas or 1))
    for v, n in [(2, "norte"), (3, "leste"), (4, "sul"), (5, "oeste")]:
        add(f"Frente para o {n}", "Orientação", orientacao=v)
    if casa.area_janelas > 0:
        for v, n in [(1, "espalhadas"), (2, "no norte"), (3, "no leste"), (4, "no sul"), (5, "no oeste")]:
            add(f"Janelas {n}", "Distribuição", distribuicao_janelas=v)

    # mesma casa com o formato vizinho (mais compacto e mais alongado), se existir na base
    formas = sorted(d.groupby("compacidade_relativa").first().reset_index().itertuples(), key=lambda r: r.compacidade_relativa)
    comp = [f.compacidade_relativa for f in formas]
    i = int(np.argmin(np.abs(np.array(comp) - casa.compacidade_relativa)))
    for j, rot in [(i + 1, "Formato um passo mais compacto"), (i - 1, "Formato um passo mais alongado")]:
        if 0 <= j < len(formas):
            f = formas[j]
            add(f"{rot} (compacidade {f.compacidade_relativa:.2f}, {'2 andares' if f.altura_total == 7 else '1 andar'})", "Formato",
                compacidade_relativa=f.compacidade_relativa, area_superficie=f.area_superficie, area_parede=f.area_parede,
                area_telhado=f.area_telhado, altura_total=f.altura_total)

    faixa_forma = mesma_forma["carga_aquecimento"]
    return {
        "previsao_atual_svr": atual,
        "cenarios": cenarios,
        "mesmo_formato_na_base": {
            "quantidade_de_casas": int(len(mesma_forma)),
            "minimo": round(float(faixa_forma.min()), 2),
            "media": round(float(faixa_forma.mean()), 2),
            "maximo": round(float(faixa_forma.max()), 2),
        },
    }
