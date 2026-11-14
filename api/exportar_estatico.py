"""Exporta os modelos treinados e os dados necessários para a versão estática (GitHub Pages).

Uso, na raiz do projeto:  python -m api.exportar_estatico
Gera docs/modelos.json, que o docs/index.html usa para fazer as previsões no navegador.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from api.treinamento import BASE, COLUNAS, PASTA_MODELOS, RAIZ, treinar

SAIDA = RAIZ / "docs" / "modelos.json"


def svr_para_dict(m):
    # previsão = soma(dual_coef_i * exp(-gamma * |x - sv_i|²)) + intercept
    return {"sv": m.support_vectors_.round(6).tolist(), "coef": m.dual_coef_[0].round(8).tolist(),
            "intercept": float(m.intercept_[0]), "gamma": float(m._gamma)}


def mlp_para_dict(m):
    return {"pesos": [w.round(7).tolist() for w in m.coefs_], "bias": [b.round(7).tolist() for b in m.intercepts_],
            "ativacao": m.activation}


def main():
    if not (PASTA_MODELOS / "svr_resf.pkl").exists():
        treinar()
    # joblib/pickle só é seguro com arquivos confiáveis: aqui são gerados pelo próprio treinamento.py
    carregar = lambda n: joblib.load(PASTA_MODELOS / f"{n}.pkl")
    scaler = carregar("scaler")

    dados = pd.read_excel(BASE); dados.columns = COLUNAS
    _, idx_teste = train_test_split(dados.index, test_size=0.2, random_state=42)
    teste = set(int(i) for i in idx_teste)
    base = [[*map(float, linha), 1 if i in teste else 0] for i, linha in enumerate(dados[COLUNAS].to_numpy())]

    tarifas = json.loads((RAIZ / "api" / "tarifas_goias.json").read_text(encoding="utf-8"))
    saida = {
        "colunas": COLUNAS,
        "scaler": {"media": scaler.mean_.tolist(), "escala": scaler.scale_.tolist()},
        "svr": svr_para_dict(carregar("svr")),
        "svr_resf": svr_para_dict(carregar("svr_resf")),
        "mlp": mlp_para_dict(carregar("mlp")),
        "limites": carregar("limites"),
        "metricas_resf": {k: v for k, v in carregar("metricas_resf").items() if k != "params"},
        "base": base,                     # 768 linhas: 8 entradas, 2 saídas e 1 = conjunto de teste
        "tarifas": tarifas,
    }
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("Gerado", SAIDA, round(SAIDA.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
