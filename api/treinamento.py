"""Treina os modelos (mesma configuração do notebook) e salva em modelos_salvos/."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

RAIZ = Path(__file__).resolve().parent.parent
BASE = RAIZ / "ENB2012_data.xlsx"
PASTA_MODELOS = RAIZ / "modelos_salvos"

COLUNAS = [
    "compacidade_relativa", "area_superficie", "area_parede", "area_telhado",
    "altura_total", "orientacao", "area_janelas", "distribuicao_janelas",
    "carga_aquecimento", "carga_resfriamento",
]
ENTRADAS = COLUNAS[:8]


def treinar():
    dados = pd.read_excel(BASE)
    dados.columns = COLUNAS
    X = dados[ENTRADAS]
    y = dados["carga_aquecimento"]

    X_treino, _, y_treino, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_treino_pad = scaler.fit_transform(X_treino)  # ajusta só no treino

    # SVR com os melhores parâmetros encontrados pelo GridSearchCV no notebook
    svr = SVR(kernel="rbf", C=1000, epsilon=0.1, gamma=0.1).fit(X_treino_pad, y_treino)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    mlp.fit(X_treino_pad, y_treino)

    limites = y.quantile([1 / 3, 2 / 3]).tolist()
    y_classe = pd.qcut(y, q=3, labels=["Baixo", "Médio", "Alto"])
    yc_treino, _ = train_test_split(y_classe, test_size=0.2, random_state=42)
    svc = SVC(kernel="rbf", C=10).fit(X_treino_pad, yc_treino)

    # SVR para a carga de RESFRIAMENTO (Y2), usada na estimativa da conta de luz (ar-condicionado).
    # Mesmo pré-processamento e mesma divisão treino/teste; hiperparâmetros escolhidos com GridSearchCV.
    y2 = dados["carga_resfriamento"]
    _, _, y2_treino, y2_teste = train_test_split(X, y2, test_size=0.2, random_state=42)
    X_teste_pad = scaler.transform(train_test_split(X, test_size=0.2, random_state=42)[1])
    busca = GridSearchCV(SVR(kernel="rbf"),
                         {"C": [10, 100, 1000], "epsilon": [0.1, 0.5], "gamma": ["scale", 0.1]},
                         cv=5, scoring="neg_mean_absolute_error").fit(X_treino_pad, y2_treino)
    svr_resf = busca.best_estimator_
    prev = svr_resf.predict(X_teste_pad)
    metricas_resf = {"params": busca.best_params_,
                     "mae_teste": round(float(mean_absolute_error(y2_teste, prev)), 2),
                     "r2_teste": round(float(r2_score(y2_teste, prev)), 3)}
    print("SVR resfriamento:", metricas_resf)

    PASTA_MODELOS.mkdir(exist_ok=True)
    for nome, obj in {"scaler": scaler, "svr": svr, "mlp": mlp, "svc": svc, "limites": limites,
                      "svr_resf": svr_resf, "metricas_resf": metricas_resf}.items():
        joblib.dump(obj, PASTA_MODELOS / f"{nome}.pkl")
    print("Modelos salvos em", PASTA_MODELOS)


if __name__ == "__main__":
    treinar()
