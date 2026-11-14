#!/bin/bash

set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo ">> Criando o ambiente virtual (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo ">> Instalando as bibliotecas..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f "ENB2012_data.xlsx" ]; then
    echo ">> Baixando a base de dados da UCI..."
    curl -sSL -o energy_efficiency.zip "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
    unzip -o -q energy_efficiency.zip ENB2012_data.xlsx
    rm energy_efficiency.zip
fi

echo ">> Executando o notebook..."
jupyter nbconvert --to notebook --execute --inplace projeto_previsao_energia.ipynb

echo ">> Pronto! Abra o projeto_previsao_energia.ipynb no VS Code para ver os resultados."
