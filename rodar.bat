@echo off

cd /d "%~dp0"

if not exist ".venv" (
    echo ^>^> Criando o ambiente virtual ^(.venv^)...
    python -m venv .venv || goto erro
)

call .venv\Scripts\activate.bat
echo ^>^> Instalando as bibliotecas...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt || goto erro

if not exist "ENB2012_data.xlsx" (
    echo ^>^> Baixando a base de dados da UCI...
    curl -sSL -o energy_efficiency.zip "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip" || goto erro
    tar -xf energy_efficiency.zip ENB2012_data.xlsx || goto erro
    del energy_efficiency.zip
)

echo ^>^> Executando o notebook...
jupyter nbconvert --to notebook --execute --inplace projeto_previsao_energia.ipynb || goto erro

echo ^>^> Feito! Abra o projeto_previsao_energia.ipynb no VS Code para ver os resultados.
pause
exit /b 0

:erro
echo ^>^> Ocorreu um erro. Veja a mensagem acima.
pause
exit /b 1
