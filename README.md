# Previsão do consumo energético de edificações (SVR, MLP e SVC)

Trabalho de Inteligência Artificial, tema 05, por Amábile Honorato Zucchetti e Gabriel Fernandes de Carvalho.
Base: Energy Efficiency (UCI), 768 edificações simuladas. Alvo: carga de aquecimento (kWh/m²).

## Estrutura
| Caminho | Conteúdo |
|---|---|
| `projeto_previsao_energia.ipynb` | Notebook: análise, treino, GridSearch e avaliação dos modelos |
| `api/treinamento.py` | Treina SVR, MLP e SVC (mesma configuração do notebook) e salva em `modelos_salvos/` |
| `api/schemas.py` | Validação das entradas (regras de negócio) |
| `api/main.py` | API FastAPI com os endpoints |
| `api/tarifas_goias.json` | Tarifa, ICMS, PIS/COFINS e bandeiras de Goiás, com fonte e data de consulta |
| `front/index.html` | Front interativo que consome a API |
| `Trabalho_IA_Tema05_Amabile_Gabriel.docx` | Documento teórico e prático |

## Como rodar
- Notebook: `rodar.bat` (Windows) ou `rodar.sh`.
- API + front: `rodar_api.bat` (Windows) ou `rodar_api.sh`. Depois abra:
  - http://127.0.0.1:8000: front interativo
  - http://127.0.0.1:8000/docs: Swagger para testar os endpoints

Na primeira execução, a API treina os modelos e salva os arquivos `.pkl`. Esse treino leva poucos segundos.

## Endpoints
| Método | Rota | Função |
|---|---|---|
| GET | `/health` | Estado da API e dos modelos |
| GET | `/modelos` | Modelos disponíveis, parâmetros, métricas e faixas |
| POST | `/prever/svr` | Carga de aquecimento prevista pelo SVR |
| POST | `/prever/mlp` | Carga de aquecimento prevista pela Rede Neural |
| POST | `/classificar/svc` | Faixa de consumo (Baixo, Médio ou Alto) |
| POST | `/prever` | Os três modelos de uma vez, mais o valor real da mesma casa na base (linha da planilha, treino/teste, erro) |
| POST | `/analisar` | Casa ou apartamento descrito com janelas e portas medidas (lado, quantidade, largura, altura, material): converte para as variáveis da base e explica cada passo |
| POST | `/conta` | Estimativa da conta de luz em Goiás: CEP, área climatizada, eficiência do ar, outros aparelhos, bandeira |
| POST | `/simular` | Cenários "e se": muda uma característica por vez e compara a previsão do SVR com o valor real da base |

Exemplo de corpo:
```json
{"compacidade_relativa": 0.98, "area_superficie": 514.5, "area_parede": 294, "area_telhado": 110.25,
 "altura_total": 7, "orientacao": 2, "area_janelas": 0, "distribuicao_janelas": 0}
```

## Regras de negócio (validadas, com erro HTTP 422)
- As 8 características são obrigatórias e numéricas.
- Os valores contínuos devem estar dentro dos limites da base. Exemplo: compacidade de 0,62 a 0,98.
- Os valores discretos só aceitam o que existe na base:
  - altura: 3,5 ou 7;
  - orientação: 2 a 5;
  - janelas: 0, 0,10, 0,25 ou 0,40;
  - distribuição: 0 a 5.
- `area_janelas = 0` exige `distribuicao_janelas = 0`, e vice-versa.
- A entrada é padronizada com o mesmo StandardScaler ajustado no treino.
- Se os modelos não estiverem carregados, a API responde HTTP 503.

## Resultados no conjunto de teste
| Modelo | MAE | R² |
|---|---|---|
| Baseline (média) | 9,27 | −0,006 |
| SVR ajustado | 0,34 | 0,998 |
| Rede Neural (MLP) | 0,45 | 0,996 |

A acurácia do SVC foi de 0,909.

O documento `Logica_do_Sistema_e_Origem_dos_Dados.docx` explica a origem dos dados e a lógica completa do sistema.

## Produção (Docker)
A imagem instala só as dependências da API (`requirements-api.txt`, versões fixadas), treina os modelos durante o build e sobe o Uvicorn na porta da variável `PORT` (padrão 8000).

```bash
docker build -t quanto-gasta .
docker run -p 8000:8000 quanto-gasta
```

Para atualizar a tarifa, a bandeira ou os impostos, edite `api/tarifas_goias.json` e faça um novo deploy.

## Versão estática (GitHub Pages)
A pasta `docs/` tem uma versão que roda inteira no navegador, sem servidor Python. Os modelos treinados são exportados para `docs/modelos.json`, e `docs/motor.js` refaz no navegador a mesma lógica da API, com os mesmos resultados (comparados caso a caso com o Python). Só o CEP é enviado ao ViaCEP.

Para atualizar depois de mudar os modelos, a tarifa ou o front:
```bash
python -m api.exportar_estatico
python docs/gerar_site.py
```
No GitHub: Settings → Pages → Branch `main`, pasta `/docs`.
