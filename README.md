# Trabalho A2 — Data Science

Classificação binária supervisionada em três domínios (spam, fraude bancária e exame médico) e um programa reutilizável de **faixas de decisão por histograma**.

- **Parte 1 — `src/cutoff_hist.py`**: recebe rótulos reais (0/1) e a probabilidade estimada da classe positiva de qualquer modelo binário. Gera o histograma (azul = todas as instâncias, vermelho = instâncias com rótulo real positivo, **as duas sobre o mesmo total N**), escolhe dois cortes t1 < t2 nos dados de validação e avalia as três faixas: negativa automática (p < t1), análise manual (t1 ≤ p < t2) e positiva automática (p ≥ t2).
- **Parte 2 — `src/pipeline.py`, `src/evaluate.py`, `src/explain.py` + notebooks**: split 60/20/20 estratificado, três algoritmos por domínio com tuning por validação cruzada no treino, comparação na validação, avaliação final no teste, curvas ROC/PR, SHAP beeswarm e aplicação da Parte 1 ao melhor modelo.

Decisões de implementação não cobertas pela especificação estão em [`DECISOES.md`](DECISOES.md).

## Estrutura

```
a2/
├── data/raw/            # arquivos brutos (colocados manualmente)
├── data/*.csv           # bases padronizadas (geradas por prepare_data.py)
├── src/                 # config, prepare_data, cutoff_hist (Parte 1), pipeline, evaluate, explain
├── notebooks/           # 01_spam, 02_fraude, 03_exame_medico
├── tests/               # testes da Parte 1 (pytest)
└── outputs/<dominio>/   # figuras, tabelas (CSV + MD), modelos, resumo_resultados.md
```

## 1. Dados brutos

O código **não baixa dados**. Coloque os arquivos em `data/raw/`:

| Domínio | Arquivo em `data/raw/` | Onde baixar |
|---|---|---|
| Spam | `SMSSpamCollection` (dentro do zip) | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Fraude bancária | `creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (requer login no Kaggle) |
| Exame médico | nenhum: carregado por `sklearn.datasets.load_breast_cancer()` | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |

Depois, a partir da raiz `a2/`:

```bash
python -m src.prepare_data            # gera data/spam.csv, data/fraude.csv (+ fraude.zip), data/exame_medico.csv
```

O script padroniza a coluna-alvo como `target` (1 = classe positiva), remove duplicatas exatas e imprime shape, contagem e percentual por classe. No exame médico, o alvo do scikit-learn (0 = maligno) é invertido para **1 = maligno**. O `data/fraude.csv` tem ~150 MB; o `data/fraude.zip` (~69 MB) é a versão para versionar/entregar.

## 2. Fontes e licenças

| Base | Fonte | Classe positiva | Licença |
|---|---|---|---|
| SMS Spam Collection | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/228/sms+spam+collection (Almeida & Hidalgo, 2011) | é spam? | LICENÇA: <preencher conforme página oficial> |
| Credit Card Fraud Detection | Kaggle, Machine Learning Group – ULB — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | é fraude? | LICENÇA: <preencher conforme página oficial> |
| Breast Cancer Wisconsin (Diagnostic) | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (carregada via scikit-learn) | é maligno? | LICENÇA: <preencher conforme página oficial> |

## 3. Como rodar no Google Colab

1. Suba a pasta `a2/` inteira para o Google Drive (ex.: `MyDrive/a2`), já com os arquivos brutos em `data/raw/`.
2. Abra um notebook no Colab. Se a pasta estiver em outro lugar, ajuste `PROJECT_DIR` na primeira célula.
3. Gere as bases uma vez (numa célula do Colab, após montar o Drive):
   ```python
   %cd /content/drive/MyDrive/a2
   !pip install -q -r requirements.txt
   !python -m src.prepare_data
   ```
4. Rode os notebooks de cima a baixo, nesta ordem: `03_exame_medico.ipynb`, `01_spam.ipynb`, `02_fraude.ipynb`.

Tudo é salvo em `outputs/<dominio>/` no Drive. Após o tuning, cada modelo é salvo em `outputs/<dominio>/models/<modelo>.joblib`; se o runtime desconectar, basta rodar o notebook de novo e os modelos já treinados são carregados. Para forçar o retreino, use `FORCE_RETRAIN = True` no topo do notebook.

### Tempo aproximado de execução

| Notebook | Local (medido) | Colab gratuito (estimado, ~2 núcleos) |
|---|---|---|
| `03_exame_medico` | ~1 min | 2–4 min |
| `01_spam` | ~1 min | 3–6 min |
| `02_fraude` | ~5 min numa amostra de 15% (base completa não rodada localmente) | 2–4 h (só o tuning da Random Forest pode passar de 2 h; os checkpoints evitam perder o que já terminou) |

Com os checkpoints já salvos, qualquer notebook roda em poucos segundos (spam e exame médico) ou poucos minutos (fraude, por causa do SHAP e das predições).

## 4. Parte 1 isoladamente

Como função, com qualquer modelo que forneça `predict_proba`:

```python
import pandas as pd
from src.cutoff_hist import plot_cutoff_histogram, band_report, choose_cutoffs

val = pd.read_csv("outputs/spam/preds_validacao.csv")     # colunas y_true, y_proba
test = pd.read_csv("outputs/spam/preds_teste.csv")

# cortes escolhidos SOMENTE na validação
(t1, t2), rep_val, tradeoff = choose_cutoffs(val.y_true, val.y_proba,
                                             max_fn_rate=0.10, max_fp_rate=0.01)
# aplicados ao teste sem ajuste
rep_test = band_report(test.y_true, test.y_proba, t1, t2)
plot_cutoff_histogram(test.y_true, test.y_proba, bin_width=0.1, t1=t1, t2=t2,
                      positive_label="spam", title="Faixas de decisão — spam",
                      dataset_name="teste", save_path="faixas_spam_teste.png")
```

Pela linha de comando (o CSV precisa das colunas `y_true` e `y_proba`):

```bash
python -m src.cutoff_hist --csv outputs/spam/preds_teste.csv --bin 0.1 --t1 0.3 --t2 0.7 --label "spam" --out fig.png
```

O CLI gera o gráfico e imprime o `band_report` (tabela por faixa + erros e cobertura, com numerador e denominador explícitos).

## 5. Testes

```bash
pytest tests/
```

Cobrem: p = 0 e p = 1 em bins válidos; soma das barras azuis = 100%; barras vermelhas somando (positivos ÷ N); p = t1 na faixa manual e p = t2 na positiva automática; larguras de bin 0,05 / 0,1 / 0,2; e `choose_cutoffs` respeitando as restrições (inclusive o aviso quando não há par viável).

## 6. Saídas por domínio (`outputs/<dominio>/`)

| Arquivo | Conteúdo |
|---|---|
| `balanceamento.*`, `01_balanceamento_classes.png` | contagem e % de cada classe |
| `split.*` | tamanho e % de positivos de treino / validação / teste |
| `hiperparametros.*` | melhores hiperparâmetros, média ± desvio da CV, tempo de tuning |
| `comparacao_validacao.*`, `02_matriz_confusao_validacao_*.png` | métricas de todos os modelos na validação (limiar 0,5) |
| `03_curva_roc_validacao.png`, `04_curva_pr_validacao.png` | curvas de todos os modelos na validação |
| `varredura_limiar.*` | TPR, FPR, precisão e recall do melhor modelo para limiares 0,1 … 0,9 |
| `resultado_teste.*`, `05`–`07_*.png` | melhor modelo no teste: métricas, matriz de confusão, ROC e PR |
| `08_shap_beeswarm.png`, `shap_top10.*`, `shap_info.txt` | SHAP do melhor modelo |
| `faixas/` | trade-off e `band_report` (validação e teste), histogramas `09` e `10` |
| `preds_validacao.csv`, `preds_teste.csv` | `y_true`, `y_proba` (entrada do CLI da Parte 1) |
| `resumo_resultados.md` | resumo só com números e tabelas |
