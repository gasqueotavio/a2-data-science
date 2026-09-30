# Resumo de resultados — exame_medico

## Balanceamento

| classe | n | pct_do_total |
|---|---|---|
| 1 = maligno (é maligno?) | 212 | 37.26 |
| 0 = benigno | 357 | 62.74 |
| Total | 569 | 100.0 |

## Split 60/20/20

| conjunto | n | pct_do_total | n_positivos | pct_positivos | n_negativos |
|---|---|---|---|---|---|
| treino | 341 | 59.93 | 127 | 37.24 | 214 |
| validacao | 114 | 20.04 | 43 | 37.72 | 71 |
| teste | 114 | 20.04 | 42 | 36.84 | 72 |

## Hiperparâmetros vencedores (CV 5 folds, roc_auc)

| modelo | metrica_cv | cv_roc_auc_media | cv_roc_auc_desvio | cv_folds | candidatos_testados | tempo_tuning_s | melhores_hiperparametros |
|---|---|---|---|---|---|---|---|
| Regressão Logística | roc_auc | 0.9941 | 0.0078 | 5 | 6 | 28.0622 | C=1 |
| Random Forest | roc_auc | 0.9874 | 0.01 | 5 | 48 | 17.8298 | max_depth=8; max_features=sqrt; min_samples_leaf=5; n_estimators=100 |
| SVC (RBF) | roc_auc | 0.9941 | 0.0083 | 5 | 16 | 0.2136 | C=10; gamma=0.01 |

## Comparação na validação (limiar = 0.5)

| modelo | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | 0.5 | 0.9737 | 0.9545 | 0.9767 | 0.9655 | 0.9957 | 0.9946 | 0.9946 | 69 | 2 | 1 | 42 |
| Random Forest | 0.5 | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9885 | 0.9864 | 0.9865 | 70 | 1 | 2 | 41 |
| SVC (RBF) | 0.5 | 0.9649 | 0.9333 | 0.9767 | 0.9545 | 0.9957 | 0.9946 | 0.9946 | 68 | 3 | 1 | 42 |

## Melhor modelo

Regressão Logística — maior roc_auc na validação = 0.9957

## Resultado no teste (limiar = 0.5)

| modelo | conjunto | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | teste | 0.5 | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9954 | 0.9935 | 0.9936 | 71 | 1 | 2 | 40 |

## SHAP — informações

- Domínio: exame_medico
- Modelo explicado: Regressão Logística (LogisticRegression)
- Explainer: LinearExplainer (interventional)
- Saída do modelo explicada: log-odds (escala linear) da classe positiva da Regressão Logística
- Dados explicados: 114 instâncias aleatórias do conjunto de TESTE (random_state=42), transformadas pelo pré-processamento do pipeline
- Conjunto de fundo (background): 100 instâncias aleatórias do TREINO
- Número de features após o pré-processamento: 30
- Beeswarm: max_display=15; cor = valor da feature (vermelho alto, azul baixo); eixo X = contribuição SHAP para a saída explicada

## SHAP — top 10 features (média |SHAP|)

| posicao | feature | media_abs_shap | correlacao_valor_shap | leitura |
|---|---|---|---|---|
| 1 | worst texture | 1.1791 | 1.0 | valores altos aumentam a saída |
| 2 | mean concave points | 0.8246 | 1.0 | valores altos aumentam a saída |
| 3 | worst symmetry | 0.6661 | 1.0 | valores altos aumentam a saída |
| 4 | radius error | 0.6407 | 1.0 | valores altos aumentam a saída |
| 5 | worst radius | 0.6103 | 1.0 | valores altos aumentam a saída |
| 6 | worst concavity | 0.5825 | 1.0 | valores altos aumentam a saída |
| 7 | compactness error | 0.5674 | -1.0 | valores altos diminuem a saída |
| 8 | worst area | 0.5582 | 1.0 | valores altos aumentam a saída |
| 9 | mean concavity | 0.5568 | 1.0 | valores altos aumentam a saída |
| 10 | worst perimeter | 0.4912 | 1.0 | valores altos aumentam a saída |

## Cortes (escolhidos na validação)

- t1 = 0.10
- t2 = 0.21
- max_fn_rate = 0.01 (sobre positivos)
- max_fp_rate = 0.1 (sobre negativos)

## Trade-off (10 melhores pares viáveis, validação)

| t1 | t2 | volume_manual | cobertura_automatica | fn_auto | fn_auto_sobre_positivos | fp_auto | fp_auto_sobre_negativos | violacao | viavel |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.21 | 0.0614 | 0.9386 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.1 | 0.22 | 0.0614 | 0.9386 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.1 | 0.23 | 0.0614 | 0.9386 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.1 | 0.24 | 0.0614 | 0.9386 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.1 | 0.25 | 0.0614 | 0.9386 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.1 | 0.26 | 0.0702 | 0.9298 | 0 | 0.0 | 6 | 0.0845 | 0.0 | True |
| 0.09 | 0.21 | 0.0702 | 0.9298 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.09 | 0.22 | 0.0702 | 0.9298 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.09 | 0.23 | 0.0702 | 0.9298 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |
| 0.09 | 0.24 | 0.0702 | 0.9298 | 0 | 0.0 | 7 | 0.0986 | 0.0 | True |

## band_report — validação — faixas

| faixa | regra | n | pct_populacao_sobre_N | n_positivos | pct_positivos_sobre_n_faixa | n_negativos | pct_negativos_sobre_n_faixa |
|---|---|---|---|---|---|---|---|
| Negativa automática | p < t1 | 58 | 50.88 | 0 | 0.0 | 58 | 100.0 |
| Análise manual | t1 ≤ p < t2 | 7 | 6.14 | 1 | 14.29 | 6 | 85.71 |
| Positiva automática | p ≥ t2 | 49 | 42.98 | 42 | 85.71 | 7 | 14.29 |
| Total | t1 = 0.1, t2 = 0.21 | 114 | 100.0 | 43 | 37.72 | 71 | 62.28 |

## band_report — validação — erros

| metrica | valor | valor_pct | numerador | denominador | descricao_denominador |
|---|---|---|---|---|---|
| fn_auto | 0.0 |  | 0 |  | — (contagem) |
| fn_auto_sobre_positivos | 0.0 | 0.0 | 0 | 43 | total de positivos |
| fn_auto_sobre_N | 0.0 | 0.0 | 0 | 114 | N (total de instâncias) |
| fp_auto | 7.0 |  | 7 |  | — (contagem) |
| fp_auto_sobre_negativos | 0.0986 | 9.86 | 7 | 71 | total de negativos |
| fp_auto_sobre_N | 0.0614 | 6.14 | 7 | 114 | N (total de instâncias) |
| cobertura_automatica | 0.9386 | 93.86 | 107 | 114 | N (total de instâncias) |
| volume_manual | 0.0614 | 6.14 | 7 | 114 | N (total de instâncias) |

## band_report — teste — faixas

| faixa | regra | n | pct_populacao_sobre_N | n_positivos | pct_positivos_sobre_n_faixa | n_negativos | pct_negativos_sobre_n_faixa |
|---|---|---|---|---|---|---|---|
| Negativa automática | p < t1 | 66 | 57.89 | 1 | 1.52 | 65 | 98.48 |
| Análise manual | t1 ≤ p < t2 | 3 | 2.63 | 0 | 0.0 | 3 | 100.0 |
| Positiva automática | p ≥ t2 | 45 | 39.47 | 41 | 91.11 | 4 | 8.89 |
| Total | t1 = 0.1, t2 = 0.21 | 114 | 100.0 | 42 | 36.84 | 72 | 63.16 |

## band_report — teste — erros

| metrica | valor | valor_pct | numerador | denominador | descricao_denominador |
|---|---|---|---|---|---|
| fn_auto | 1.0 |  | 1 |  | — (contagem) |
| fn_auto_sobre_positivos | 0.0238 | 2.38 | 1 | 42 | total de positivos |
| fn_auto_sobre_N | 0.0088 | 0.88 | 1 | 114 | N (total de instâncias) |
| fp_auto | 4.0 |  | 4 |  | — (contagem) |
| fp_auto_sobre_negativos | 0.0556 | 5.56 | 4 | 72 | total de negativos |
| fp_auto_sobre_N | 0.0351 | 3.51 | 4 | 114 | N (total de instâncias) |
| cobertura_automatica | 0.9737 | 97.37 | 111 | 114 | N (total de instâncias) |
| volume_manual | 0.0263 | 2.63 | 3 | 114 | N (total de instâncias) |

## Tempo de execução (s)

| etapa | segundos |
|---|---|
| carregar dados | 0.0049 |
| split | 0.0042 |
| tuning Regressão Logística | 0.0021 |
| tuning Random Forest | 0.0174 |
| tuning SVC (RBF) | 0.0013 |
| avaliação na validação | 0.0314 |
| avaliação no teste | 0.0075 |
| SHAP | 0.8167 |

## Figuras geradas

- 01_balanceamento_classes.png
- 02_matriz_confusao_validacao_regressao_logistica.png
- 02_matriz_confusao_validacao_random_forest.png
- 02_matriz_confusao_validacao_svc__rbf.png
- 03_curva_roc_validacao.png
- 04_curva_pr_validacao.png
- 05_matriz_confusao_teste.png
- 06_curva_roc_teste.png
- 07_curva_pr_teste.png
- 08_shap_beeswarm.png
- faixas/09_histograma_faixas_validacao.png
- faixas/10_histograma_faixas_teste.png
