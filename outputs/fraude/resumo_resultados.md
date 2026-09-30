# Resumo de resultados — fraude

## Balanceamento

| classe | n | pct_do_total |
|---|---|---|
| 1 = fraude (é fraude?) | 473 | 0.17 |
| 0 = legítima | 283253 | 99.83 |
| Total | 283726 | 100.0 |

## Split 60/20/20

| conjunto | n | pct_do_total | n_positivos | pct_positivos | n_negativos |
|---|---|---|---|---|---|
| treino | 170235 | 60.0 | 284 | 0.17 | 169951 |
| validacao | 56745 | 20.0 | 94 | 0.17 | 56651 |
| teste | 56746 | 20.0 | 95 | 0.17 | 56651 |

## Hiperparâmetros vencedores (CV 5 folds, average_precision)

| modelo | metrica_cv | cv_average_precision_media | cv_average_precision_desvio | cv_folds | candidatos_testados | tempo_tuning_s | melhores_hiperparametros |
|---|---|---|---|---|---|---|---|
| Regressão Logística | average_precision | 0.7384 | 0.055 | 5 | 15 | 19.0334 | C=0.984674 |
| Random Forest | average_precision | 0.8297 | 0.0726 | 5 | 15 | 2936.3837 | n_estimators=100; min_samples_leaf=1; max_features=sqrt; max_depth=None |
| Gradient Boosting (HGB) | average_precision | 0.7486 | 0.0664 | 5 | 15 | 65.5414 | l2_regularization=0; learning_rate=0.0305156; max_iter=200; max_leaf_nodes=63; min_samples_leaf=100 |

## Comparação na validação (limiar = 0.5)

| modelo | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | 0.5 | 0.9735 | 0.0539 | 0.9043 | 0.1017 | 0.9789 | 0.8106 | 0.7868 | 55159 | 1492 | 9 | 85 |
| Random Forest | 0.5 | 0.9996 | 0.9487 | 0.7872 | 0.8605 | 0.9512 | 0.8789 | 0.8735 | 56647 | 4 | 20 | 74 |
| Gradient Boosting (HGB) | 0.5 | 0.9978 | 0.4154 | 0.8617 | 0.5606 | 0.9155 | 0.8258 | 0.825 | 56537 | 114 | 13 | 81 |

## Melhor modelo

Random Forest — maior average_precision_AP na validação = 0.8735

## Resultado no teste (limiar = 0.5)

| modelo | conjunto | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | teste | 0.5 | 0.9995 | 0.9571 | 0.7053 | 0.8121 | 0.9245 | 0.8113 | 0.8029 | 56648 | 3 | 28 | 67 |

## SHAP — informações

- Domínio: fraude
- Modelo explicado: Random Forest (RandomForestClassifier)
- Explainer: TreeExplainer
- Saída do modelo explicada: probabilidade da classe positiva (predict_proba[:, 1])
- Dados explicados: 1000 instâncias aleatórias do conjunto de TESTE (random_state=42), transformadas pelo pré-processamento do pipeline
- Conjunto de fundo (background): 100 instâncias aleatórias do TREINO
- Número de features após o pré-processamento: 30
- Beeswarm: max_display=15; cor = valor da feature (vermelho alto, azul baixo); eixo X = contribuição SHAP para a saída explicada
- Limitação: V1–V28 são componentes PCA anônimos fornecidos pela base; não têm significado de negócio interpretável. Apenas Time e Amount são variáveis originais.

## SHAP — top 10 features (média |SHAP|)

| posicao | feature | media_abs_shap | correlacao_valor_shap | leitura |
|---|---|---|---|---|
| 1 | V14 | 0.0818 | -0.901 | valores altos diminuem a saída |
| 2 | V12 | 0.0712 | -0.915 | valores altos diminuem a saída |
| 3 | V4 | 0.055 | 0.9284 | valores altos aumentam a saída |
| 4 | V3 | 0.049 | -0.9025 | valores altos diminuem a saída |
| 5 | V11 | 0.047 | 0.929 | valores altos aumentam a saída |
| 6 | V10 | 0.0408 | -0.79 | valores altos diminuem a saída |
| 7 | V17 | 0.0218 | -0.3187 | valores altos diminuem a saída |
| 8 | V16 | 0.0148 | -0.8619 | valores altos diminuem a saída |
| 9 | V1 | 0.013 | -0.6065 | valores altos diminuem a saída |
| 10 | V7 | 0.0117 | -0.8472 | valores altos diminuem a saída |

## Cortes (escolhidos na validação)

- t1 = 0.02
- t2 = 0.16
- max_fn_rate = 0.1 (sobre positivos)
- max_fp_rate = 0.0002 (sobre negativos)

## Trade-off (10 melhores pares viáveis, validação)

| t1 | t2 | volume_manual | cobertura_automatica | fn_auto | fn_auto_sobre_positivos | fp_auto | fp_auto_sobre_negativos | violacao | viavel |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.16 | 0.0045 | 0.9955 | 9 | 0.0957 | 10 | 0.0002 | 0.0 | True |
| 0.02 | 0.17 | 0.0046 | 0.9954 | 9 | 0.0957 | 8 | 0.0001 | 0.0 | True |
| 0.02 | 0.18 | 0.0046 | 0.9954 | 9 | 0.0957 | 8 | 0.0001 | 0.0 | True |
| 0.02 | 0.19 | 0.0046 | 0.9954 | 9 | 0.0957 | 8 | 0.0001 | 0.0 | True |
| 0.02 | 0.2 | 0.0046 | 0.9954 | 9 | 0.0957 | 8 | 0.0001 | 0.0 | True |
| 0.02 | 0.21 | 0.0046 | 0.9954 | 9 | 0.0957 | 7 | 0.0001 | 0.0 | True |
| 0.02 | 0.22 | 0.0046 | 0.9954 | 9 | 0.0957 | 6 | 0.0001 | 0.0 | True |
| 0.02 | 0.23 | 0.0046 | 0.9954 | 9 | 0.0957 | 6 | 0.0001 | 0.0 | True |
| 0.02 | 0.24 | 0.0046 | 0.9954 | 9 | 0.0957 | 6 | 0.0001 | 0.0 | True |
| 0.02 | 0.25 | 0.0046 | 0.9954 | 9 | 0.0957 | 6 | 0.0001 | 0.0 | True |

## band_report — validação — faixas

| faixa | regra | n | pct_populacao_sobre_N | n_positivos | pct_positivos_sobre_n_faixa | n_negativos | pct_negativos_sobre_n_faixa |
|---|---|---|---|---|---|---|---|
| Negativa automática | p < t1 | 56397 | 99.39 | 9 | 0.02 | 56388 | 99.98 |
| Análise manual | t1 ≤ p < t2 | 256 | 0.45 | 3 | 1.17 | 253 | 98.83 |
| Positiva automática | p ≥ t2 | 92 | 0.16 | 82 | 89.13 | 10 | 10.87 |
| Total | t1 = 0.02, t2 = 0.16 | 56745 | 100.0 | 94 | 0.17 | 56651 | 99.83 |

## band_report — validação — erros

| metrica | valor | valor_pct | numerador | denominador | descricao_denominador |
|---|---|---|---|---|---|
| fn_auto | 9.0 |  | 9 |  | — (contagem) |
| fn_auto_sobre_positivos | 0.0957 | 9.57 | 9 | 94 | total de positivos |
| fn_auto_sobre_N | 0.0002 | 0.02 | 9 | 56745 | N (total de instâncias) |
| fp_auto | 10.0 |  | 10 |  | — (contagem) |
| fp_auto_sobre_negativos | 0.0002 | 0.02 | 10 | 56651 | total de negativos |
| fp_auto_sobre_N | 0.0002 | 0.02 | 10 | 56745 | N (total de instâncias) |
| cobertura_automatica | 0.9955 | 99.55 | 56489 | 56745 | N (total de instâncias) |
| volume_manual | 0.0045 | 0.45 | 256 | 56745 | N (total de instâncias) |

## band_report — teste — faixas

| faixa | regra | n | pct_populacao_sobre_N | n_positivos | pct_positivos_sobre_n_faixa | n_negativos | pct_negativos_sobre_n_faixa |
|---|---|---|---|---|---|---|---|
| Negativa automática | p < t1 | 56416 | 99.42 | 16 | 0.03 | 56400 | 99.97 |
| Análise manual | t1 ≤ p < t2 | 247 | 0.44 | 4 | 1.62 | 243 | 98.38 |
| Positiva automática | p ≥ t2 | 83 | 0.15 | 75 | 90.36 | 8 | 9.64 |
| Total | t1 = 0.02, t2 = 0.16 | 56746 | 100.0 | 95 | 0.17 | 56651 | 99.83 |

## band_report — teste — erros

| metrica | valor | valor_pct | numerador | denominador | descricao_denominador |
|---|---|---|---|---|---|
| fn_auto | 16.0 |  | 16 |  | — (contagem) |
| fn_auto_sobre_positivos | 0.1684 | 16.84 | 16 | 95 | total de positivos |
| fn_auto_sobre_N | 0.0003 | 0.03 | 16 | 56746 | N (total de instâncias) |
| fp_auto | 8.0 |  | 8 |  | — (contagem) |
| fp_auto_sobre_negativos | 0.0001 | 0.01 | 8 | 56651 | total de negativos |
| fp_auto_sobre_N | 0.0001 | 0.01 | 8 | 56746 | N (total de instâncias) |
| cobertura_automatica | 0.9956 | 99.56 | 56499 | 56746 | N (total de instâncias) |
| volume_manual | 0.0044 | 0.44 | 247 | 56746 | N (total de instâncias) |

## Tempo de execução (s)

| etapa | segundos |
|---|---|
| carregar dados | 0.8602 |
| split | 0.1575 |
| tuning Regressão Logística | 0.0013 |
| tuning Random Forest | 0.013 |
| tuning Gradient Boosting (HGB) | 0.0062 |
| avaliação na validação | 0.4281 |
| avaliação no teste | 0.1731 |
| SHAP | 6.9804 |

## Figuras geradas

- 01_balanceamento_classes.png
- 02_matriz_confusao_validacao_regressao_logistica.png
- 02_matriz_confusao_validacao_random_forest.png
- 02_matriz_confusao_validacao_gradient_boosting__hgb.png
- 03_curva_roc_validacao.png
- 04_curva_pr_validacao.png
- 05_matriz_confusao_teste.png
- 06_curva_roc_teste.png
- 07_curva_pr_teste.png
- 08_shap_beeswarm.png
- faixas/09_histograma_faixas_validacao.png
- faixas/10_histograma_faixas_teste.png
