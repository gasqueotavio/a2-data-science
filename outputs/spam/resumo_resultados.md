# Resumo de resultados — spam

## Balanceamento

| classe | n | pct_do_total |
|---|---|---|
| 1 = spam (é spam?) | 653 | 12.63 |
| 0 = ham (legítima) | 4518 | 87.37 |
| Total | 5171 | 100.0 |

## Split 60/20/20

| conjunto | n | pct_do_total | n_positivos | pct_positivos | n_negativos |
|---|---|---|---|---|---|
| treino | 3102 | 59.99 | 392 | 12.64 | 2710 |
| validacao | 1034 | 20.0 | 130 | 12.57 | 904 |
| teste | 1035 | 20.02 | 131 | 12.66 | 904 |

## Hiperparâmetros vencedores (CV 5 folds, average_precision)

| modelo | metrica_cv | cv_average_precision_media | cv_average_precision_desvio | cv_folds | candidatos_testados | tempo_tuning_s | melhores_hiperparametros |
|---|---|---|---|---|---|---|---|
| Regressão Logística | average_precision | 0.9736 | 0.0119 | 5 | 48 | 33.6023 | C=10; tfidf__min_df=1; tfidf__ngram_range=(1, 2); tfidf__sublinear_tf=True |
| Naive Bayes Multinomial | average_precision | 0.9715 | 0.013 | 5 | 60 | 6.9058 | alpha=0.1; tfidf__min_df=2; tfidf__ngram_range=(1, 2); tfidf__sublinear_tf=True |
| LinearSVC calibrado | average_precision | 0.9744 | 0.0119 | 5 | 48 | 5.0927 | estimator__C=1; tfidf__min_df=1; tfidf__ngram_range=(1, 2); tfidf__sublinear_tf=True |

## Comparação na validação (limiar = 0.5)

| modelo | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | 0.5 | 0.9816 | 0.9664 | 0.8846 | 0.9237 | 0.9956 | 0.9789 | 0.979 | 900 | 4 | 15 | 115 |
| Naive Bayes Multinomial | 0.5 | 0.9845 | 1.0 | 0.8769 | 0.9344 | 0.9887 | 0.9678 | 0.9678 | 904 | 0 | 16 | 114 |
| LinearSVC calibrado | 0.5 | 0.9826 | 0.9118 | 0.9538 | 0.9323 | 0.9956 | 0.9769 | 0.977 | 892 | 12 | 6 | 124 |

## Melhor modelo

Regressão Logística — maior average_precision_AP na validação = 0.9790

## Resultado no teste (limiar = 0.5)

| modelo | conjunto | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | teste | 0.5 | 0.9807 | 0.9912 | 0.855 | 0.918 | 0.9903 | 0.9664 | 0.9665 | 903 | 1 | 19 | 112 |

## SHAP — informações

- Domínio: spam
- Modelo explicado: Regressão Logística (LogisticRegression)
- Explainer: LinearExplainer (interventional)
- Saída do modelo explicada: log-odds (escala linear) da classe positiva da Regressão Logística
- Dados explicados: 1000 instâncias aleatórias do conjunto de TESTE (random_state=42), transformadas pelo pré-processamento do pipeline
- Conjunto de fundo (background): 100 instâncias aleatórias do TREINO
- Número de features após o pré-processamento: 34311
- Beeswarm: max_display=15; cor = valor da feature (vermelho alto, azul baixo); eixo X = contribuição SHAP para a saída explicada
- Somente as 200 features de maior média |SHAP| foram mantidas para o gráfico (vocabulário TF-IDF muito grande).

## SHAP — top 10 features (média |SHAP|)

| posicao | feature | media_abs_shap | correlacao_valor_shap | leitura |
|---|---|---|---|---|
| 1 | to | 0.1831 | 1.0 | valores altos aumentam a saída |
| 2 | call | 0.1637 | 1.0 | valores altos aumentam a saída |
| 3 | my | 0.0988 | -1.0 | valores altos diminuem a saída |
| 4 | me | 0.0939 | -1.0 | valores altos diminuem a saída |
| 5 | your | 0.0894 | 1.0 | valores altos aumentam a saída |
| 6 | for | 0.0763 | 1.0 | valores altos aumentam a saída |
| 7 | or | 0.0737 | 1.0 | valores altos aumentam a saída |
| 8 | that | 0.0688 | -1.0 | valores altos diminuem a saída |
| 9 | free | 0.0657 | 1.0 | valores altos aumentam a saída |
| 10 | it | 0.0647 | -1.0 | valores altos diminuem a saída |

## Cortes (escolhidos na validação)

- t1 = 0.11
- t2 = 0.70
- max_fn_rate = 0.03 (sobre positivos)
- max_fp_rate = 0.002 (sobre negativos)

## Trade-off (10 melhores pares viáveis, validação)

| t1 | t2 | volume_manual | cobertura_automatica | fn_auto | fn_auto_sobre_positivos | fp_auto | fp_auto_sobre_negativos | violacao | viavel |
|---|---|---|---|---|---|---|---|---|---|
| 0.11 | 0.7 | 0.0435 | 0.9565 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.11 | 0.71 | 0.0445 | 0.9555 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.1 | 0.7 | 0.0455 | 0.9545 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.11 | 0.72 | 0.0464 | 0.9536 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.1 | 0.71 | 0.0464 | 0.9536 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.11 | 0.73 | 0.0474 | 0.9526 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.11 | 0.74 | 0.0474 | 0.9526 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.11 | 0.75 | 0.0484 | 0.9516 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.1 | 0.72 | 0.0484 | 0.9516 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |
| 0.1 | 0.73 | 0.0493 | 0.9507 | 3 | 0.0231 | 1 | 0.0011 | 0.0 | True |

## band_report — validação — faixas

| faixa | regra | n | pct_populacao_sobre_N | n_positivos | pct_positivos_sobre_n_faixa | n_negativos | pct_negativos_sobre_n_faixa |
|---|---|---|---|---|---|---|---|
| Negativa automática | p < t1 | 881 | 85.2 | 3 | 0.34 | 878 | 99.66 |
| Análise manual | t1 ≤ p < t2 | 45 | 4.35 | 20 | 44.44 | 25 | 55.56 |
| Positiva automática | p ≥ t2 | 108 | 10.44 | 107 | 99.07 | 1 | 0.93 |
| Total | t1 = 0.11, t2 = 0.7 | 1034 | 100.0 | 130 | 12.57 | 904 | 87.43 |

## band_report — validação — erros

| metrica | valor | valor_pct | numerador | denominador | descricao_denominador |
|---|---|---|---|---|---|
| fn_auto | 3.0 |  | 3 |  | — (contagem) |
| fn_auto_sobre_positivos | 0.0231 | 2.31 | 3 | 130 | total de positivos |
| fn_auto_sobre_N | 0.0029 | 0.29 | 3 | 1034 | N (total de instâncias) |
| fp_auto | 1.0 |  | 1 |  | — (contagem) |
| fp_auto_sobre_negativos | 0.0011 | 0.11 | 1 | 904 | total de negativos |
| fp_auto_sobre_N | 0.001 | 0.1 | 1 | 1034 | N (total de instâncias) |
| cobertura_automatica | 0.9565 | 95.65 | 989 | 1034 | N (total de instâncias) |
| volume_manual | 0.0435 | 4.35 | 45 | 1034 | N (total de instâncias) |

## band_report — teste — faixas

| faixa | regra | n | pct_populacao_sobre_N | n_positivos | pct_positivos_sobre_n_faixa | n_negativos | pct_negativos_sobre_n_faixa |
|---|---|---|---|---|---|---|---|
| Negativa automática | p < t1 | 881 | 85.12 | 8 | 0.91 | 873 | 99.09 |
| Análise manual | t1 ≤ p < t2 | 52 | 5.02 | 21 | 40.38 | 31 | 59.62 |
| Positiva automática | p ≥ t2 | 102 | 9.86 | 102 | 100.0 | 0 | 0.0 |
| Total | t1 = 0.11, t2 = 0.7 | 1035 | 100.0 | 131 | 12.66 | 904 | 87.34 |

## band_report — teste — erros

| metrica | valor | valor_pct | numerador | denominador | descricao_denominador |
|---|---|---|---|---|---|
| fn_auto | 8.0 |  | 8 |  | — (contagem) |
| fn_auto_sobre_positivos | 0.0611 | 6.11 | 8 | 131 | total de positivos |
| fn_auto_sobre_N | 0.0077 | 0.77 | 8 | 1035 | N (total de instâncias) |
| fp_auto | 0.0 |  | 0 |  | — (contagem) |
| fp_auto_sobre_negativos | 0.0 | 0.0 | 0 | 904 | total de negativos |
| fp_auto_sobre_N | 0.0 | 0.0 | 0 | 1035 | N (total de instâncias) |
| cobertura_automatica | 0.9498 | 94.98 | 983 | 1035 | N (total de instâncias) |
| volume_manual | 0.0502 | 5.02 | 52 | 1035 | N (total de instâncias) |

## Tempo de execução (s)

| etapa | segundos |
|---|---|
| carregar dados | 0.0093 |
| split | 0.0047 |
| tuning Regressão Logística | 0.0368 |
| tuning Naive Bayes Multinomial | 0.0086 |
| tuning LinearSVC calibrado | 0.0372 |
| avaliação na validação | 0.0677 |
| avaliação no teste | 0.0196 |
| SHAP | 1.1534 |

## Figuras geradas

- 01_balanceamento_classes.png
- 02_matriz_confusao_validacao_regressao_logistica.png
- 02_matriz_confusao_validacao_naive_bayes_multinomial.png
- 02_matriz_confusao_validacao_linearsvc_calibrado.png
- 03_curva_roc_validacao.png
- 04_curva_pr_validacao.png
- 05_matriz_confusao_teste.png
- 06_curva_roc_teste.png
- 07_curva_pr_teste.png
- 08_shap_beeswarm.png
- faixas/09_histograma_faixas_validacao.png
- faixas/10_histograma_faixas_teste.png
