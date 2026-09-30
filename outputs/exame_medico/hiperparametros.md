| modelo | metrica_cv | cv_roc_auc_media | cv_roc_auc_desvio | cv_folds | candidatos_testados | tempo_tuning_s | melhores_hiperparametros |
|---|---|---|---|---|---|---|---|
| Regressão Logística | roc_auc | 0.9941 | 0.0078 | 5 | 6 | 28.0622 | C=1 |
| Random Forest | roc_auc | 0.9874 | 0.01 | 5 | 48 | 17.8298 | max_depth=8; max_features=sqrt; min_samples_leaf=5; n_estimators=100 |
| SVC (RBF) | roc_auc | 0.9941 | 0.0083 | 5 | 16 | 0.2136 | C=10; gamma=0.01 |
