| modelo | metrica_cv | cv_average_precision_media | cv_average_precision_desvio | cv_folds | candidatos_testados | tempo_tuning_s | melhores_hiperparametros |
|---|---|---|---|---|---|---|---|
| Regressão Logística | average_precision | 0.7384 | 0.055 | 5 | 15 | 19.0334 | C=0.984674 |
| Random Forest | average_precision | 0.8297 | 0.0726 | 5 | 15 | 2936.3837 | n_estimators=100; min_samples_leaf=1; max_features=sqrt; max_depth=None |
| Gradient Boosting (HGB) | average_precision | 0.7486 | 0.0664 | 5 | 15 | 65.5414 | l2_regularization=0; learning_rate=0.0305156; max_iter=200; max_leaf_nodes=63; min_samples_leaf=100 |
