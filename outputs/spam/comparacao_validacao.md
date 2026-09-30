| modelo | limiar | acuracia | precisao | recall | f1 | roc_auc | pr_auc_trapezoidal | average_precision_AP | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | 0.5 | 0.9816 | 0.9664 | 0.8846 | 0.9237 | 0.9956 | 0.9789 | 0.979 | 900 | 4 | 15 | 115 |
| Naive Bayes Multinomial | 0.5 | 0.9845 | 1.0 | 0.8769 | 0.9344 | 0.9887 | 0.9678 | 0.9678 | 904 | 0 | 16 | 114 |
| LinearSVC calibrado | 0.5 | 0.9826 | 0.9118 | 0.9538 | 0.9323 | 0.9956 | 0.9769 | 0.977 | 892 | 12 | 6 | 124 |
