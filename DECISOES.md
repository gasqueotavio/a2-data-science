# Decisões de implementação

Decisões não cobertas explicitamente pela especificação, ou que exigiram escolher entre alternativas.

## Ambiente

1. **Primeira célula dos notebooks.** Mantém o bloco do Colab (mount do Drive, `PROJECT_DIR`, `%cd`, `pip install`) dentro de um `try/except ImportError`. No Colab o comportamento é idêntico ao especificado; fora do Colab usa a pasta local do projeto e não roda o `pip install`. Isso permitiu validar os notebooks localmente.
2. **Checkpoint e versão do scikit-learn.** O `.joblib` guarda a versão do scikit-learn usada no treino. Se a versão do ambiente for diferente (ex.: modelo treinado localmente e carregado no Colab), o modelo é retreinado em vez de carregado, para evitar incompatibilidade de pickle.

## Dados

3. **Duplicatas removidas antes do split**, em `prepare_data.py`: spam 403 linhas (de 5.574), fraude 1.081 (de 284.807), exame médico 0. Assim, uma mesma instância não aparece ao mesmo tempo no treino e no teste.
4. **Balanceamento** é calculado sobre a base completa (após remoção de duplicatas). A tabela `split` mostra o % de positivos em cada conjunto.
5. **Split 60/20/20**: o primeiro `train_test_split` separa 20% para teste; o segundo separa 25% dos 80% restantes para validação (= 20% do total).

## Modelos e tuning

6. **Fraude**: `RandomizedSearchCV(n_iter=15)` para os três modelos. Para a Regressão Logística o `C` é amostrado de `loguniform(1e-3, 1e2)` (com uma lista pequena, `n_iter=15` excederia o espaço). O `StandardScaler` age só em `Time` e `Amount` (`ColumnTransformer` com `remainder="passthrough"`).
7. **Spam**: `TfidfVectorizer(strip_accents="unicode")` como primeiro passo do pipeline; `ngram_range`, `min_df` e `sublinear_tf` entram na busca. O LinearSVC é envolvido em `CalibratedClassifierCV(cv=3, method="sigmoid")` para ter `predict_proba`.
8. **Desempate na seleção do melhor modelo**: maior métrica-alvo na validação; em caso de empate, maior AP, depois maior ROC-AUC, depois a ordem de declaração dos modelos. No exame médico, Regressão Logística e SVC empataram em ROC-AUC (0,9957) e AP (0,9946) na validação; venceu a Regressão Logística pela ordem de declaração.
9. **PR-AUC trapezoidal** = `sklearn.metrics.auc(recall, precision)` sobre os pontos de `precision_recall_curve`. **AP** = `average_precision_score` (soma em degraus, sem interpolação). As duas aparecem em colunas separadas.

## Pontos de corte (Parte 1)

10. **Grade de `choose_cutoffs`**: t1 e t2 variam de 0,00 a 1,00 com passo 0,01, sempre com t1 < t2. Desempate entre pares com o mesmo volume manual: menos erros automáticos (fn_auto + fp_auto), depois maior t1.
11. **Spam com os limites atuais (max_fn_rate = 0,10; max_fp_rate = 0,01)**: o par ótimo na validação foi t1 = 0,30 e t2 = 0,31, com **volume manual zero**. Os limites são folgados o suficiente para que um único corte satisfaça as duas restrições, então a faixa manual fica vazia. Isso não é erro do código; é consequência dos limites. Para ter uma faixa manual efetiva, a equipe deve apertar os limites em `src/config.py` (por exemplo, max_fn_rate = 0,03 e max_fp_rate = 0,002) e justificar a escolha.

## SHAP

12. **Conflito entre a regra 1 ("saída sempre em probabilidade") e o uso de `LinearExplainer`.** O `LinearExplainer` explica a saída linear do modelo (log-odds na Regressão Logística), não a probabilidade. Explicar a probabilidade exigiria um explainer por permutação, inviável no spam (34 mil features TF-IDF) e divergente da especificação, que pede `LinearExplainer` para modelos lineares. Decisão:
    - Todas as **métricas, curvas, histogramas e cortes** usam probabilidade (`predict_proba[:, 1]`), sem exceção.
    - No **SHAP**, modelos lineares são explicados em log-odds; a sigmoide é monotônica, então a ordem de importância e o sentido de cada contribuição (aumenta/diminui) são os mesmos da probabilidade. A saída explicada fica escrita no `shap_info.txt`, no título e no eixo do beeswarm.
    - Modelos de árvore (TreeExplainer) e SVC (permutação sobre `predict_proba[:, 1]`) são explicados diretamente em probabilidade. No HGB usa-se `model_output="probability"` com fundo de 100 linhas do treino.
13. **Naive Bayes Multinomial**: o log-odds posterior é exatamente linear nas features, então é explicado com `LinearExplainer((coef, intercepto))`, onde coef = `feature_log_prob_[1] - feature_log_prob_[0]`.
14. **LinearSVC calibrado**: explicado pelo escore linear médio dos LinearSVC internos do `CalibratedClassifierCV` (antes da calibração sigmoide, que é monotônica).
15. **Spam — tamanho da explicação**: das 34 mil features TF-IDF, só as 200 com maior média |SHAP| são mantidas no gráfico; a linha "Soma das outras N features" do beeswarm se refere às restantes entre essas 200.
16. **Coluna `correlacao_valor_shap`** no top 10: correlação de Pearson entre o valor da feature e o SHAP, para indicar se valores altos aumentam ou diminuem a saída explicada. Apoio à leitura do beeswarm, não substitui o gráfico.

## Execução local (Windows)

17. No Windows, o joblib imprime `KeyError ... joblib_memmapping_folder` ao encerrar os processos paralelos. É um ruído conhecido da limpeza de arquivos temporários do joblib/loky e não afeta os resultados. Não ocorre no Colab (Linux).
