"""SHAP beeswarm do melhor modelo de cada dominio.

Dados explicados: amostra de ate SHAP_MAX_SAMPLES linhas do conjunto de TESTE,
transformadas pelo pre-processamento do proprio pipeline. Fundo (background):
ate SHAP_BACKGROUND linhas do TREINO.

Explainer por tipo de modelo:
  - RandomForest / HistGradientBoosting -> TreeExplainer (saida: probabilidade)
  - LogisticRegression / MultinomialNB / LinearSVC calibrado -> LinearExplainer
    (saida: escala linear nativa do modelo; ver DECISOES.md)
  - demais (ex.: SVC) -> shap.Explainer (permutacao) sobre predict_proba[:, 1]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.config import RANDOM_STATE, SHAP_BACKGROUND, SHAP_MAX_SAMPLES
from src.evaluate import DPI, save_table

MAX_FEATURES_KEPT = 200  # no texto (dezenas de milhares de tokens) guarda so as mais relevantes


def _sample(X, n):
    n = min(n, len(X))
    return X.sample(n=n, random_state=RANDOM_STATE) if n < len(X) else X


def _dense(M):
    return M.toarray() if sparse.issparse(M) else np.asarray(M)


def _linear_params(clf):
    """(coef, intercepto, descricao da saida) para modelos lineares."""
    if isinstance(clf, LogisticRegression):
        return (clf.coef_.ravel(), float(clf.intercept_[0]),
                "log-odds (escala linear) da classe positiva da Regressão Logística")
    if isinstance(clf, MultinomialNB):
        # No NB multinomial o log-odds posterior e exatamente linear nas features.
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        return (coef, float(clf.class_log_prior_[1] - clf.class_log_prior_[0]),
                "log-odds (escala linear) da classe positiva do Naive Bayes Multinomial")
    if isinstance(clf, CalibratedClassifierCV):
        bases = [c.estimator for c in clf.calibrated_classifiers_]
        if all(hasattr(b, "coef_") for b in bases):
            coef = np.mean([b.coef_.ravel() for b in bases], axis=0)
            inter = float(np.mean([np.ravel(b.intercept_)[0] for b in bases]))
            return (coef, inter,
                    "escore linear médio dos LinearSVC internos (antes da calibração "
                    "sigmoide, que é monotônica)")
    return None


def _translate_beeswarm(fig, output_name):
    """Traduz os textos fixos (em ingles) do beeswarm do SHAP."""
    ax = fig.axes[0]
    ax.set_xlabel(f"Valor SHAP (impacto na saída explicada: {output_name})")
    labels = []
    for t in ax.get_yticklabels():
        txt = t.get_text()
        if txt.startswith("Sum of"):
            n = txt.split()[2]
            txt = f"Soma das outras {n} features"
        labels.append(txt)
    ax.set_yticks(ax.get_yticks(), labels)
    for cax in fig.axes[1:]:
        cax.set_ylabel("Valor da feature")
        cax.set_yticklabels(["Baixo" if l.get_text() == "Low" else
                             "Alto" if l.get_text() == "High" else l.get_text()
                             for l in cax.get_yticklabels()])


def explain_model(domain, model_name, pipe, X_train, X_test, out_dir, fig_path):
    import shap

    prep, clf = pipe[:-1], pipe[-1]
    X_exp = _sample(X_test, SHAP_MAX_SAMPLES)
    X_bg = _sample(X_train, SHAP_BACKGROUND)
    Xe, Xb = prep.transform(X_exp), prep.transform(X_bg)
    names = np.array([str(n) for n in prep.get_feature_names_out()])

    linear = _linear_params(clf)
    if isinstance(clf, RandomForestClassifier):
        explainer_name = "TreeExplainer"
        output = "probabilidade da classe positiva (predict_proba[:, 1])"
        sv = shap.TreeExplainer(clf)(_dense(Xe), check_additivity=False)
    elif isinstance(clf, HistGradientBoostingClassifier):
        explainer_name = "TreeExplainer (interventional, model_output='probability')"
        output = "probabilidade da classe positiva (predict_proba[:, 1])"
        sv = shap.TreeExplainer(clf, data=_dense(Xb), model_output="probability",
                                feature_perturbation="interventional")(_dense(Xe))
    elif linear is not None:
        coef, intercept, output = linear
        explainer_name = "LinearExplainer (interventional)"
        sv = shap.LinearExplainer((coef, intercept), Xb)(Xe)
    else:
        explainer_name = "shap.Explainer (permutação) sobre predict_proba[:, 1]"
        output = "probabilidade da classe positiva (predict_proba[:, 1])"
        Xb_d = _dense(Xb)
        masker = shap.maskers.Independent(Xb_d, max_samples=len(Xb_d))
        sv = shap.Explainer(lambda d: clf.predict_proba(d)[:, 1], masker,
                            algorithm="permutation", seed=RANDOM_STATE)(_dense(Xe))

    values = np.asarray(sv.values)
    if values.ndim == 3:            # (amostras, features, classes) -> classe positiva
        values = values[:, :, 1]
    base = np.asarray(sv.base_values)
    if base.ndim == 2:
        base = base[:, 1]
    data = _dense(Xe)

    if values.shape[1] > MAX_FEATURES_KEPT:  # texto: mantem as colunas mais relevantes
        keep = np.argsort(np.abs(values).mean(axis=0))[::-1][:MAX_FEATURES_KEPT]
        values, data, names = values[:, keep], data[:, keep], names[keep]
    explanation = shap.Explanation(values=values, base_values=base, data=data,
                                   feature_names=list(names))

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    _translate_beeswarm(fig, output.split(" (")[0])
    plt.title(f"SHAP beeswarm — {domain} — {model_name}\n"
              f"Saída explicada: {output.split(' (')[0]}; "
              f"{len(X_exp)} instâncias do teste", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1][:10]
    rows = []
    for j in order:
        x, s = data[:, j], values[:, j]
        corr = np.corrcoef(x, s)[0, 1] if x.std() > 0 and s.std() > 0 else np.nan
        rows.append({
            "posicao": len(rows) + 1, "feature": names[j], "media_abs_shap": importance[j],
            "correlacao_valor_shap": corr,
            "leitura": ("valores altos aumentam a saída" if corr > 0 else
                        "valores altos diminuem a saída" if corr < 0 else "indefinido"),
        })
    top10 = pd.DataFrame(rows)
    save_table(top10, out_dir / "shap_top10")

    info = [
        f"Domínio: {domain}",
        f"Modelo explicado: {model_name} ({type(clf).__name__})",
        f"Explainer: {explainer_name}",
        f"Saída do modelo explicada: {output}",
        f"Dados explicados: {len(X_exp)} instâncias aleatórias do conjunto de TESTE "
        f"(random_state={RANDOM_STATE}), transformadas pelo pré-processamento do pipeline",
        f"Conjunto de fundo (background): {len(X_bg)} instâncias aleatórias do TREINO",
        f"Número de features após o pré-processamento: {len(prep.get_feature_names_out())}",
        "Beeswarm: max_display=15; cor = valor da feature (vermelho alto, azul baixo); "
        "eixo X = contribuição SHAP para a saída explicada",
    ]
    if values.shape[1] < len(prep.get_feature_names_out()):
        info.append(f"Somente as {values.shape[1]} features de maior média |SHAP| foram "
                    "mantidas para o gráfico (vocabulário TF-IDF muito grande).")
    if domain == "fraude":
        info.append("Limitação: V1–V28 são componentes PCA anônimos fornecidos pela base; "
                    "não têm significado de negócio interpretável. Apenas Time e Amount "
                    "são variáveis originais.")
    (out_dir / "shap_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    return {"info": info, "top10": top10, "explainer": explainer_name, "output": output}
