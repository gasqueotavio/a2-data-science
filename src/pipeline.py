"""Carga, split 60/20/20, pipelines por dominio, tuning por validacao cruzada e checkpoint.

Todo o pre-processamento fica DENTRO do Pipeline, entao e ajustado apenas no
treino (e dentro de cada fold da validacao cruzada). O teste nunca entra aqui.
"""

import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.config import CV_SPLITS, DOMAINS, RANDOM_STATE, TEST_SIZE, VAL_SIZE, output_dir


# ---------------------------------------------------------------------------
# Tempo de execucao
# ---------------------------------------------------------------------------
TIMINGS = {}


@contextmanager
def timed(label):
    """Imprime e registra (em TIMINGS) o tempo de execucao de uma etapa."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    TIMINGS[label] = elapsed
    print(f"[tempo] {label}: {elapsed:.1f} s")


# ---------------------------------------------------------------------------
# Dados e split
# ---------------------------------------------------------------------------
def load_domain_data(domain):
    """Retorna (X, y). No spam, X e a Series de texto bruto."""
    cfg = DOMAINS[domain]
    if not cfg["csv"].exists():
        raise FileNotFoundError(f"{cfg['csv']} não existe. Rode antes: python -m src.prepare_data")
    df = pd.read_csv(cfg["csv"])
    y = df.pop("target").astype(int).to_numpy()
    X = df["text"].fillna("") if domain == "spam" else df
    return X, y


def split_data(X, y):
    """Split estratificado 60/20/20 com dois train_test_split(stratify=y)."""
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=VAL_SIZE, stratify=y_dev, random_state=RANDOM_STATE)
    return {"treino": (X_train, y_train), "validacao": (X_val, y_val), "teste": (X_test, y_test)}


def split_table(splits):
    n_total = sum(len(y) for _, y in splits.values())
    rows = []
    for name, (_, y) in splits.items():
        rows.append({"conjunto": name, "n": len(y), "pct_do_total": 100 * len(y) / n_total,
                     "n_positivos": int(y.sum()), "pct_positivos": 100 * y.mean(),
                     "n_negativos": int(len(y) - y.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modelos e espacos de busca
# ---------------------------------------------------------------------------
def build_models(domain):
    """dict nome -> (Pipeline, espaco de hiperparametros)."""
    rs = RANDOM_STATE
    if domain == "spam":
        tfidf = {"tfidf__ngram_range": [(1, 1), (1, 2)],
                 "tfidf__min_df": [1, 2, 3],
                 "tfidf__sublinear_tf": [True, False]}

        def text_pipe(clf):
            return Pipeline([("tfidf", TfidfVectorizer(strip_accents="unicode")), ("clf", clf)])

        return {
            "Regressão Logística": (
                text_pipe(LogisticRegression(max_iter=3000, random_state=rs)),
                {**tfidf, "clf__C": [0.1, 1, 10, 100]}),
            "Naive Bayes Multinomial": (
                text_pipe(MultinomialNB()),
                {**tfidf, "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}),
            "LinearSVC calibrado": (
                text_pipe(CalibratedClassifierCV(LinearSVC(random_state=rs), cv=3,
                                                 method="sigmoid")),
                {**tfidf, "clf__estimator__C": [0.01, 0.1, 1, 10]}),
        }

    if domain == "fraude":
        def fraud_pipe(clf):
            prep = ColumnTransformer([("escala", StandardScaler(), ["Time", "Amount"])],
                                     remainder="passthrough", verbose_feature_names_out=False)
            return Pipeline([("prep", prep), ("clf", clf)])

        return {
            "Regressão Logística": (
                fraud_pipe(LogisticRegression(class_weight="balanced", max_iter=3000,
                                              random_state=rs)),
                {"clf__C": loguniform(1e-3, 1e2)}),
            "Random Forest": (
                fraud_pipe(RandomForestClassifier(class_weight="balanced", random_state=rs)),
                {"clf__n_estimators": [100, 200, 300],
                 "clf__max_depth": [6, 10, 14, None],
                 "clf__min_samples_leaf": [1, 3, 10],
                 "clf__max_features": ["sqrt", 0.3]}),
            "Gradient Boosting (HGB)": (
                fraud_pipe(HistGradientBoostingClassifier(class_weight="balanced",
                                                          random_state=rs)),
                {"clf__learning_rate": loguniform(0.02, 0.3),
                 "clf__max_leaf_nodes": [15, 31, 63],
                 "clf__min_samples_leaf": [20, 50, 100],
                 "clf__l2_regularization": [0.0, 0.1, 1.0],
                 "clf__max_iter": [200, 400]}),
        }

    if domain == "exame_medico":
        def med_pipe(clf):
            return Pipeline([("escala", StandardScaler()), ("clf", clf)])

        return {
            "Regressão Logística": (
                med_pipe(LogisticRegression(max_iter=5000, random_state=rs)),
                {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}),
            "Random Forest": (
                med_pipe(RandomForestClassifier(random_state=rs)),
                {"clf__n_estimators": [100, 300],
                 "clf__max_depth": [3, 5, 8, None],
                 "clf__min_samples_leaf": [1, 3, 5],
                 "clf__max_features": ["sqrt", "log2"]}),
            "SVC (RBF)": (
                med_pipe(SVC(kernel="rbf", probability=True, random_state=rs)),
                {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]}),
        }

    raise ValueError(f"Domínio desconhecido: {domain}")


def slug(name):
    table = str.maketrans("ãáâçéêíóôõú", "aaaceeiooou")
    return "".join(c if c.isalnum() else "_" for c in name.lower().translate(table)).strip("_")


# ---------------------------------------------------------------------------
# Tuning com checkpoint
# ---------------------------------------------------------------------------
def tune_model(domain, name, pipe, space, X_train, y_train, force_retrain=False):
    """Busca de hiperparametros por CV estratificada SOMENTE no treino.

    Salva/carrega checkpoint em outputs/<dominio>/models/<modelo>.joblib.
    """
    cfg = DOMAINS[domain]
    path = output_dir(domain) / "models" / f"{slug(name)}.joblib"
    if path.exists() and not force_retrain:
        ckpt = joblib.load(path)
        if ckpt.get("sklearn_version") == sklearn.__version__:
            print(f"[checkpoint] {name}: carregado de {path.name}")
            return ckpt
        print(f"[checkpoint] {name}: versão do scikit-learn diferente "
              f"({ckpt.get('sklearn_version')} != {sklearn.__version__}); retreinando.")

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    common = dict(scoring=cfg["target_metric"], cv=cv, n_jobs=-1, refit=True)
    if cfg["search"] == "random":
        search = RandomizedSearchCV(pipe, space, n_iter=cfg["n_iter"],
                                    random_state=RANDOM_STATE, **common)
    else:
        search = GridSearchCV(pipe, space, **common)

    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    i = search.best_index_
    ckpt = {
        "modelo": name,
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_mean": float(search.cv_results_["mean_test_score"][i]),
        "cv_std": float(search.cv_results_["std_test_score"][i]),
        "n_candidatos": len(search.cv_results_["params"]),
        "tuning_seconds": elapsed,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(ckpt, path)
    print(f"[tuning] {name}: {cfg['target_metric']} CV = {ckpt['cv_mean']:.4f} "
          f"± {ckpt['cv_std']:.4f} | {elapsed:.1f} s | salvo em {path.name}")
    return ckpt


def tune_all(domain, X_train, y_train, force_retrain=False):
    tuned = {}
    for name, (pipe, space) in build_models(domain).items():
        with timed(f"tuning {name}"):
            tuned[name] = tune_model(domain, name, pipe, space, X_train, y_train, force_retrain)
    return tuned


def _fmt_param(v):
    return f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)


def hyperparams_table(tuned, metric):
    rows = []
    for name, ck in tuned.items():
        rows.append({
            "modelo": name,
            "metrica_cv": metric,
            f"cv_{metric}_media": ck["cv_mean"],
            f"cv_{metric}_desvio": ck["cv_std"],
            "cv_folds": CV_SPLITS,
            "candidatos_testados": ck["n_candidatos"],
            "tempo_tuning_s": ck["tuning_seconds"],
            "melhores_hiperparametros": "; ".join(
                f"{k.replace('clf__', '')}={_fmt_param(v)}" for k, v in ck["best_params"].items()),
        })
    return pd.DataFrame(rows)


def predict_positive(estimator, X):
    """Probabilidade da classe positiva (coluna 1 de predict_proba), para TODAS as instancias."""
    classes = list(estimator.classes_)
    return estimator.predict_proba(X)[:, classes.index(1)]
