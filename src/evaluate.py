"""Metricas, curvas, matrizes de confusao, tabelas (CSV + Markdown) e resumo numerico.

Todas as analises usam a probabilidade da classe positiva (predict_proba[:, 1]).
Metricas que dependem de decisao usam o limiar THRESHOLD (0,5), informado em
toda tabela e titulo de figura correspondente.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src.config import METRIC_COLUMN, THRESHOLD

DPI = 200
COLORS = ["#1f4e9c", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
STYLES = ["-", "--", ":", "-.", "-"]


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def _is_pct(col):
    return col.startswith("pct_") or col.endswith("_pct") or col.startswith("percentual")


def round_table(df):
    """Percentuais com 2 casas, demais numeros com 4."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(2 if _is_pct(c) else 4)
    return out


def to_markdown(df):
    df = round_table(df)

    def cell(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return ""
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_table(df, path_stem):
    """Salva <path_stem>.csv e <path_stem>.md (valores arredondados)."""
    path_stem = Path(path_stem)
    round_table(df).to_csv(path_stem.with_suffix(".csv"), index=False)
    path_stem.with_suffix(".md").write_text(to_markdown(df) + "\n", encoding="utf-8")
    return path_stem.with_suffix(".csv")


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------------------
# Balanceamento
# ---------------------------------------------------------------------------
def class_balance_table(y, cfg):
    y = np.asarray(y)
    n = len(y)
    n_pos = int(y.sum())
    return pd.DataFrame({
        "classe": [f"1 = {cfg['positive_label']} ({cfg['positive_question']})",
                   f"0 = {cfg['negative_label']}", "Total"],
        "n": [n_pos, n - n_pos, n],
        "pct_do_total": [100 * n_pos / n, 100 * (n - n_pos) / n, 100.0],
    })


def plot_class_balance(y, cfg, path, title):
    tab = class_balance_table(y, cfg).iloc[:2]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([cfg["positive_label"], cfg["negative_label"]], tab["n"],
                  color=[COLORS[1], COLORS[0]])
    for b, n, p in zip(bars, tab["n"], tab["pct_do_total"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n} ({p:.2f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set(title=title, xlabel="Classe", ylabel="Quantidade de instâncias")
    ax.set_ylim(0, tab["n"].max() * 1.15)
    return _save_fig(fig, path)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def compute_metrics(y, p, threshold=THRESHOLD):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "limiar": threshold,
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_trapezoidal": auc(recall, precision),
        "average_precision_AP": average_precision_score(y, p),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def comparison_table(y, probas, threshold=THRESHOLD):
    """Uma linha por modelo. `probas` = dict nome -> probabilidade da classe positiva."""
    rows = [{"modelo": name, **compute_metrics(y, p, threshold)} for name, p in probas.items()]
    return pd.DataFrame(rows)


def select_best(comparison, target_metric):
    """Maior metrica-alvo na validacao. Empate: maior AP, depois maior ROC-AUC,
    depois a ordem de declaracao dos modelos (ordenacao estavel)."""
    col = METRIC_COLUMN[target_metric]
    keys = list(dict.fromkeys([col, "average_precision_AP", "roc_auc"]))
    best = comparison.sort_values(keys, ascending=False, kind="mergesort").iloc[0]
    return best["modelo"], col, float(best[col])


def threshold_sweep(y, p, thresholds=None):
    """TPR, FPR, precisao e recall para varios limiares (como as curvas sao construidas)."""
    y, p = np.asarray(y), np.asarray(p)
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "limiar": t, "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "TPR_recall": tp / (tp + fn) if tp + fn else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan,
            "precisao": tp / (tp + fp) if tp + fp else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def plot_roc(y, probas, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, p), color, ls in zip(probas.items(), COLORS, STYLES):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls,
                label=f"{name} (ROC-AUC = {roc_auc_score(y, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Classificador aleatório (AUC = 0,5)")
    ax.set(title=title, xlabel="Taxa de falsos positivos (FPR)",
           ylabel="Taxa de verdadeiros positivos (TPR = Recall)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save_fig(fig, path)


def plot_pr(y, probas, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, p), color, ls in zip(probas.items(), COLORS, STYLES):
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=color, lw=2, ls=ls,
                label=(f"{name} (AP = {average_precision_score(y, p):.4f}; "
                       f"PR-AUC trap. = {auc(recall, precision):.4f})"))
    base = float(np.mean(y))
    ax.axhline(base, color="k", ls="--", lw=1,
               label=f"Linha de base (proporção de positivos = {base:.4f})")
    ax.set(title=title, xlabel="Revocação (Recall)", ylabel="Precisão",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save_fig(fig, path)


def plot_confusion(y, p, cfg, path, title, threshold=THRESHOLD):
    pred = (np.asarray(p) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    n = cm.sum()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / n:.2f}% de N)", ha="center",
                    va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    labels = [f"0 = {cfg['negative_label']}", f"1 = {cfg['positive_label']}"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(title=f"{title}\n(limiar = {threshold}; N = {n})", xlabel="Classe prevista",
           ylabel="Classe real")
    return _save_fig(fig, path)


# ---------------------------------------------------------------------------
# Resumo numerico
# ---------------------------------------------------------------------------
def write_summary(path, title, blocks):
    """Gera um .md so com numeros e tabelas. `blocks` = lista de (titulo, conteudo),
    onde conteudo e DataFrame, str ou lista de str."""
    parts = [f"# {title}", ""]
    for heading, content in blocks:
        parts += [f"## {heading}", ""]
        if isinstance(content, pd.DataFrame):
            parts.append(to_markdown(content))
        elif isinstance(content, (list, tuple)):
            parts += [f"- {c}" for c in content]
        else:
            parts.append(str(content))
        parts.append("")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
    return Path(path)
