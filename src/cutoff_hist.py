"""PARTE 1 - Histograma de probabilidades e faixas de decisao (programa reutilizavel).

Funciona com qualquer base e qualquer modelo binario que forneca a probabilidade
da classe positiva. Entrada minima: y_true (0/1) e y_proba (0-1), para TODAS as
instancias do conjunto avaliado, independentemente do rotulo real.

Faixas (t1 < t2):
    Negativa automatica : p < t1
    Analise manual      : t1 <= p < t2
    Positiva automatica : p >= t2

Uso como biblioteca:
    from src.cutoff_hist import plot_cutoff_histogram, band_report, choose_cutoffs

Uso por linha de comando (CSV com colunas y_true e y_proba):
    python -m src.cutoff_hist --csv preds.csv --bin 0.1 --t1 0.3 --t2 0.7 --label "spam" --out fig.png
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BLUE = "#1f4e9c"
RED = "#d62728"
BAND_NAMES = ["Negativa automática", "Análise manual", "Positiva automática"]
BAND_RULES = ["p < t1", "t1 ≤ p < t2", "p ≥ t2"]


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _validate(y_true, y_proba):
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    if y_true.shape != y_proba.shape:
        raise ValueError("y_true e y_proba precisam ter o mesmo tamanho.")
    if len(y_true) == 0:
        raise ValueError("Conjunto vazio.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true deve conter apenas 0 (negativo) e 1 (positivo).")
    if np.isnan(y_proba).any() or y_proba.min() < 0 or y_proba.max() > 1:
        raise ValueError("y_proba deve conter probabilidades no intervalo [0, 1].")
    return y_true.astype(int), y_proba


def _check_cutoffs(t1, t2):
    if not (0 <= t1 <= 1 and 0 <= t2 <= 1):
        raise ValueError(f"t1 e t2 devem estar em [0, 1] (recebido t1={t1}, t2={t2}).")
    if not t1 < t2:
        raise ValueError(f"É preciso t1 < t2 (recebido t1={t1}, t2={t2}).")


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------
def make_bins(bin_width):
    """Bordas 0, w, 2w, ..., 1.0. Se 1/w nao for inteiro, o ultimo bin fica menor."""
    if not 0 < bin_width <= 1:
        raise ValueError("bin_width deve estar em (0, 1]. Ex.: 0.1 = 10 pontos percentuais.")
    edges = np.round(np.arange(0.0, 1.0, bin_width), 10)
    return np.append(edges, 1.0)


def assign_bins(y_proba, edges):
    """Indice do bin de cada instancia. Bins [a, b); o ultimo e [a, 1.0]."""
    idx = np.searchsorted(edges, np.asarray(y_proba, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def histogram_table(y_true, y_proba, bin_width=0.1):
    """Contagens e percentuais por bin. Denominador das duas distribuicoes = N."""
    y_true, y_proba = _validate(y_true, y_proba)
    edges = make_bins(bin_width)
    idx = assign_bins(y_proba, edges)
    n = len(y_true)
    n_bins = len(edges) - 1
    n_all = np.bincount(idx, minlength=n_bins)
    n_pos = np.bincount(idx[y_true == 1], minlength=n_bins)
    assert n_all.sum() == n, "Cada instância deve cair em exatamente um bin."

    labels = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        close = "]" if i == n_bins - 1 else ")"
        labels.append(f"[{100 * a:g}%, {100 * b:g}%{close}")
    return pd.DataFrame({
        "bin": labels,
        "inicio": edges[:-1],
        "fim": edges[1:],
        "n_todas": n_all,
        "n_positivas": n_pos,
        "pct_todas_sobre_N": 100 * n_all / n,
        "pct_positivas_sobre_N": 100 * n_pos / n,
    })


# ---------------------------------------------------------------------------
# Faixas
# ---------------------------------------------------------------------------
def assign_band(y_proba, t1, t2):
    """0 = negativa automatica, 1 = analise manual, 2 = positiva automatica."""
    _check_cutoffs(t1, t2)
    p = np.asarray(y_proba, dtype=float)
    return np.where(p < t1, 0, np.where(p < t2, 1, 2))


def band_report(y_true, y_proba, t1, t2):
    """Retorna {"faixas": DataFrame por faixa, "erros": DataFrame de erros e cobertura}."""
    _check_cutoffs(t1, t2)
    y_true, y_proba = _validate(y_true, y_proba)
    band = assign_band(y_proba, t1, t2)
    n = len(y_true)
    total_pos = int(y_true.sum())
    total_neg = n - total_pos

    rows = []
    for k in range(3):
        m = band == k
        n_k = int(m.sum())
        pos_k = int(y_true[m].sum())
        neg_k = n_k - pos_k
        rows.append({
            "faixa": BAND_NAMES[k],
            "regra": BAND_RULES[k],
            "n": n_k,
            "pct_populacao_sobre_N": 100 * n_k / n,
            "n_positivos": pos_k,
            "pct_positivos_sobre_n_faixa": 100 * pos_k / n_k if n_k else np.nan,
            "n_negativos": neg_k,
            "pct_negativos_sobre_n_faixa": 100 * neg_k / n_k if n_k else np.nan,
        })
    rows.append({
        "faixa": "Total", "regra": f"t1 = {t1:g}, t2 = {t2:g}", "n": n,
        "pct_populacao_sobre_N": 100.0,
        "n_positivos": total_pos, "pct_positivos_sobre_n_faixa": 100 * total_pos / n,
        "n_negativos": total_neg, "pct_negativos_sobre_n_faixa": 100 * total_neg / n,
    })
    bands = pd.DataFrame(rows)

    fn_auto = int(bands.loc[0, "n_positivos"])   # positivos com p < t1
    fp_auto = int(bands.loc[2, "n_negativos"])   # negativos com p >= t2
    n_manual = int(bands.loc[1, "n"])

    def count(metric, value, text):
        return {"metrica": metric, "valor": value, "valor_pct": np.nan, "numerador": value,
                "denominador": np.nan, "descricao_denominador": "— (contagem)",
                "descricao": text}

    def rate(metric, num, den, den_text, text):
        value = num / den if den else np.nan
        return {"metrica": metric, "valor": value, "valor_pct": 100 * value, "numerador": num,
                "denominador": den, "descricao_denominador": den_text, "descricao": text}

    errors = pd.DataFrame([
        count("fn_auto", fn_auto,
              "positivos classificados automaticamente como negativos (p < t1)"),
        rate("fn_auto_sobre_positivos", fn_auto, total_pos, "total de positivos",
             "proporção dos positivos que caíram na faixa negativa automática"),
        rate("fn_auto_sobre_N", fn_auto, n, "N (total de instâncias)",
             "proporção de N que é positivo e caiu na faixa negativa automática"),
        count("fp_auto", fp_auto,
              "negativos classificados automaticamente como positivos (p ≥ t2)"),
        rate("fp_auto_sobre_negativos", fp_auto, total_neg, "total de negativos",
             "proporção dos negativos que caíram na faixa positiva automática"),
        rate("fp_auto_sobre_N", fp_auto, n, "N (total de instâncias)",
             "proporção de N que é negativo e caiu na faixa positiva automática"),
        rate("cobertura_automatica", n - n_manual, n, "N (total de instâncias)",
             "proporção de N decidida automaticamente (fora da faixa manual)"),
        rate("volume_manual", n_manual, n, "N (total de instâncias)",
             "proporção de N encaminhada à análise manual"),
    ])
    errors["numerador"] = errors["numerador"].astype("Int64")
    errors["denominador"] = errors["denominador"].astype("Int64")
    return {"faixas": bands, "erros": errors}


# ---------------------------------------------------------------------------
# Escolha dos cortes (somente dados de validacao)
# ---------------------------------------------------------------------------
def choose_cutoffs(y_val, p_val, max_fn_rate, max_fp_rate, step=0.01):
    """Busca em grade todos os pares t1 < t2 (passo `step`) na VALIDACAO.

    Restricoes: fn_auto_sobre_positivos <= max_fn_rate e
                fp_auto_sobre_negativos <= max_fp_rate.
    Objetivo:   minimizar o volume manual (maximizar a cobertura automatica).
    Desempate:  menos erros automaticos (fn_auto + fp_auto).

    Retorna ((t1, t2), band_report na validacao, tabela de trade-off com os 10
    melhores pares viaveis). Se nenhum par for viavel, devolve o par de menor
    violacao e emite um aviso (coluna `viavel` = False na tabela).
    """
    y_val, p_val = _validate(y_val, p_val)
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    grid = grid[grid <= 1.0]
    pos = np.sort(p_val[y_val == 1])
    neg = np.sort(p_val[y_val == 0])
    allp = np.sort(p_val)
    n, n_pos, n_neg = len(allp), len(pos), len(neg)

    fn = np.searchsorted(pos, grid, side="left")          # positivos com p < t
    fp = n_neg - np.searchsorted(neg, grid, side="left")  # negativos com p >= t
    below = np.searchsorted(allp, grid, side="left")      # instancias com p < t

    i1, i2 = np.triu_indices(len(grid), k=1)              # todos os pares t1 < t2
    fn_rate = fn[i1] / n_pos if n_pos else np.zeros(len(i1))
    fp_rate = fp[i2] / n_neg if n_neg else np.zeros(len(i2))
    manual = (below[i2] - below[i1]) / n
    violation = np.maximum(fn_rate - max_fn_rate, 0) + np.maximum(fp_rate - max_fp_rate, 0)

    table = pd.DataFrame({
        "t1": grid[i1], "t2": grid[i2],
        "volume_manual": manual, "cobertura_automatica": 1 - manual,
        "fn_auto": fn[i1], "fn_auto_sobre_positivos": fn_rate,
        "fp_auto": fp[i2], "fp_auto_sobre_negativos": fp_rate,
        "violacao": violation,
    })
    table["erros_auto"] = table["fn_auto"] + table["fp_auto"]
    feasible = table[table["violacao"] == 0]
    if len(feasible):
        ranked = feasible.sort_values(["volume_manual", "erros_auto", "t1"],
                                      ascending=[True, True, False])
        ranked = ranked.assign(viavel=True)
    else:
        warnings.warn(
            "Nenhum par (t1, t2) satisfaz as restrições "
            f"(max_fn_rate={max_fn_rate}, max_fp_rate={max_fp_rate}). "
            "Retornando o par de MENOR violação; revise os limites.", stacklevel=2)
        ranked = table.sort_values(["violacao", "volume_manual"]).assign(viavel=False)

    best = ranked.iloc[0]
    t1, t2 = float(best["t1"]), float(best["t2"])
    tradeoff = ranked.head(10).drop(columns="erros_auto").reset_index(drop=True)
    return (t1, t2), band_report(y_val, p_val, t1, t2), tradeoff


# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------
def _fmt_pct(v):
    return f"{v:.2f}%" if v == 0 or v >= 0.01 else f"{v:.3f}%"


def plot_cutoff_histogram(y_true, y_proba, bin_width=0.1, t1=None, t2=None,
                          positive_label="positivo", title="", save_path=None,
                          dataset_name=None, dpi=200):
    """Histograma lado a lado: azul = todas as instancias, vermelho = rotulo real positivo.

    Eixo Y = % do TOTAL N do conjunto avaliado, para as duas cores.
    Retorna (fig, tabela_do_histograma). Salva em `save_path` se informado.
    """
    import matplotlib.pyplot as plt

    y_true, y_proba = _validate(y_true, y_proba)
    n = len(y_true)
    tab = histogram_table(y_true, y_proba, bin_width)
    left = 100 * tab["inicio"].to_numpy()
    width = 100 * (tab["fim"] - tab["inicio"]).to_numpy()
    bar_w = width * 0.42
    x_blue = left + width / 2 - bar_w / 2
    x_red = left + width / 2 + bar_w / 2

    n_bins = len(tab)
    fig, ax = plt.subplots(figsize=(max(11, 0.6 * n_bins + 4), 6.5))
    ax.bar(x_blue, tab["pct_todas_sobre_N"], width=bar_w, color=BLUE,
           label="Todas as instâncias", zorder=3)
    ax.bar(x_red, tab["pct_positivas_sobre_N"], width=bar_w, color=RED,
           label=f"Rótulo real = {positive_label}", zorder=3)

    fs = 8 if n_bins <= 12 else 6.5
    rot = 0 if n_bins <= 12 else 90
    for xs, col, color in [(x_blue, "pct_todas_sobre_N", BLUE),
                           (x_red, "pct_positivas_sobre_N", RED)]:
        for x, v in zip(xs, tab[col]):
            ax.text(x, v, " " + _fmt_pct(v) if rot else _fmt_pct(v), ha="center",
                    va="bottom", fontsize=fs, color=color, rotation=rot, zorder=4)

    top = max(tab["pct_todas_sobre_N"].max(), 1.0)
    ax.set_ylim(0, top * 1.45)

    if t1 is not None and t2 is not None:
        rep = band_report(y_true, y_proba, t1, t2)["faixas"]
        spans = [(0, 100 * t1), (100 * t1, 100 * t2), (100 * t2, 100)]
        shades = ["#e6eefa", "#fff3d1", "#fde4e4"]
        for k, ((a, b), shade) in enumerate(zip(spans, shades)):
            ax.axvspan(a, b, color=shade, alpha=0.7, zorder=0)
            r = rep.iloc[k]
            txt = f"{r['faixa']}\n{r['regra']}\n{r['pct_populacao_sobre_N']:.2f}% de N (n={r['n']})"
            y_txt = 0.98 if k != 1 else 0.86  # alterna altura p/ faixas estreitas
            x_txt = min(max((a + b) / 2, 9), 91)
            ax.text(x_txt, y_txt, txt, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
                    zorder=5)
        for name, t in (("t1", t1), ("t2", t2)):
            ax.axvline(100 * t, color="black", ls="--", lw=1.3, zorder=4)
            ax.text(100 * t, 0.55, f"{name} = {100 * t:g}% ", transform=ax.get_xaxis_transform(),
                    rotation=90, ha="right", va="center", fontsize=9, zorder=5)

    ax.set_xlim(0, 100)
    ax.set_xticks(100 * np.append(tab["inicio"].to_numpy(), 1.0))
    ax.tick_params(axis="x", labelsize=8 if n_bins <= 12 else 7)
    ax.set_xlabel("Probabilidade estimada da classe positiva (%)")
    ax.set_ylabel("% das instâncias sobre o total N do conjunto")
    base = title or "Distribuição das probabilidades previstas"
    suffix = f"{dataset_name} — N = {n}" if dataset_name else f"N = {n}"
    ax.set_title(f"{base}\n({suffix})")
    ax.legend(loc="center right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig, tab


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Histograma de probabilidades e faixas de decisão")
    ap.add_argument("--csv", required=True, help="CSV com colunas y_true e y_proba")
    ap.add_argument("--bin", type=float, default=0.1, help="largura do bin (0.1 = 10 p.p.)")
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--t2", type=float, required=True)
    ap.add_argument("--label", default="positivo", help="nome da classe positiva")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="histograma_faixas.png")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    missing = {"y_true", "y_proba"} - set(df.columns)
    if missing:
        raise SystemExit(f"Colunas ausentes no CSV: {sorted(missing)}")
    plot_cutoff_histogram(df["y_true"], df["y_proba"], a.bin, a.t1, a.t2, a.label,
                          a.title, a.out, dataset_name=Path(a.csv).name)
    rep = band_report(df["y_true"], df["y_proba"], a.t1, a.t2)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("=== Faixas ===")
        print(rep["faixas"].round(4).to_string(index=False))
        print("\n=== Erros e cobertura ===")
        print(rep["erros"].drop(columns="descricao").round(4).to_string(index=False))
    print(f"\nGráfico salvo em {a.out}")


if __name__ == "__main__":
    main()
