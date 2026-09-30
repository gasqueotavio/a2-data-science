import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.cutoff_hist import (assign_band, assign_bins, band_report, choose_cutoffs, make_bins,
                             histogram_table, plot_cutoff_histogram)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(42)
    n = 2000
    y = (rng.random(n) < 0.25).astype(int)
    p = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    p[:4] = [0.0, 1.0, 0.3, 0.7]
    return y, p


def test_extremos_caem_no_primeiro_e_no_ultimo_bin():
    edges = make_bins(0.1)
    idx = assign_bins(np.array([0.0, 1.0, 0.1, 0.999999]), edges)
    assert idx[0] == 0
    assert idx[1] == len(edges) - 2
    assert idx[2] == 1           # 0.1 abre o segundo bin: [0.1, 0.2)
    assert idx[3] == len(edges) - 2


def test_azul_soma_100(synthetic):
    y, p = synthetic
    tab = histogram_table(y, p, 0.1)
    assert tab["n_todas"].sum() == len(y)
    assert tab["pct_todas_sobre_N"].sum() == pytest.approx(100.0)


def test_vermelho_usa_denominador_N(synthetic):
    y, p = synthetic
    tab = histogram_table(y, p, 0.1)
    assert tab["n_positivas"].sum() == y.sum()
    assert tab["pct_positivas_sobre_N"].sum() == pytest.approx(100 * y.sum() / len(y))


def test_fronteiras_das_faixas():
    band = assign_band(np.array([0.2999, 0.3, 0.6999, 0.7, 1.0]), 0.3, 0.7)
    assert band.tolist() == [0, 1, 1, 2, 2]   # p = t1 -> manual; p = t2 -> positiva


@pytest.mark.parametrize("w", [0.05, 0.1, 0.2])
def test_larguras_de_bin(synthetic, w):
    y, p = synthetic
    tab = histogram_table(y, p, w)
    assert len(tab) == round(1 / w)
    assert tab["n_todas"].sum() == len(y)
    fig, _ = plot_cutoff_histogram(y, p, w, 0.3, 0.7, positive_label="teste")
    assert fig is not None


def test_band_report_consistente(synthetic):
    y, p = synthetic
    rep = band_report(y, p, 0.3, 0.7)
    faixas = rep["faixas"].set_index("faixa")
    assert faixas.loc["Total", "n"] == len(y)
    assert faixas["n"].iloc[:3].sum() == len(y)
    erros = rep["erros"].set_index("metrica")["valor"]
    assert erros["fn_auto"] == ((y == 1) & (p < 0.3)).sum()
    assert erros["fp_auto"] == ((y == 0) & (p >= 0.7)).sum()
    assert erros["fn_auto_sobre_positivos"] == pytest.approx(erros["fn_auto"] / y.sum())
    assert erros["fp_auto_sobre_N"] == pytest.approx(erros["fp_auto"] / len(y))
    assert erros["cobertura_automatica"] + erros["volume_manual"] == pytest.approx(1.0)


def test_cortes_invalidos():
    y, p = np.array([0, 1]), np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        band_report(y, p, 0.7, 0.3)
    with pytest.raises(ValueError):
        band_report(y, p, -0.1, 0.5)


def test_choose_cutoffs_respeita_restricoes(synthetic):
    y, p = synthetic
    (t1, t2), rep, tradeoff = choose_cutoffs(y, p, max_fn_rate=0.02, max_fp_rate=0.01)
    assert t1 < t2
    erros = rep["erros"].set_index("metrica")["valor"]
    assert erros["fn_auto_sobre_positivos"] <= 0.02
    assert erros["fp_auto_sobre_negativos"] <= 0.01
    assert tradeoff["viavel"].all()
    assert tradeoff["volume_manual"].is_monotonic_increasing


def test_choose_cutoffs_caso_separavel():
    # classes perfeitamente separadas: nenhum erro e volume manual zero
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.7, 1.0, 50)])
    (t1, t2), rep, _ = choose_cutoffs(y, p, 0.0, 0.0)
    erros = rep["erros"].set_index("metrica")["valor"]
    assert erros["fn_auto"] == 0 and erros["fp_auto"] == 0
    assert erros["volume_manual"] == 0


def test_choose_cutoffs_inviavel_avisa():
    y = np.array([0, 0, 1, 1])
    p = np.array([1.0, 1.0, 0.0, 0.0])   # negativos com p = 1: fp nunca chega a 0
    with pytest.warns(UserWarning):
        _, _, tradeoff = choose_cutoffs(y, p, 0.0, 0.0)
    assert not tradeoff["viavel"].any()
