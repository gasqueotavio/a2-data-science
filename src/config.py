"""Configuracao central do projeto: sementes, caminhos e parametros por dominio.

Os limites de erro das faixas de decisao (max_fn_rate / max_fp_rate) ficam aqui,
num unico lugar, para a equipe ajustar e justificar.
"""

from pathlib import Path

RANDOM_STATE = 42
THRESHOLD = 0.5          # limiar das metricas que dependem de decisao
CV_SPLITS = 5
TEST_SIZE = 0.20         # 60/20/20 -> teste = 20% do total
VAL_SIZE = 0.25          # validacao = 25% dos 80% restantes = 20% do total
SHAP_MAX_SAMPLES = 1000  # linhas do teste explicadas
SHAP_BACKGROUND = 100    # linhas do treino usadas como fundo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = ROOT / "outputs"

DOMAINS = {
    "spam": {
        "csv": DATA_DIR / "spam.csv",
        "positive_question": "é spam?",
        "positive_label": "spam",
        "negative_label": "ham (legítima)",
        "target_metric": "average_precision",
        "search": "grid",
        "bin_width": 0.10,
        "max_fn_rate": 0.10,   # sobre o total de positivos
        "max_fp_rate": 0.01,   # sobre o total de negativos
    },
    "fraude": {
        "csv": DATA_DIR / "fraude.csv",
        "positive_question": "é fraude?",
        "positive_label": "fraude",
        "negative_label": "legítima",
        "target_metric": "average_precision",
        "search": "random",
        "n_iter": 15,
        "bin_width": 0.05,
        "max_fn_rate": 0.05,
        "max_fp_rate": 0.02,
    },
    "exame_medico": {
        "csv": DATA_DIR / "exame_medico.csv",
        "positive_question": "é maligno?",
        "positive_label": "maligno",
        "negative_label": "benigno",
        "target_metric": "roc_auc",
        "search": "grid",
        "bin_width": 0.10,
        "max_fn_rate": 0.01,
        "max_fp_rate": 0.10,
    },
}

# nome da metrica de tuning -> coluna correspondente na tabela de comparacao
METRIC_COLUMN = {"average_precision": "average_precision_AP", "roc_auc": "roc_auc"}


def output_dir(domain):
    """Cria (se preciso) e retorna outputs/<dominio>/ com subpastas models/ e faixas/."""
    out = OUTPUTS_DIR / domain
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "faixas").mkdir(parents=True, exist_ok=True)
    return out
