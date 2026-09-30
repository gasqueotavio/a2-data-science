"""Padroniza as bases brutas e salva CSVs em data/ com a coluna `target` (1 = positivo).

NAO faz download: a equipe coloca os arquivos brutos em data/raw/ manualmente.

    data/raw/SMSSpamCollection  -> data/spam.csv          (positivo: é spam?)
    data/raw/creditcard.csv     -> data/fraude.csv + .zip (positivo: é fraude?)
    sklearn load_breast_cancer  -> data/exame_medico.csv  (positivo: é maligno?)

Execucao (a partir da raiz do projeto):
    python -m src.prepare_data                 # as tres bases
    python -m src.prepare_data --only spam     # apenas uma
"""

import argparse
import csv
import sys
import zipfile

import pandas as pd

from src.config import DATA_DIR, RAW_DIR

RAW_FILES = {
    "spam": ("SMSSpamCollection",
             "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"),
    "fraude": ("creditcard.csv",
               "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"),
}


def _require(domain):
    name, url = RAW_FILES[domain]
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"[{domain}] Arquivo bruto não encontrado: {path}\n"
            f"  Baixe manualmente em {url}\n"
            f"  e coloque o arquivo '{name}' na pasta {RAW_DIR}"
        )
    return path


def _drop_duplicates(df, domain):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[{domain}] duplicatas exatas removidas: {removed} (de {before} linhas)")
    return df, removed


def _report(df, domain):
    counts = df["target"].value_counts().reindex([1, 0], fill_value=0)
    pct = 100 * counts / len(df)
    print(f"[{domain}] shape = {df.shape}")
    for cls in (1, 0):
        print(f"    target = {cls}: {counts[cls]:>7d}  ({pct[cls]:.2f}%)")


def prepare_spam():
    path = _require("spam")
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df["target"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    return df[["text", "target"]]


def prepare_fraude():
    path = _require("fraude")
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError(f"Coluna 'Class' não encontrada em {path}.")
    return df.rename(columns={"Class": "target"}).astype({"target": int})


def prepare_exame_medico():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # No sklearn: 0 = maligno, 1 = benigno. Invertido para 1 = maligno (classe positiva).
    df["target"] = 1 - df["target"]
    return df


PREPARERS = {"spam": prepare_spam, "fraude": prepare_fraude,
             "exame_medico": prepare_exame_medico}


def prepare(domain):
    df = PREPARERS[domain]()
    df, removed = _drop_duplicates(df, domain)
    out = DATA_DIR / f"{domain}.csv"
    df.to_csv(out, index=False)
    _report(df, domain)
    print(f"[{domain}] salvo em {out}")
    if domain == "fraude":
        zpath = DATA_DIR / "fraude.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(out, arcname="fraude.csv")
        print(f"[{domain}] compactado em {zpath}")
    return {"dominio": domain, "linhas": len(df), "duplicatas_removidas": removed}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Padroniza as bases brutas")
    ap.add_argument("--only", choices=list(PREPARERS), default=None)
    a = ap.parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for domain in ([a.only] if a.only else PREPARERS):
        prepare(domain)
        print()


if __name__ == "__main__":
    main()
