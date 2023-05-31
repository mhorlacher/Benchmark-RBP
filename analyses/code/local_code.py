from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics


class Metadata_DataFrame:
    def __init__(self, dataframe):
        self.df = dataframe
        self.meta = {}


def parse_df(
    read_df: pd.DataFrame, meta_info: dict = None, verbose: bool = False
) -> Metadata_DataFrame:
    # Verifications
    for column in ["dataset", "RBP_dataset", "fold", "model_negativeset"]:
        if not read_df[column].unique().shape[0] == 1:
            raise ValueError(f"More than one unique value in `{column}`")

    if read_df["prediction"].isna().any():
        if read_df["prediction"].isna().all():
            raise ValueError(f"ALL PREDICTIONS ARE NA")
        else:
            warnings.warn(
                f"Dropping {read_df['prediction'].isna().sum():,} / {read_df['prediction'].shape[0]:,}"
            )

    # Currently we only expect one type of negative in the `true_class` column
    # As i) we did not implement multi-class such as Pysster and ii) cross-prediction is not applied yet.

    if read_df["true_class"].unique().shape[0] > 2:
        raise NotImplementedError("More than two classes found in `true_class`")

    known_true_classes = ["positive", "negative-1", "negative-2"]
    unknown_class = set(read_df["true_class"].unique()) - set(known_true_classes)

    if len(unknown_class) > 0:
        raise NotImplementedError(
            f"Detected one or more `true_class` that is unknown: {unknown_class}"
        )

    sample_negset = (
        "negative-1" if "negative-1" in read_df["true_class"].unique() else "negative-2"
    )

    # Proceed with the metadata annotated dataframe.

    df = Metadata_DataFrame(dataframe=read_df)
    if not meta_info:
        meta_info = {}

    for field in [
        "dataset",
        "RBP_dataset",
        "fold",
        "model_negativeset",
        "sample_negset",
    ]:
        if field not in meta_info:
            if read_df[field].unique().shape[0] > 1:
                raise ValueError(
                    f"No `{field}` provided, and more than one unique value in `dataset`"
                )
            else:
                df.meta[field] = meta_info.get(field, read_df[field].unique()[0])

    df.meta["dataset"] = meta_info.get("dataset", read_df["dataset"].unique()[0])
    df.meta["RBP_dataset"] = meta_info.get(
        "RBP_dataset", read_df["RBP_dataset"].unique()[0]
    )
    df.meta["fold"] = meta_info.get("fold", read_df["fold"].unique()[0])
    df.meta["model_negativeset"] = meta_info.get(
        "model_negativeset", read_df["model_negativeset"].unique()[0]
    )
    df.meta["sample_negset"] = meta_info.get("sample_negset", sample_negset)

    if verbose:
        print(";".join(["{k}={v}" for k, v in df.meta.items()] + ["N={df.shape[0]:,}"]))

    return df


def get_auroc(df):
    y_true = (
        df["true_class"].map({"negative-1": 0, "negative-2": 0, "positive": 1}).values
    )
    y_pred = df["prediction"].values
    auroc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    return auroc


def summarize_preds(df):
    df = df.copy()
    df["true_class"] = (
        df["true_class"]
        .map({"negative-1": "neg", "negative-2": "neg", "positive": "pos"})
        .values
    )

    mean_min_max = {
        "cts": df.shape[0],
        "pred.min": df["prediction"].min(),
        "pred.max": df["prediction"].max(),
        "pred.mean": df["prediction"].mean(),
        "pred.std": df["prediction"].std(),
    }

    pc_cts = {
        "pc_cts." + k: v for k, v in df.groupby("true_class").size().to_dict().items()
    }
    pc_mean_preds = {
        "pc_avg." + k: v
        for k, v in df.groupby("true_class")["prediction"].mean().to_dict().items()
    }
    pc_min_preds = {
        "pc_min." + k: v
        for k, v in df.groupby("true_class")["prediction"].min().to_dict().items()
    }
    pc_max_preds = {
        "pc_max." + k: v
        for k, v in df.groupby("true_class")["prediction"].max().to_dict().items()
    }
    pc_preds = {**pc_cts, **pc_min_preds, **pc_max_preds, **pc_mean_preds}

    thresh = df["prediction"].mean()
    y_true = (
        df["true_class"]
        .map(
            {
                "neg": 0,
                "pos": 1,
            }
        )
        .values
    )
    y_pred = (df["prediction"] > thresh).astype(int).values

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        y_true, y_pred, normalize="all"
    ).ravel()
    cm = {"cm.tn": tn, "cm.fp": fp, "cm.fn": fn, "cm.tp": tp, "cm.thresh": thresh}

    summarized = pd.Series(
        {
            **mean_min_max,
            **pc_preds,
            **cm,
        }
    )
    return summarized
