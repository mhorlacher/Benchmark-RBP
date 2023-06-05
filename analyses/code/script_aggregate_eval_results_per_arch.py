#! /usr/bin/env python


import sys

import pandas as pd
import polars as pl

sys.path.insert(0, "code/")

import argparse
import logging
from pathlib import Path

import local_code
from dotmap import DotMap

logging.getLogger().setLevel(logging.INFO)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_results_table",
        help=("Table of results (e.g. 'results.Pysster.csv.gz'"),
        required=True,
        type=str,
    )

    parser.add_argument(
        "--arch",
        help=('Name of the method (e.g. "Pysster-101")'),
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_fp",
        help=("Path to output file to write aggregate results into."),
        required=True,
        type=str,
    )

    parser.add_argument(
        "--columns",
        help=('Columns of the results table ; "c1,c2,c3..."'),
        required=True,
        type=str,
    )

    return parser


def main():
    parser = argparser()
    args = parser.parse_args()

    params = DotMap()
    params["arch"] = args.arch
    params["columns"] = (args.columns).split(",")
    print(params.columns)

    input = DotMap()
    input["path_results_table"] = args.path_results_table

    output = DotMap()
    output["output_fp"] = args.output_fp

    auroc_table = None
    summarized_table = None

    logging.info(f"{params.arch} - LOADING RESULTS TABLES")

    path = Path(input.path_results_table)
    if not path.exists():
        raise ValueError(f"\t{params.arch} results not found.")

    d = pl.read_csv(path, separator=",", has_header=False).to_pandas()
    d.columns = params.columns

    logging.info(f"{params.arch} - PARSING")

    grouping = ["dataset", "RBP_dataset", "fold", "model_negativeset"]
    results_arch = []
    for group, group_df in d.groupby(grouping):
        meta_info = dict(zip(grouping, group))

        try:
            df = local_code.parse_df(group_df, meta_info=meta_info)
            # NOTE: this is done when merging across methods now.
            # Add in a unique ID.  This corresponds to a unique combination (dataset, rbp_dataset, fold, model_negativeset, sample_negativeset)
            # df.meta['unique_id'] = unique_id
            # unique_id +=1
            results_arch.append(df)

        except Exception as e:
            group_name = ";".join([f"{k}:{v}" for k, v in meta_info.items()])
            print(e)
            logging.error(f"ERROR parsing {group_name} - {e}")
            logging.info(f"Error parsing {group_name} ; skipping.")
            raise e

    logging.info(f"{params.arch} - auROCs computation")

    r_arch = pd.DataFrame(
        [
            pd.Series(
                {
                    **{"auroc": local_code.get_auroc(ann_df.df), "arch": params.arch},
                    **ann_df.meta,
                }
            )
            for ann_df in results_arch
        ]
    )
    auroc_table = r_arch

    logging.info(f"{params.arch} - summary computation")

    r_arch = pd.DataFrame(
        [
            pd.Series(
                {**ann_df.meta, **local_code.summarize_preds(ann_df.df).to_dict()}
            )
            for ann_df in results_arch
        ]
    )

    summarized_table = r_arch

    if auroc_table.shape[0] != summarized_table.shape[0]:
        error_msg = f"{params.arch} - ERROR: tables should be in sync."
        logging.error(error_msg)
        raise ValueError(error_msg)

    full_summarized_table = pd.concat(
        [
            auroc_table,
            summarized_table,  # .iloc[:,6:]
        ],
        axis=1,
    )

    # if not (
    #    full_summarized_table["model_negativeset"].values
    #    == full_summarized_table["sample_negset"].values
    # ).all():
    #    warn_msg = f"{params.arch} Table contain cross-negative predictions."
    #    logging.warn(warn_msg)

    full_summarized_table.to_csv(
        output.output_fp,
        header=True,
        index=False,
        sep="\t",
        compression="gzip",
    )

    logging.info(f"{params.arch} - DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
