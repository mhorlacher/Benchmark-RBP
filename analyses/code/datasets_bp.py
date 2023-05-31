from __future__ import annotations

import numpy as np
import pandas as pd



def read_bp_predictions(path: str) -> [DataFrame_bp]:
    # On the 2023-01-16: prediction files contain predictions from negative-1 and negative-2 models.
    # These can be separated on the column 'negativeset_model'. 
    # The file DOES NOT contain cross-negative predictions (i.e. applying a negative-1 model to negative-2 data).
    # So for all samples labels e.g. negative-1, the column 'negativeset_model' will be 'negative-1'.


class DataFrame_bp:

    EXPECTED_COLUMNS = ['model','train_dataset','rbp_dataset','fold',
                        'negativeset_model','matched_negativeset_test']

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._validate(dataframe)
        self.dataframe = dataframe
        self._parse_metadata(self)
    
    def _validate(self, dataframe) -> None:
        if any([col not in self.dataframe.columns for col in self.EXPECTED_COLUMNS]):
            raise ValueError("Dataframe does not contain all expected columns (see `DataFrame_bp.EXPECTED_COLUMNS`)")
    
    def _parse_metadata(self) -> None:
        meta = {}

        meta_names = ['model','train_dataset','rbp_dataset','fold','negativeset_model','matched_negativeset_test']

        model = self.dataframe['model'].unique()
        if model.shape[0] > 0: raise ValueError("More than one model in dataframe")

        meta['model'] = model[0]

        train_dataset = self.dataframe['train_dataset'].unique()
        if train_dataset.shape[0] > 0: raise ValueError("More than one train_dataset in dataframe")
        meta['train_dataset'] = train_dataset[0]

        rbp_dataset = self.dataframe['rbp_dataset'].unique()
        if rbp_dataset.shape[0] > 0: raise ValueError("More than one rbp_dataset in dataframe")
        meta['rbp_dataset'] = rbp_dataset[0]

