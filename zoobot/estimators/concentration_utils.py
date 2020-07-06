import json

import numpy as np
import pandas as pd


def get_mean_concentration(df, concentration_cols):
    answer_predictions = []
    for col_n in range(len(concentration_cols)):
        answer_predictions.append(np.stack(list(df[concentration_cols[col_n]].values)))
    return np.stack(answer_predictions).transpose(1, 0, 2)


def load_all_concentrations(df, concentration_cols):
    temp = []
    for col in concentration_cols:
        temp.append(np.stack(df[col].apply(json.loads).values, axis=0))
    return np.stack(temp, axis=2).transpose(0, 2, 1)