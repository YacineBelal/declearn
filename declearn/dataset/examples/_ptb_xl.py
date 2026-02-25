
import ast
import os
from typing import Optional

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer


def convert_to_superclass(y_dict, agg_df):
    tmp = []
    for k in y_dict.keys():
        if k in agg_df.index:
            tmp.append(agg_df.loc[k].diagnostic_class)
    return list(set(tmp))



def load_ptb_xl(
    path: Optional[str] = None,
    sampling_rate: int=100, 
    super_class: bool= True
    ):
    """_summary_

    Parameters
    ----------
    path : Optional[str], optional
        _description_, by default None
    sampling_rate : int, optional
        Frequency in Hz, by default 100
    super_class : bool, optional
        Whether the superclass of each anomaly using a secondary file, by default True

    Returns
    -------
    signals: np.ndarray
    Input signals, as a (n_signals/n_patients, len_signal, n_channels) float numpy array.
    labels: np.ndarray 
    Target classes, as a (n_signals/n_patients, n_classes) binarized numpy array. 
    """    

    signals, labels = _load_ptb_xl(path, sampling_rate, super_class)
    if super_class:
        labels = _preprocess_ptb_xl_superclass(labels)
    # else encode subclasses 

    return signals, labels

def _load_ptb_xl(
        path: Optional[str],
        sampling_rate: int=100, 
        super_class: bool= True
):
    # Read dataset from local file
    if path is not None and os.path.isfile(os.path.join(path, "ptbxl_database.csv")):
        patients = pd.read_csv(os.path.join(path, "ptbxl_database.csv", index_col="ecg_id"))
        patients.scp_codes = patients.scp_codes.apply(lambda x: ast.literal_eval(x))
        #TODO: avoid doing this for all patients 
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in patients.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in patients.filename_hr]

    # Read superclass info if needed and file exist 
    if super_class and path is not None and os.path.isfile(os.path.join(path, "scp_statements.csv")):
            agg_df = pd.read_csv(os.path.join(path,"scp_statements.csv"), index_col=0)
            agg_df = agg_df[agg_df.diagnostic == 1]

    #TODO: add download dataset (both signals and scp statements to use if needed)

    signals = np.array([signal for signal, _ in data])
    patients["diagnostic_superclass"] = patients.scp_codes.apply(
        lambda x: _convert_to_superclass(x, agg_df)
    )

    return signals, patients["diagnostic_superclass"].to_numpy() 

def _convert_to_superclass(y_dict, agg_df):
    tmp = []
    for k in y_dict.keys():
        if k in agg_df.index:
            tmp.append(agg_df.loc[k].diagnostic_class)
    return list(set(tmp))

def _preprocess_ptb_xl_superclass(
        labels: np.ndarray):
    
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    return labels 