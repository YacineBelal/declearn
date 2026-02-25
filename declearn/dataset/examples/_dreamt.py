import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

0
def load_dreamt(nb_clients: int,
                folder: Optional[str]= None):

    samples, labels = _load_dreamt(nb_clients, folder)
    signals, labels, mean, std = _preprocess_into_signals(samples, labels)

    return signals, labels, mean, std 



def _load_dreamt(
        nb_clients: int, 
        folder: Optional[str] = None
) -> Tuple[List[np.ndarray],List[np.ndarray]]:
    #TODO: could add other tasks 
    cols_to_drop = [
                    "IBI",
                    "TIMESTAMP",
                    "Obstructive_Apnea",
                    "Central_Apnea",
                    "Hypopnea",
                    "Multiple_Events",
                ]
    samples = []
    labels = []
    #TODO: fix number of users
    pattern = re.compile(r""".*S(\d\d\d).*\.csv""")
    
    #TODO: sample randomly users' identifiers 
    #TODO: add more targeted conditions here 
    #TODO: add download possibility 
    if folder is not None and os.path.isdir(folder):
        matching_files = (name for name in os.listdir(folder) if 
                        (os.path.isfile(os.path.join(folder, name))) and pattern.match(name)) 
        
        for i,file_name in enumerate(matching_files):
            if i >= nb_clients:
                    break 
    
            df = pd.read_csv(os.path.join(folder, file_name))
            df["Sleep_Stage"] = df["Sleep_Stage"].replace("P", "W")
            df = df.drop(columns=cols_to_drop)
            df = df[df["Sleep_Stage"] != "Missing"]
            labels.append(df["Sleep_Stage"].to_numpy())
            samples.append(df.drop(columns=["Sleep_Stage"]).to_numpy())
        

    return samples, labels 


def _preprocess_into_signals(
        signals: np.ndarray, 
        labels: np.ndarray, 
        signal_len: int=64
        ) -> Tuple[List[np.ndarray],List[np.ndarray]]:
   
    signals_preprocessed = []
    labels_preprocessed = [] 
    for X_p, y_p in zip(signals,labels):
        signals_preprocessed.append(X_p[:-1].reshape(-1, signal_len, 7))
        labels_preprocessed.append(y_p[:-1].reshape(-1, signal_len)[:,0])

    signals_concat_all_clients = np.concat(signals_preprocessed, axis=0)
    labels_concat_all_clients=  np.concat(labels_preprocessed, axis=0)
    lb = LabelEncoder()
    lb.fit(labels_concat_all_clients)
    #TODO: this could definitely be improved, we're regrouping data that should be
    #collected
    #Same questions for normalization 
    labels_preprocessed_encoded = []
    for y_p in labels_preprocessed:
        labels_preprocessed_encoded.append(lb.transform(y_p))

    mean = np.mean(signals_concat_all_clients, axis=(0,1))
    std = np.std(signals_concat_all_clients, axis=(0,1))
    

    return signals_preprocessed, labels_preprocessed_encoded, mean, std 










