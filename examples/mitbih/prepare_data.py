
import os

import fire
import numpy as np

from declearn.dataset.examples import load_mit_bih
from declearn.dataset.utils import aami_split, save_data_array

DATADIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_mitbih(
        folder: str=DATADIR,
) -> str:
    
    X_all, y_all, RR_all  = load_mit_bih(folder)
    
    splits = aami_split()
    # full test set shared between all clients. related TODO: override FederatedServer to only evaluate one client at the end (global model)
    X_test = np.concatenate([X_all[i] for i in splits.test])
    y_test = np.concatenate([y_all[i] for i in splits.test])
    RR_test = np.concatenate([RR_all[i] for i in splits.test])


    folder = os.path.join(folder,"mit-bih-aami")


    #TODO: add normalization before storing
    for client_id, patient_id in enumerate(splits.train):
        save_data_array(
            os.path.join(folder, f"client_{client_id}", "train_data"),
            X_all[patient_id])
        save_data_array(
            os.path.join(folder, f"client_{client_id}", "train_arr"),
            RR_all[patient_id])
        save_data_array(
            os.path.join(folder, f"client_{client_id}", "train_target"),
            y_all[patient_id])
        save_data_array(
            os.path.join(folder, f"client_{client_id}", "valid_data"),
            X_test)
        save_data_array(
            os.path.join(folder, f"client_{client_id}", "valid_arr"),
            RR_test)
        save_data_array(
            os.path.join(folder, f"client_{client_id}", "valid_target"),
            y_test)
                

    return folder



if __name__ == "__main__":
    fire.Fire(prepare_mitbih)