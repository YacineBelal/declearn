
import os

import fire
import numpy as np

from declearn.dataset.examples import load_mit_bih
from declearn.dataset.utils import aami_split, save_data_array

DATADIR = os.path.join(os.path.dirname(__file__), "data")

AAMI_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "S": "S",
    "J": "S",
    "V": "V",
    "E": "V",
}

def prepare_mitbih(
        folder: str=DATADIR,
) -> str:
    X_all, y_all, SYM_all, RR_all = load_mit_bih(folder)
    
    splits = aami_split()
    # full test set shared between all clients. related TODO: override FederatedServer to only evaluate one client at the end (global model)
    X_test = np.concatenate([X_all[pid] for pid in splits.test])
    y_test = np.concatenate([y_all[pid] for pid in splits.test])
    RR_test = np.concatenate([RR_all[pid] for pid in splits.test])


    folder = os.path.join(folder,"mit-bih-aami")

    X_train_all = np.concatenate([X_all[pid] for pid in splits.train])
    y_train_all = np.concatenate([y_all[pid] for pid in splits.train])
    sym_train = np.concatenate([SYM_all[pid] for pid in splits.train])

    beat_symbols = list(AAMI_MAP.keys())

    matched_filters = np.stack(
        [
            X_train_all[sym_train == s].mean(axis=0)
            for s in beat_symbols
            if (sym_train == s).any()
        ]
    ).astype("float32")

    n_classes = len(np.unique(y_train_all))
    class_weights = len(y_train_all) / (n_classes * np.bincount(y_train_all))
    np.save(
        os.path.join(folder, "class_weights.npy"),
        class_weights.astype("float32"),
    )

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

    save_data_array(os.path.join(folder, "matched_filters"), matched_filters)

    return folder



if __name__ == "__main__":
    fire.Fire(prepare_mitbih)