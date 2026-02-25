
import os
from typing import Optional

import fire
import numpy as np

from declearn.dataset.examples import load_dreamt
from declearn.dataset.utils import save_data_array, train_valid_split

DATADIR = os.path.join(os.path.dirname(__file__), "data")

def prepare_dreamt(
        nb_clients: int, 
        folder: str=DATADIR,
        p_valid: float=0.2, 
        seed: Optional[int]= None,
) -> str:
    

    signals, labels, mean, std = load_dreamt(nb_clients, folder)
  
    
    print(f"DREAMT is already splitted by client. This simulation samples {nb_clients} out of 100 client.")

    
    #TODO: could probably avoid loading csv files into npy and use csv ones directly
    #TODO: we should perhaps add a splitting method that only splits into 
    # train/val for datasets where clients are explicit  
    rng = np.random.default_rng(seed)

    split_data = [train_valid_split(signals[i], labels[i], p_valid, rng) for i in range(len(signals))]
    
    folder = os.path.join(folder,"dreamt_natural")
    for idx, ((x_t, y_t), (x_v, y_v)) in enumerate(split_data):
        save_data_array(
            os.path.join(folder, f"client_{idx}", "train_data"), np.permute_dims(x_t, axes=(0,2,1)).astype("float32")
        )
        save_data_array(
            os.path.join(folder, f"client_{idx}", "train_target"), y_t
        )
        save_data_array(
            os.path.join(folder, f"client_{idx}", "valid_data"), np.permute_dims( x_v, axes=(0,2,1)).astype("float32")
        )
        save_data_array(
            os.path.join(folder, f"client_{idx}", "valid_target"), y_v
        )

    return folder 



if __name__ == "__main__":
    fire.Fire(prepare_dreamt)