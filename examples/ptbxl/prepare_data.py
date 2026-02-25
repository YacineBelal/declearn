import os
from typing import Literal, Optional

from declearn.dataset.examples import load_ptb_xl
from declearn.dataset.utils import save_data_array

DATADIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")



def prepare_ptb_xl(
    nb_clients: int,
    scheme: Literal["same", "random", "biased"] = "same",
    folder: str=DATADIR,
    sampling_rate: int=100,
    p_valid: float=0.2,
    seed: Optional[int] = None 
) -> str: 
        
    datadir_raw = os.path.join(folder, "physionet.org")
    signals, labels = load_ptb_xl(datadir_raw, sampling_rate)

    signal_len = signals[0].shape[1]
    #dataset is already split patient-wise, #TODO: test if other schemes work/make sense
    if scheme == "same":
        nb_clients = min(nb_clients, signals.shape[0])
        signals = signals[:nb_clients]
        labels =  labels[:nb_clients]
    

    folder = os.path.join(folder, f"ptb_xl_{scheme}")


    for idx in range(len(signals)):
        save_data_array(
            os.path.join(folder,f"client_{idx}","train_data", signals[idx, :(1-p_valid) * signal_len])
        )
        save_data_array(
            os.path.join(folder,f"client_{idx}","train_target", labels[idx, :(1-p_valid) * signal_len])
        )
        save_data_array(
            os.path.join(folder,f"client_{idx}","train_data", signals[idx, :(1-p_valid) * signal_len])
        )
        save_data_array(
            os.path.join(folder,f"client_{idx}","train_data", signals[idx, :(1-p_valid) * signal_len])
        )




