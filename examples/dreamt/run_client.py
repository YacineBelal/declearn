import datetime
import logging
import os

import fire

import declearn
import declearn.model.torch
from declearn.utils import config_client_loggers

FILEDIR = os.path.dirname(os.path.abspath(__file__))


def run_client(
    client_name: str,
    data_folder: str,
    protocol: str = "websockets",
    serv_uri: str = "wss://localhost:8765",
    verbose: bool = True,    
) -> None: 


    declearn.utils.set_device_policy(gpu=True)
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", client_name)
    config_client_loggers(
        client_name=client_name,
        level=logging.INFO,
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    data_folder = os.path.join(FILEDIR, data_folder, client_name)

    train = declearn.dataset.InMemoryDataset(
        os.path.join(data_folder, "train_data.npy"),
        os.path.join(data_folder, "train_target.npy"),
    )
    valid = declearn.dataset.InMemoryDataset(
        os.path.join(data_folder, "valid_data.npy"),
        os.path.join(data_folder, "valid_target.npy"),
    )

    network = declearn.communication.build_client(
        protocol=protocol,
        server_uri=serv_uri,
        name=client_name,
    )


    client = declearn.main.FederatedClient(
        netwk=network,
        train_data=train,
        valid_data=valid,
        checkpoint=checkpoint,
        verbose=verbose,
    )
    client.run()



def main():
    fire.Fire(run_client)


if __name__ == "__main__":
    main()
