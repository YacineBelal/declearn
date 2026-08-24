"""Script to run a federated client on the MIT-BIH example."""

import datetime
import logging
import os

import fire  # type: ignore
import numpy as np
import tensorflow as tf

import declearn
import declearn.model.tensorflow
from declearn.dataset.tensorflow._tensorflow import TensorflowDataset
from declearn.dataset.utils import load_data_array
from declearn.utils import config_client_loggers

FILEDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CERT = os.path.join(FILEDIR, "ca-cert.pem")

#TODO: This is a temporary helper for test purposes. It shouldn't live here 
def prepare_data(X, rr, y):
    n_classes = np.unique(y).size
    n_samples = y.shape[0]
    weights = load_data_array(
        os.path.join(FILEDIR, "data", "mit-bih-aami", "class_weights.npy")
    )
    # n_samples / n_classes * np.bincount(y)
    deriv_x = np.diff(X, axis=-1, prepend=X[..., :1])
    dataset = tf.data.Dataset.from_tensor_slices(
        # ((deriv_x, rr), y, weights[y])
        ((deriv_x, rr), y, weights[y])
    )

    return dataset, weights 

def run_client(
    client_name: str,
    data_folder: str,
    protocol: str = "websockets",
    serv_uri: str = "wss://localhost:8765",
    verbose: bool = True,
) -> None:
    """Instantiate and run a given client.

    Parameters
    ---------
    client_name: str
        Name of the client (i.e. center data from which to use).
    data_folder: str
        The parent folder of this client's data
    ca_cert: str, default="./ca-cert.pem"
        Path to the certificate authority file that was used to
        sign the server's SSL certificate.
    protocol: str, default="websockets"
        Name of the communication protocol to use.
    serv_uri: str, default="wss://localhost:8765"
        URI of the server to which to connect.
    verbose: bool, default=True
        Whether to log everything to the console, or filter out most non-error
        information.
    """

    declearn.utils.set_device_policy(gpu=True)

    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", client_name)
    config_client_loggers(
        client_name=client_name,
        level=logging.INFO,
        fpath=os.path.join(checkpoint, "logs.txt"),
    )
    data_folder = os.path.join(FILEDIR, data_folder, client_name)

    X_train = load_data_array(os.path.join(data_folder, "train_data.npy"))
    RR_train = load_data_array(os.path.join(data_folder, "train_arr.npy"))
    y_train = load_data_array(os.path.join(data_folder, "train_target.npy"))

    tf_dataset, weights = prepare_data(X_train, RR_train, y_train)
    train = TensorflowDataset(tf_dataset, seed=42)
    X_test = load_data_array(os.path.join(data_folder, "valid_data.npy"))
    RR_test = load_data_array(os.path.join(data_folder, "valid_arr.npy"))
    y_test = load_data_array(os.path.join(data_folder, "valid_target.npy"))
    tf_dataset_valid, _ = prepare_data(X_test, RR_test, y_test)
    valid = TensorflowDataset(tf_dataset_valid)

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
    "Fire-wrapped `run_client`."
    fire.Fire(run_client)


if __name__ == "__main__":
    main()
