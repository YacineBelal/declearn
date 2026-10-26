import datetime
import logging
import os
import random

import fire
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from model_tf import tinyCNN

import declearn
import declearn.model.tensorflow
from declearn.adversarial.rmia import RmiaControllerServer
from declearn.dataset.tensorflow._tensorflow import TensorflowDataset
from declearn.dataset.utils import load_data_array
from declearn.model.tensorflow import TensorflowModel
from declearn.optimizer.modules import (
    AdamModule,
    ScaffoldClientModule,
    ScaffoldServerModule,
)
from declearn.utils import config_server_loggers

FILEDIR = os.path.dirname(os.path.abspath(__file__))


def run_server(
    nb_clients: int,
    protocol: str = "websockets",
    host: str = "localhost",
    port: int = 8764,
) -> None:
    declearn.utils.set_device_policy(gpu=True)
    metrics = declearn.metrics.MetricSet(
        [
            declearn.metrics.MulticlassAccuracyPrecisionRecall(
                labels=range(3)
            ),
        ]
    )
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", "server")
    # Set up a logger, records from which will go to a file.
    config_server_loggers(
        level=logging.INFO,
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    weights = load_data_array(
        os.path.join(FILEDIR, "data", "mit-bih-aami", "class_weights.npy")
    )

    matched_filters = load_data_array(
        os.path.join(FILEDIR, "data", "mit-bih-aami", "matched_filters.npy")
    )

    model = TensorflowModel(
        tinyCNN(matched_filters),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(),
    )

    from tensorflow import TensorShape

    model._model.build(
        input_shape=[TensorShape([None, 64, 1]), TensorShape([None, 4])]
    )

    aggregator = declearn.aggregator.AveragingAggregator(steps_weighted=False)

    server_opt = declearn.optimizer.Optimizer(
        lrate=5e-4,
        w_decay=0.0,
        modules=[AdamModule(), ScaffoldServerModule()],
        regularizers=[],
    )

    client_opt = declearn.optimizer.Optimizer(
        lrate=1e-3,
        w_decay=0.0,
        regularizers=[],
        modules=[ScaffoldClientModule()],
    )

    random.seed(42)
    pop_dataset_client = random.randint(0, nb_clients - 1)
    data_folder = os.path.join(
        FILEDIR, "data/mit-bih-aami", f"client_{pop_dataset_client}"
    )

    X_train = load_data_array(os.path.join(data_folder, "train_data.npy"))     
    X_train = np.transpose(X_train, axes=(0,2,1))
    RR_train = load_data_array(os.path.join(data_folder, "train_arr.npy"))
    y_train = load_data_array(os.path.join(data_folder, "train_target.npy"))
    X_train_deriv = np.diff(X_train, prepend=X_train[..., :1], axis=-1)
    tf_dataset_train = tf.data.Dataset.from_tensor_slices(
        ((X_train_deriv, RR_train), y_train, weights[y_train])
    )

    pop_dataset = TensorflowDataset(tf_dataset_train)

    target_datasets = []
    for training_client in range(nb_clients):
        if training_client == pop_dataset_client:
            continue
        data_folder = os.path.join(
            FILEDIR, "data/mit-bih-aami", f"client_{training_client}"
        )

        X_train = load_data_array(os.path.join(data_folder, "train_data.npy"))
        X_train = np.transpose(X_train, axes=(0,2,1))
        RR_train = load_data_array(os.path.join(data_folder, "train_arr.npy"))
        y_train = load_data_array(
            os.path.join(data_folder, "train_target.npy")
        )
        X_train_deriv = np.diff(X_train, prepend=X_train[..., :1], axis=-1)

        tf_dataset_train = tf.data.Dataset.from_tensor_slices(
            ((X_train_deriv, RR_train), y_train, weights[y_train])
        )

        target_datasets.append(TensorflowDataset(tf_dataset_train))

    adversarial = RmiaControllerServer(
        pop_dataset=pop_dataset,
        target_data=target_datasets,
        target_model=model,
        online=False,
        n_target_samples=1000,
        n_population_samples=500,
        num_references_models=10,
        n_reference_batch_size=128,
        reference_train_epochs=20,
        seed=42,
        alpha=0.2,
    )

    optim = declearn.main.config.FLOptimConfig.from_params(
        aggregator=aggregator,
        server_opt=server_opt,
        client_opt=client_opt,
        adversarial=adversarial,
    )

    network = declearn.communication.build_server(
        protocol=protocol,
        host=host,
        port=port,
    )

    server = declearn.main.FederatedServer(
        model=model,
        netwk=network,
        optim=optim,
        metrics=metrics,
        checkpoint=checkpoint,
    )

    register = declearn.main.config.RegisterConfig(
        min_clients=nb_clients,
        max_clients=nb_clients,
    )

    training = declearn.main.config.TrainingConfig(
        batch_size=128,
        n_epoch=1,
        shuffle=True,
        drop_remainder=True,
    )

    evaluate = declearn.main.config.EvaluateConfig(
        batch_size=1024,
        shuffle=False,
        frequency=10,
    )

    run_config = declearn.main.config.FLRunConfig.from_params(
        rounds=1000,
        register=register,
        training=training,
        evaluate=evaluate,
    )
    server.run(run_config)


def main():
    fire.Fire(run_server)


if __name__ == "__main__":
    main()
