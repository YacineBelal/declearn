import datetime
import logging
import os

import fire
import torch
from model import tinyCNN

import declearn
import declearn.model.torch
from declearn.dataset.utils import load_data_array
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
    port: int = 8765,
) -> None: 
    
    declearn.utils.set_device_policy(gpu=True)
    metrics = declearn.metrics.MetricSet(
        [
            declearn.metrics.MulticlassAccuracyPrecisionRecall(
                labels=range(3)  # 0:'N', 1:'S', 2:'V'
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

    weights = torch.from_numpy(weights)
    model = declearn.model.torch.TorchModel(
        model=tinyCNN(
            matched_filters=matched_filters, trainable_conv=True, seed=42
        ),
        loss=torch.nn.CrossEntropyLoss(),
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


    optim = declearn.main.config.FLOptimConfig.from_params(
        aggregator=aggregator,
        server_opt=server_opt,
        client_opt=client_opt,
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
    )

    evaluate = declearn.main.config.EvaluateConfig(
        batch_size=1024,
        shuffle=False,
        frequency=10,
    )

    run_config = declearn.main.config.FLRunConfig.from_params(
        rounds=2000,
        register=register,
        training=training,
        evaluate=evaluate,
    )
    server.run(run_config)    


    
def main():
    fire.Fire(run_server)


if __name__ == "__main__":
    main()






