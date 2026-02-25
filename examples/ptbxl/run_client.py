"""Script to run a federated client on the ptbxl example."""

import datetime 
import logging 
import os 


import fire 

import declearn 


FILEDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CERT = os.path.join(FILEDIR, "ca-cert.pem")



def run_client(
        client_name : str, 
        data_folder: str,
        ca_cert: str= DEFAULT_CERT,
        protocol: str="websockets",
        serv_uri: str="wss://localhost:8765",
        verbose: bool= True, 
) -> None:

    declearn.utils.set_device_policy(gpu=False)

    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"results_{stamp}", client_name)
    logger = declearn.utils.get_logger(
        name=client_name,
        fpath=os.path.join(checkpoint,"logs.txt")
    ) 

    if not verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(declearn.utils.LOGGING_LEVEL_MAJOR)

    data_folder = os.path.join(FILEDIR, data_folder, client_name)
    