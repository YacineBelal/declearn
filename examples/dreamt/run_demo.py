import os
from typing import Optional

import fire

from declearn.test_utils import make_importable
from declearn.utils import run_as_processes

with make_importable(os.path.dirname(__file__)):
    from prepare_data import prepare_dreamt
    from run_client import run_client
    from run_server import run_server


def run_demo(
    nb_clients: int =20, 
    seed: Optional[int] = None,
) -> None:
    data_folder = prepare_dreamt(nb_clients, seed=seed)
    server = (run_server, {"nb_clients": nb_clients})
    client_kwargs = {
        "data_folder" : data_folder,
        "verbose": False
    }
    clients = [
        (run_client, 
        (f"client_{idx}",), client_kwargs) 
        for idx in range(nb_clients)
    ] 
 
    success, outp = run_as_processes(server, *clients)
    if not success:
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                "\n".join(str(e) for e in outp if isinstance(e, RuntimeError))
            )



if __name__ == "__main__":
     fire.Fire(run_demo)