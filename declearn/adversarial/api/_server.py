
import abc
from typing import ClassVar, Dict, Optional, Union

import numpy as np

from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.utils import create_types_registry


@create_types_registry(name="AdversarialControllerServer")
class AdversarialControllerServer(metaclass=abc.ABCMeta):

    attack: ClassVar[str] 
    
   # TODO: perhaps an AdversarialConfig object is in order here?
   # or perhaps just an FLRunConfig?
   # should it be abstract tho?
   # do these methods need to be async?


    def __init__(self) -> None:
        pass 

    @abc.abstractmethod 
    def setup_adversarial(
        self,
        target_optim : Optimizer,
        ):
        pass 


    @abc.abstractmethod
    def run_adversarial_round(
        self,
        round_model: Model,
        target_optim: Optional[Optimizer] = None,  
    ) -> Dict[str, Union[float, np.ndarray]]:
        pass 

    