import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.adversarial.api import AdversarialControllerServer
from declearn.adversarial.core import (
    build_dataset_from_arrays,
    compute_reference_probs,
    extract_full_dataset_arrays,
    index_data_arrays,
    to_numpy,
)
from declearn.dataset import Dataset, InMemoryDataset
from declearn.dataset.utils import load_data_array
from declearn.metrics import BinaryRocAUC
from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.utils.serialize import json_load


class RmiaControllerServer(AdversarialControllerServer):
    attack = "rmia"

    def __init__(
        self,
        pop_dataset: Union[str, Dataset],
        target_data: Union[List[str], Dataset],
        target_model: Union[Model, str, Dict[str, Any]],
        gamma: float = 1.0,
        alpha: Optional[float] = None,
        online: bool = False,
        n_target_samples: int = 10,
        n_population_samples: int = 5,
        n_reference_batch_size: int = 128,
        num_references_models: int = 5,
        reference_train_epochs: int = 10,
        seed: Optional[int] = None,
    ):
        """

        Parameters
        ----------
        pop_dataset:
            dataset available to the adversary to sample points for the purpose of training reference models
        target_data:
            data used in training by clients targeted by the attack. Has to be disjoint from
            pop_dataset samples
        target_model:
            model specification
        gamma:
            lower decision boundary for the pairwise likelihood ratio P(theta|X) / P(theta|Z) where X is a targeted sample
        alpha:
            if provided, used to unbias P(X) through a linear combination of P(X)_{out}. Not used in the online attack
        n_target_samples:
            number of target samples to attack
        n_population_samples:
            number of samples to use to approximate P(Z), probability of observing any point z
        num_references_models:
            number of reference models to train
        reference_train_epochs:
            number of epochs to train reference models
        """
        self.pop_dataset = self._parse_pop_dataset(pop_dataset)
        self.target_model = self._parse_model(target_model)
        self.target_data = self._parse_target_data(target_data)
        self.n_population_samples = n_population_samples
        self.num_references_models = num_references_models
        self.n_target_samples = n_target_samples
        self.n_reference_batch_size = n_reference_batch_size
        self.reference_train_epochs = reference_train_epochs
        self.gamma = gamma
        self.alpha = alpha
        self._online = online
        self._rng = np.random.default_rng(seed)
        self._initialized = False

    @staticmethod
    def _parse_pop_dataset(pop_dataset: Union[str, Dataset]) -> Dataset:

        if isinstance(pop_dataset, Dataset):
            return pop_dataset
        if isinstance(pop_dataset, str):
            pop_dataset = load_data_array(pop_dataset)
            return InMemoryDataset(pop_dataset)

        raise TypeError(
            "'pop_dataset' should be of type str indicating a file path, "
            f" or a Dataset object. Not '{type(pop_dataset)}'"
        )

    @staticmethod
    def _parse_model(
        target_model: Union[Model, str, Dict[str, Any]],
    ) -> Model:
        if isinstance(target_model, Model):
            return target_model

        if isinstance(target_model, str):  # Path to model config JSON file.
            try:
                target_model = json_load(target_model)
            except Exception as exc:
                raise TypeError(
                    "JSON-deserialization of 'model' file path failed."
                ) from exc
        if isinstance(target_model, dict):  # Model config dictionary.
            try:
                return Model.from_config(target_model, allow_bin=False)
            except Exception as exc:
                raise TypeError(
                    "Model construction from 'model' configuration dictionary "
                    "failed."
                ) from exc

        raise TypeError(
            "'model' should be a declearn Model, optionally in serialized "
            f"form, not '{type(target_model)}'"
        )

    @staticmethod
    def _parse_target_data(
        target_data: Union[List[str], Dataset],
    ) -> Dataset:
        if isinstance(target_data, Dataset):
            return target_data
        if isinstance(target_data, list):
            if all(isinstance(f_path, str) for f_path in target_data):
                target_data_arrs = []
                for f_path in target_data:
                    target_data_arrs.append(load_data_array(f_path))

                return InMemoryDataset(
                    np.concatenate(target_data_arrs, axis=0)
                )

            elif all(isinstance(item, Dataset) for item in target_data):
                target_data_x = []
                target_data_y = []
                for item in target_data:
                    x, y, _ = extract_full_dataset_arrays(item)
                    target_data_x.append(x)
                    target_data_y.append(y)

                # multi inputs case
                if isinstance(target_data_x[0], tuple):
                    n_inputs = len(target_data_x[0])
                    target_data_x = tuple(
                        np.concatenate([x[i] for x in target_data_x])
                        for i in range(n_inputs)
                    )

                else:
                    target_data_x = np.concatenate(target_data_x, axis=0)

                target_data_y = np.concatenate(target_data_y, axis=0)
                # reverting backend to tensorflow to build a Dataset object, since it \
                # it handles multiple inputs and as opposed to torch wrapper, does not \
                # require a torch utils.data.Dataset class.
                # TODO: make inmemory handle multiple inputs?

                return build_dataset_from_arrays(target_data_x, target_data_y)

        raise TypeError(
            "'target_samples' must be a Dataset, a list of file paths, "
            "or a list of Dataset instances."
        )

    def setup_adversarial(
        self,
        target_optim: Optimizer,
    ):
        # TODO: could we get model here instead of feeding them
        self.holdout_nonmembers = self._sample_heldout_nonmembers()

        self.population_batch = self._sample_population_batch()
        self.reference_train_batches_out = (
            self._sample_reference_train_batches()
        )
        self.target_samples = self._sample_target_batch()
        self.out_reference_model_weights = self._train_out_reference_models(
            target_optim
        )
        self.population_probs_under_out_models = compute_reference_probs(
            self.target_model,
            self.out_reference_model_weights,
            self.population_batch,
        ).mean(axis=0)

        self.target_probs_under_out_models = compute_reference_probs(
            self.target_model,
            self.out_reference_model_weights,
            self.target_samples,
        ).mean(axis=0)

        self.nonmember_probs_under_out_models = compute_reference_probs(
            self.target_model,
            self.out_reference_model_weights,
            self.holdout_nonmembers,
        ).mean(axis=0)

        if self.alpha is not None:
            unbiasing = np.vectorize(
                lambda x: 0.5 * ((1 + self.alpha) * x + (1 - self.alpha))
            )
            self.population_probs_under_out_models = unbiasing(
                self.population_probs_under_out_models
            )
            self.target_probs_under_out_models = unbiasing(
                self.target_probs_under_out_models
            )
            self.nonmember_probs_under_out_models = unbiasing(
                self.nonmember_probs_under_out_models
            )

        self._initialized = True

    def run_adversarial_round(
        self,
        round_model: Model,
        target_optim: Optional[Optimizer] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        if not self._initialized:
            raise RuntimeError(
                "RmiaControllerServer has not been initialized. "
                "Call setup_adversarial() first."
            )
        elif self._online and target_optim is None:
            raise ValueError(
                f"Optimizer needs to be fed to {inspect.stack()[0][3]}"
                "when Rmia is in online mode."
            )

        population_probs_under_round_model = compute_reference_probs(
            self.target_model,
            [round_model.get_weights()],
            self.population_batch,
        ).squeeze()

        # if self._online:
        #     population_probs_under_in_models = []
        #     inputs, y, s_weights = self.target_samples
        #     for i in range(self.n_target_samples):
        #         reference_train_batches_in = (
        #             self._sample_reference_train_batches()
        #         )
        #         in_reference_model_weights = self._train_in_reference_models(
        #             target_optim, reference_train_batches_in
        #         )
        #         for i, weights in enumerate(in_reference_model_weights):
        #             # to make reference model trained on targeted samples

        #             self.target_model.set_weights(weights)

        #             print("here")
        #             print(f"{len(inputs)}")
        #             print(f"{len(inputs[0][i])}")
        #             print(f"{type(y[i])}")
        #             # print(f"{len(s_weights)}")

        #             sys.stdout.flush()

        #             target_optim.run_train_step(
        #                 self.target_model,
        #                 (
        #                     index_data_arrays(inputs, np.array([i])),
        #                     y[i],
        #                     None,
        #                 ),
        #             )
        #             print("here2")
        #             sys.stdout.flush()

        #             in_reference_model_weights[i] = (
        #                 self.target_model.get_weights()
        #             )

        #         population_probs_under_in_models.append(
        #             compute_reference_probs(
        #                 self.target_model,
        #                 in_reference_model_weights,
        #                 reference_train_batches_in,
        #             )
        #         )
        #     print(np.array(population_probs_under_in_models).shape)
        #     sys.stdout.flush()
        #     exit(1)

        likelihood_ratio_z = (
            population_probs_under_round_model
            / self.population_probs_under_out_models
        )

        target_probs_under_round_model = compute_reference_probs(
            self.target_model,
            [round_model.get_weights()],
            self.target_samples,
        ).squeeze()
        likelihood_ratio_x = (
            target_probs_under_round_model / self.target_probs_under_out_models
        )

        nonmember_probs_under_round_model = compute_reference_probs(
            self.target_model,
            [round_model.get_weights()],
            self.holdout_nonmembers,
        ).squeeze()

        nonmember_likelihood_ratio_x = (
            nonmember_probs_under_round_model
            / self.nonmember_probs_under_out_models
        )

        member_scores = (
            (
                likelihood_ratio_x[:, None]
                / np.maximum(likelihood_ratio_z, 1e-6)
            )
            > self.gamma
        ).mean(axis=1)

        nonmember_scores = (
            (
                nonmember_likelihood_ratio_x[:, None]
                / np.maximum(likelihood_ratio_z, 1e-6)
            )
            > self.gamma
        ).mean(axis=1)

        result = self._compute_roc(member_scores, nonmember_scores)
        fpr = result["fpr"]
        tpr = result["tpr"]
        thresholds = result["thresh"]

        nonzero = fpr[fpr > 0]
        smallest_fpr = nonzero.min() if len(nonzero) else fpr.min()
        idx = np.where(fpr == smallest_fpr)[0][0]

        result = {
            "member_scores": member_scores,
            "nonmember_scores": nonmember_scores,
            "average_membership_scores": member_scores.mean(),
            "fpr": fpr,
            "tpr": tpr,
            f"tpr@{fpr[idx]}": tpr[idx],
            "thresholds": thresholds,
            "auc": result["roc_auc"],  # np.trapezoid(tpr, fpr),
        }

        return result

    # @staticmethod
    # def _compute_roc(member_scores, nonmember_scores):
    #     from sklearn.metrics import roc_curve

    #     y_true = np.concatenate(
    #         [np.ones_like(member_scores), np.zeros_like(nonmember_scores)]
    #     )
    #     y_score = np.concatenate([member_scores, nonmember_scores])
    #     return roc_curve(y_true, y_score)

    @staticmethod
    def _compute_roc(member_scores, nonmember_scores):
        binary_rocauc_metric = BinaryRocAUC()
        y_true = np.concatenate(
            [np.ones_like(member_scores), np.zeros_like(nonmember_scores)]
        )
        y_score = np.concatenate([member_scores, nonmember_scores])
        binary_rocauc_metric.update(y_pred=y_score, y_true=y_true)
        return binary_rocauc_metric.get_result()

    def _sample_heldout_nonmembers(
        self,
    ) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
        x, y, w = extract_full_dataset_arrays(self.pop_dataset)
        x = to_numpy(x)
        y = to_numpy(y)
        w = to_numpy(w) if w is not None else None
        n_samples = len(y)

        if self.n_target_samples >= n_samples:
            raise ValueError(
                f"Cannot hold out {self.n_target_samples} non-member samples "
                f"from a population of only {n_samples}."
            )

        holdout_idx = self._rng.choice(
            n_samples, size=self.n_target_samples, replace=False
        )
        remainder_idx = np.setdiff1d(np.arange(n_samples), holdout_idx)
        holdout_batch = (
            index_data_arrays(x, holdout_idx),
            y[holdout_idx],
            w[holdout_idx] if w is not None else None,
        )

        remainder_x = index_data_arrays(x, remainder_idx)
        remainder_y = y[remainder_idx]
        remainder_w = w[remainder_idx] if w is not None else None

        self.pop_dataset = build_dataset_from_arrays(
            remainder_x, remainder_y, remainder_w
        )

        return holdout_batch

    def _sample_population_batch(self):
        return next(
            self.pop_dataset.generate_batches(
                batch_size=self.n_population_samples, shuffle=True
            )
        )

    def _sample_reference_train_batches(self):

        return [
            next(
                self.pop_dataset.generate_batches(
                    batch_size=self.n_reference_batch_size, shuffle=True
                )
            )
            for _ in range(self.num_references_models)
        ]

    def _sample_target_batch(self):
        # TODO: should it be the same target sample or should it change overtime?
        return next(
            self.target_data.generate_batches(
                batch_size=self.n_target_samples, shuffle=True
            )
        )

    def _train_in_reference_models(
        self,
        target_optim: Optimizer,
        reference_train_batches_in: Tuple[Any, np.ndarray, Any],
    ):

        in_reference_model_weights = []
        for dataset in reference_train_batches_in:
            in_reference_model_weights.append(
                self._train_reference_weights(target_optim, dataset)
            )

        return in_reference_model_weights

    def _train_out_reference_models(self, target_optim: Optimizer):

        out_reference_model_weights = []
        for dataset in self.reference_train_batches_out:
            out_reference_model_weights.append(
                self._train_reference_weights(target_optim, dataset)
            )

        return out_reference_model_weights

    def _train_reference_weights(
        self,
        optim: Optimizer,
        reference_batch: Tuple[Any, np.ndarray, Any],
    ):
        for _ in range(self.reference_train_epochs):
            optim.run_train_step(self.target_model, reference_batch)
        return self.target_model.get_weights()
