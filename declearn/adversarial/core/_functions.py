from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from declearn.dataset import Dataset, InMemoryDataset
from declearn.dataset.tensorflow import TensorflowDataset
from declearn.model.api import Model


def index_data_arrays(
    arr: Union[np.ndarray, Tuple[np.ndarray, ...]], idx: np.ndarray
):
    """Index a single array or a tuple of arrays (multi-input case) by row."""
    if isinstance(arr, tuple):
        return tuple(a[idx] for a in arr)
    return arr[idx]

def build_dataset_from_arrays(
    x: Union[np.ndarray, Tuple[np.ndarray, ...]],
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Dataset:
    """Wrap (x, y, w) arrays back into a declearn Dataset.

    Falls back to TensorflowDataset for the multi-input case, since
    InMemoryDataset does not currently support tuples of input arrays.
    """
    if isinstance(x, tuple):
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (x, y) if w is None else (x, y, w)
        )
        return TensorflowDataset(dataset=tf_dataset)
    return InMemoryDataset(x, y, w)


def extract_full_dataset_arrays(
    dataset: Dataset,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Dumps a Dataset object into numpy arrays by generating batch of the full dataset"""
    n_samples = dataset.get_data_specs().n_samples
    whole_dataset = dataset.generate_batches(batch_size=n_samples)
    features, target, weights = next(whole_dataset)
    return features, target, weights


def to_numpy(arr):
    """Convert a torch/tf tensor (or tuple of them) to numpy, no-op otherwise."""
    if isinstance(arr, tuple):
        return tuple(to_numpy(a) for a in arr)
    if hasattr(
        arr, "numpy"
    ):  # tf.Tensor and torch.Tensor both expose .numpy()
        return arr.numpy()
    return arr


def compute_reference_probs(
    model: Model,
    reference_weights: List[Any],
    batch: Tuple[Any, np.ndarray, Any],
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute per-sample true-label probabilities under a set of reference weights.

    For each set of weights in `reference_weights`, this loads them into
    `model`, runs a forward pass on `batch`, and records the probability
    assigned to each sample's ground-truth label.

    Parameters
    ----------
    model:
        A declearn `Model` instance used as a scratch container to
        evaluate each set of reference weights. Its weights are
        overwritten in place, once per entry of `reference_weights`; by
        the time this function returns, `model` holds whichever weights
        were evaluated last. Callers that need the model's prior weights
        preserved must save and restore them around this call.
    reference_weights:
        List of weight sets (e.g. as returned by `Model.get_weights()`),
        one per reference model, to evaluate on `batch`.
    batch:
        An `(inputs, labels, weights)` tuple, e.g. as yielded by
        `Dataset.generate_batches`. `labels` must hold ground-truth
        class indices (not one-hot vectors, not model predictions).
    eps:
        Lower bound probabilities are clipped to, so that a reference
        model assigning near-zero probability to a true label doesn't
        propagate `inf`/`nan` into downstream likelihood ratios.

    Returns
    -------
    probs:
        Array of shape `(len(reference_weights), n_samples)`, where
        `probs[i, j]` is the probability assigned by the model loaded
        with `reference_weights[i]` to sample `j`'s true label.
    """
    _, y_true, _ = batch
    n_samples = len(y_true)
    probs = np.zeros((len(reference_weights), n_samples))

    for i, weights in enumerate(reference_weights):
        model.set_weights(weights)
        _, class_probs, _ = model.compute_batch_predictions(batch)
        probs[i, :] = class_probs[np.arange(n_samples), y_true]

    return np.clip(probs, eps, 1.0)
