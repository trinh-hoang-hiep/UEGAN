# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evidence lower bound (a.k.a. variational free energy) loss."""

from typing import Any

from torch import Tensor
from torch.nn.modules import loss


__all__ = [
    "ELBOLoss",
]


class ELBOLoss(loss._Loss):
    """Evidence lower bound (a.k.a. variational free energy) loss.

    References
    ----------
    .. [1] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra.
           "Weight Uncertainty in Neural Networks".
           In: ICML. 2015, pp. 1613-1622.
           URL: https://arxiv.org/abs/1505.05424

    Examples
    --------
    >>> import torch
    >>>
    >>> from bayestorch.losses import ELBOLoss
    >>>
    >>>
    >>> num_mc_samples = 5
    >>> num_train_batches = 100
    >>> batch_size = 256
    >>> log_likelihoods = torch.rand(num_mc_samples, batch_size)
    >>> kl_divs = torch.rand(num_mc_samples)
    >>> criterion = ELBOLoss(reduction="none")
    >>> loss = criterion(log_likelihoods, kl_divs, num_train_batches)

    """

    # override
    def __init__(self, reduction: "str" = "mean", **kwargs: "Any") -> "None":
        """Initialize the object.

        Parameters
        ----------
        reduction:
            The reduction to apply to the output. Must be one of the following:
            - "none": no reduction is applied;
            - "sum": the output is summed;
            - "mean": the output sum is divided by the number of elements.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if reduction not in ["none", "sum", "mean"]:
            raise ValueError(
                f"`reduction` ({reduction}) must be one of {['none', 'sum', 'mean']}"
            )
        super().__init__(reduction=reduction, **kwargs)

    # override
    def forward(
        self,
        log_likelihoods: "Tensor",
        kl_divs: "Tensor",
        kl_div_weight: "float",
    ) -> "Tensor":
        """Forward pass.

        In the following, let `B = {B_1, ..., B_k}` denote
        the batch shape and `N` the number of Monte Carlo
        samples.

        Parameters
        ----------
        log_likelihoods:
            The log likelihoods, shape: ``[N, *B]``.
        kl_divs:
            The Kullback-Leibler divergences, shape: ``[N]``.
        kl_div_weight:
            The Kullback-Leibler divergence weight (`1 / M` in the literature).
            According to reference [1], it counterbalances the bias deriving
            from summing the log likelihood over a single batch of data instead
            of over the entire dataset. It is often set equal to the number of
            training batches. More generally, it controls the strength of the
            regularization provided by the Kullback-Leibler divergence term and
            its optimal value depends on factors such as model and dataset size.

        Returns
        -------
            The loss, shape: ``[N]`` if `reduction` initialization
            argument is equal to "none", ``[1]`` otherwise.

        """
        if log_likelihoods.ndim > 1:
            # Sum along batch dimensions if any
            log_likelihoods = log_likelihoods.flatten(start_dim=1).sum(dim=-1)
        result = kl_div_weight * kl_divs - log_likelihoods
        if self.reduction == "none":
            return result
        elif self.reduction == "sum":
            return result.sum()
        elif self.reduction == "mean":
            return result.mean()
