from __future__ import annotations

import numpy as np
import torch

from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from scripts.fastwam.audit_behavior1k_normalization import TaskMoments, _mixture


def test_task_moments_and_mixture_preserve_within_task_variance() -> None:
    moments = TaskMoments(2, np)
    reference_mean = np.zeros(23, dtype=np.float64)
    reference_std = np.ones(23, dtype=np.float64)
    first = np.stack(
        (np.zeros(23, dtype=np.float64), np.full(23, 2.0, dtype=np.float64))
    )
    second = np.stack(
        (np.full(23, 10.0, dtype=np.float64), np.full(23, 14.0, dtype=np.float64))
    )
    moments.update(
        0,
        first,
        reference_mean=reference_mean,
        reference_std=reference_std,
        np=np,
    )
    moments.update(
        1,
        second,
        reference_mean=reference_mean,
        reference_std=reference_std,
        np=np,
    )

    means, stds = moments.finish(np)
    np.testing.assert_allclose(means[:, 0], [1.0, 12.0])
    np.testing.assert_allclose(stds[:, 0], [1.0, 2.0])
    mixed_mean, mixed_std = _mixture(means, stds, [0.5, 0.5], np)
    expected_values = np.asarray([0.0, 2.0, 10.0, 14.0])
    np.testing.assert_allclose(mixed_mean, expected_values.mean())
    np.testing.assert_allclose(mixed_std, expected_values.std())


def test_task_moments_report_normalizer_clamp_outliers() -> None:
    moments = TaskMoments(1, np)
    values = np.zeros((3, 23), dtype=np.float64)
    values[:, 0] = [0.0, 4.0, 6.0]
    moments.update(
        0,
        values,
        reference_mean=np.zeros(23, dtype=np.float64),
        reference_std=np.ones(23, dtype=np.float64),
        np=np,
    )

    assert moments.over_three[0, 0] == 2
    assert moments.over_five[0, 0] == 1
    assert moments.over_five.sum() == 1


def test_minmax_inverse_restores_constant_physical_dimension() -> None:
    stats = {
        "min": torch.tensor([0.0, -1.0]),
        "max": torch.tensor([0.0, 1.0]),
        "mean": torch.tensor([0.0, 0.0]),
        "std": torch.tensor([0.0, 1.0]),
    }
    normalizer = SingleFieldLinearNormalizer(stats, mode="min/max")

    restored = normalizer.backward(torch.tensor([[0.75, 2.0]]))

    torch.testing.assert_close(restored, torch.tensor([[0.0, 1.0]]))
