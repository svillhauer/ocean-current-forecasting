import math

import torch

from currentcast.metrics import LeadTimeMetrics, gaussian_crps, skill


def test_perfect_forecast():
    y = torch.randn(3, 4, 2, 5, 5)
    m = LeadTimeMetrics(4, torch.ones(5, 5, dtype=torch.bool))
    m.update(y, y, sigma=torch.full_like(y, 0.1))
    r = m.result()
    assert r["rmse_vector"] == [0.0] * 4
    assert r["coverage_1sigma"] == [1.0] * 4


def test_vector_rmse_and_land_mask():
    pred = torch.zeros(1, 1, 2, 2, 2)
    target = torch.zeros(1, 1, 2, 2, 2)
    target[0, 0, 0, 0, 0], target[0, 0, 1, 0, 0] = 3.0, 4.0      # ocean pixel: |error| = 5
    target[0, 0, :, 1, 1] = 100.0                                 # land pixel: must be ignored
    mask = torch.tensor([[True, True], [True, False]])
    m = LeadTimeMetrics(1, mask)
    m.update(pred, target)
    assert math.isclose(m.result()["rmse_vector"][0], math.sqrt(25 / 3))


def test_coverage_matches_gaussian_expectation():
    y = torch.randn(200, 1, 2, 16, 16)
    m = LeadTimeMetrics(1, torch.ones(16, 16, dtype=torch.bool))
    m.update(torch.zeros_like(y), y, sigma=torch.ones_like(y))
    r = m.result()
    assert abs(r["coverage_1sigma"][0] - 0.683) < 0.01
    assert abs(r["coverage_2sigma"][0] - 0.954) < 0.01


def test_crps_known_value_and_deterministic_limit():
    c = gaussian_crps(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))
    assert math.isclose(c.item(), (math.sqrt(2) - 1) / math.sqrt(math.pi), rel_tol=1e-5)
    c0 = gaussian_crps(torch.tensor(0.0), torch.tensor(1e-6), torch.tensor(0.7))
    assert math.isclose(c0.item(), 0.7, rel_tol=1e-4)


def test_skill_score():
    assert skill([0.5, 1.0], [1.0, 1.0]) == [0.5, 0.0]
