import torch

from currentcast.forecast import persistence, rollout


class AddOne(torch.nn.Module):
    """Predicts last frame + 1, and records the windows it was given."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x[:, :, 0, 0, 0].clone())
        mu = x[:, -1] + 1
        return mu, torch.zeros_like(mu)


def test_rollout_feeds_predictions_back_and_slides_the_window():
    x = torch.arange(3.0).view(1, 3, 1, 1, 1).expand(1, 3, 2, 2, 2).clone()  # frames 0, 1, 2
    model = AddOne()
    mu, logvar = rollout(model, x, steps=3)
    assert mu.shape == (1, 3, 2, 2, 2)
    assert mu[0, :, 0, 0, 0].tolist() == [3, 4, 5]
    assert [w[0].tolist() for w in model.seen] == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        mu = x[:, -1] * self.w
        return mu, torch.zeros_like(mu)


def test_rollout_has_no_grad_by_default_but_can_train():
    model, x = Scale(), torch.randn(1, 3, 2, 4, 4)
    assert not rollout(model, x, 2)[0].requires_grad
    assert rollout(model, x, 2, grad=True)[0].requires_grad


def test_persistence_repeats_last_frame():
    x = torch.randn(2, 5, 2, 3, 3)
    p = persistence(x, 4)
    assert p.shape == (2, 4, 2, 3, 3)
    assert torch.equal(p[:, 3], x[:, -1])
