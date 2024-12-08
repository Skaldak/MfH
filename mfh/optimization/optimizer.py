import torch
from torch import nn

from mfh.optimization.loss import PixelLoss
from submodules.metric_depth.zoedepth.trainers.loss import SILogLoss


def linear_regression(x, y, mask=None):
    # x.shape = 1, N
    # y.shape = 1, N
    if mask is not None:
        x, y = torch.masked_select(x, mask).reshape(1, -1), torch.masked_select(y, mask).reshape(1, -1)
    x_1 = torch.cat([x, torch.ones_like(x)])
    w = y @ torch.linalg.pinv(x_1)
    return w.squeeze(0)


class AffineOptimizer(object):
    def __init__(
        self,
        lr=1.0,
        total_iter=50,
        max_iter=20,
        line_search_fn="strong_wolfe",
        loss_fn="silog",
        device="cuda",
        scale=True,
        offset=True,
        inverse=False,
        inverse_gt=True,
    ):
        self.total_iter = total_iter
        self.inverse = inverse
        self.inverse_gt = inverse_gt
        self.scale = nn.Parameter(torch.ones(1, device=device), requires_grad=scale > 0)
        self.offset = nn.Parameter(torch.zeros(1, device=device), requires_grad=offset > 0)
        if loss_fn == "silog":
            self.loss = SILogLoss()
        else:
            self.loss = PixelLoss(mode=loss_fn)

        parameters = []
        if scale:
            parameters.append(self.scale)
        if offset:
            parameters.append(self.offset)
        self.optimizer = torch.optim.LBFGS(parameters, max_iter=max_iter, lr=lr, line_search_fn=line_search_fn)

    def forward(self, depth):
        if self.inverse:
            return (self.scale / depth.clip(min=1e-12) + self.offset).clip(min=1e-12)
        else:
            return (self.scale * depth + self.offset).clip(min=1e-12)

    @torch.enable_grad()
    def optimize(self, disp, target_depth, target_mask=None, verbose=False):
        B, _, H, W = disp.shape
        curr_loss = torch.inf
        target = 1 / target_depth.clip(min=1e-12) if self.inverse_gt else target_depth

        def closure():
            nonlocal curr_loss
            self.optimizer.zero_grad()

            pred = self.forward(disp)
            loss = self.loss(pred, target, mask=target_mask)
            loss.backward()
            curr_loss = loss.item()

            return loss

        for i in range(self.total_iter):
            self.optimizer.step(closure)
            if verbose:
                print(curr_loss)
