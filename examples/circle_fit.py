#!/usr/bin/env python3
"""
Based on:
https://github.com/ceres-solver/ceres-solver/blob/master/examples/circle_fit.cc
Much of this is a straightforward translation, so the copyright belongs to Google under 3-clause BSD
"""
import torch
import optorch
import numpy as np

class DistanceFromCircleCost(torch.jit.ScriptModule):
    def __init__(self, xx, yy):
        super().__init__()
        self.xx = torch.nn.Parameter(xx)
        self.yy = torch.nn.Parameter(yy)

    @torch.jit.script_method
    def forward(self, x, y, m):
        # radius = m^2
        r = m * m

        # position of sample in circle's coordinate system
        xp = self.xx - x
        yp = self.yy - y

        # see note in ceres example for explanation of choice of cost
        return r*r - xp*xp - yp*yp

if __name__ == '__main__':
    # generate noisy circle points
    np.random.seed(1337)
    true_x = 20.
    true_y = -300.
    true_r = 45.
    pts = []
    for _ in range(50):
        theta = np.random.uniform(0., np.pi*2)
        xx = true_x + np.cos(theta) * true_r + np.random.normal(scale=5.)
        yy = true_y + np.sin(theta) * true_r + np.random.normal(scale=5.)
        pts.append((xx, yy))
    pts = np.array(pts)

    # initial estimates
    x = torch.tensor(0.)
    y = torch.tensor(0.)
    # radius is m^2 so it can't be negative
    m = torch.tensor(1.)

    problem = optorch.Problem()
    for xx, yy in pts:
        cost = DistanceFromCircleCost(torch.tensor([xx]), torch.tensor([yy]))
        problem.add_residual(cost, x, y, m)
    problem.max_iterations = 200
    problem.solve(verbose=True, abort=False)

    print('final:', x.item(), y.item(), m.item()*m.item())

    import matplotlib.pyplot as plt
    plt.scatter(pts[:,0], pts[:,1])
    c0 = plt.Circle((true_x, true_y), true_r, edgecolor='r', fill=False)
    plt.gca().add_artist(c0)
    c1 = plt.Circle((x.item(), y.item()), (m*m).item(), edgecolor='g', ls='-', fill=False)
    plt.gca().add_artist(c1)
    # plt.show()
    plt.savefig('results/circle_fit.png')
