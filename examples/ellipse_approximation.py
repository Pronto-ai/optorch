#!/usr/bin/env python3
"""
Based on:
https://github.com/ceres-solver/ceres-solver/blob/master/examples/ellipse_approximation.cc
Much of this is a straightforward translation, so the copyright belongs to Google under 3-clause BSD
Go read it for more in-depth comments!

Also, this doesn't work yet... the gradient sits at zero :((
"""
import torch
import optorch
import numpy as np

from optorch.debug import make_dot, annotate

def gen_Y():
    np.random.seed(1337)
    t = np.linspace(0., 2.*np.pi, 212, endpoint=False)
    t += 2. * np.pi * .01 * np.random.randn(t.size)
    theta = np.radians(15)
    a, b = np.cos(theta), np.sin(theta)
    R = np.array([[a, -b],
                  [b, a]])
    Y = np.dot(np.c_[4. * np.cos(t), np.sin(t)], R.T)
    return Y

class PointToLineSegmentContourCost(torch.jit.ScriptModule):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.nn.Parameter(x.double())
        self.y = torch.nn.Parameter(y.double())

    @torch.jit.script_method
    def forward(self, X_, t_):
        X = X_.view(-1, 20)
        t = t_.view(-1, 1)

        self_x = self.x.view(-1, 1).repeat(X.shape[0], 1)
        self_y = self.y.view(-1, 1).repeat(X.shape[0], 1)

        int_part = int(t[0][0].detach())
        frac_part = (t - int_part)[:,0]

        i0 = int_part % 10
        i1 = (int_part + 1) % 10

        p0 = X[:,i0*2:(i0+1)*2]
        p1 = X[:,i1*2:(i1+1)*2]

        x_loss = self_x[:,0] - (1.-frac_part)*p0[:,0] + frac_part*p1[:,0]
        y_loss = self_y[:,0] - (1.-frac_part)*p0[:,1] + frac_part*p1[:,1]

        """
        annotate(X_, 'X_')
        annotate(t_, 't_')
        annotate(X, 'X')
        annotate(t, 't')
        annotate(self_x, 'self_x')
        annotate(self_y, 'self_y')
        annotate(frac_part, 'frac_part')
        annotate(p0, 'p0')
        annotate(p1, 'p1')
        annotate(x_loss, 'x_loss')
        annotate(y_loss, 'y_loss')
        """

        return torch.stack([x_loss, y_loss], dim=1)

if __name__ == '__main__':
    n_segments = 10
    Y = gen_Y()

    # X is the matrix of control pts that make up our line segments
    # We initialize it to pts on the unit circle
    thetas = np.linspace(0., 2. * np.pi, n_segments, endpoint=False)
    X = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

    t = np.zeros((Y.shape[0],), dtype=np.float64)
    for i in range(Y.shape[0]):
        diff = Y[i] - X
        t[i] = (diff*diff).sum(axis=1).argmin()

    # torch versions to optimize
    X_t = torch.from_numpy(X).double().flatten()
    t_t = []
    for x in t:
        t_t.append(torch.tensor(x).double())

    """
    cost = PointToLineSegmentContourCost(torch.tensor([Y[0][0]]), torch.tensor([Y[0][1]]))
    t_t[0] += 0.5
    inp0 = X_t
    inp0.requires_grad_(True)
    inp1 = t_t[0]
    inp1.requires_grad_(True)
    c = cost(inp0, inp1)
    # c[0,0].backward()
    make_dot(c, {})
    import sys
    sys.exit(0)
    """

    problem = optorch.Problem()
    for i, (x, y) in enumerate(Y):
        cost = PointToLineSegmentContourCost(torch.tensor([x]), torch.tensor([y]))
        if i == 0:
            print(cost.graph)
        problem.add_residual(cost, X_t, t_t[i])
    problem.set_max_iterations(100)
    problem.solve(verbose=True)

    import matplotlib.pyplot as plt
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(Y[:,0], Y[:,1])
    plt.show()
