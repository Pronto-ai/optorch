# OpTorch

Gradient descent not fast enough? Tired of managing memory and juggling template parameters to interface with your favorite nonlinear solver in C++?

OpTorch lets you write your cost functions as PyTorch modules and seamlessly optimize them in [ceres](http://ceres-solver.org), Google's industrial strength solver. We use OpTorch for automatic ground-truthing at [Pronto](https://pronto.ai/), but there may be bugs or poor performance for use cases we haven't considered &mdash; we want to make OpTorch the fastest and easiest to use nonlinear solver frontend, so Issues and PRs are welcome!

## Installation

```shell
pip install optorch
```

[API Docs](https://pronto-ai.github.io/optorch/)

## Examples

Optimizing single parameter, single residual:

```python
import torch
import optorch

class SimpleCost(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        return 10 - x

x = torch.tensor(0., dtype=torch.float64)

problem = optorch.Problem()
problem.add_residual(SimpleCost(), x)
problem.solve(verbose=True)

print(f'final x: {x.item()}')
```

```console
$ python simple.py
started optorch main
solving
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  5.000000e+01    0.00e+00    1.00e+01   0.00e+00   0.00e+00  1.00e+04        0    6.16e-02    6.16e-02
   1  4.999000e-07    5.00e+01    1.00e-03   1.00e+01   1.00e+00  3.00e+04        1    3.90e-04    6.20e-02
   2  5.554074e-16    5.00e-07    3.33e-08   1.00e-03   1.00e+00  9.00e+04        1    1.14e-04    6.22e-02
trust_region_minimizer.cc:692 Terminating: Parameter tolerance reached. Relative step_norm: 3.332852e-09 <= 1.000000e-08.
solved
final x: 9.99999996667111
```

Fitting a circle, snipped from [circle_fit.py](/examples/circle_fit.py):

```python
class DistanceFromCircleCost(torch.jit.ScriptModule):
    def __init__(self, xx, yy):
        super().__init__()
        # constant parameters
        self.xx = torch.nn.Parameter(xx)
        self.yy = torch.nn.Parameter(yy)

    @torch.jit.script_method
    def forward(self, x, y, m):
        # radius = m^2
        r = m * m
        # position of sample in circle's coordinate system
        xp = self.xx - x
        yp = self.yy - y
        # nicer, more convex loss compared to r - sqrt(xp^2 + yp^2)
        return r*r - xp*xp - yp*yp

pts = # generate noisy circle points
for xx, yy in pts:
    cost = DistanceFromCircleCost(torch.tensor([xx]), torch.tensor([yy]))
    problem.add_residual(cost, x, y, m)
problem.max_iterations = 200
problem.solve()
plt.scatter(pts[:,0], pts[:,1])
# plot circles at (true_x, true_y) and (x.item(), y.item())
```

![](/examples/results/circle_fit.png)

## Performance

Benchmarks were run for pose graph optimization on the MIT Parking Garage dataset.
Results below are for 250, 500, and 1000 vertices. Y axis is translational error.

250 (307 edges) | 500 (615 edges) | 1000 (2635 edges)
--- | --- | ---
![](/bench/plot250.png?raw=True) | ![](/bench/plot500.png?raw=True) | ![](/bench/plot1000.png?raw=True)

As you can see, OpTorch is quite a bit slower than g2opy, but consistently finds lower losses.
If you're just doing SLAM, you probably want g2opy, but if you need custom loss functions, OpTorch will
allow you to write them in Python.

## Licenses

The OpTorch codebase is released by Pronto AI under the MIT license.

The binary packages available on PyPI are statically linked to:
- [Ceres, which is under the BSD License](https://github.com/ceres-solver/ceres-solver/blob/master/LICENSE)
- [SuiteSparse, the components of which are licensed under BSD or LGPL](https://github.com/jluttine/suitesparse/blob/master/LICENSE.txt)
