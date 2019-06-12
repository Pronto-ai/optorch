#!/usr/bin/env python3
import torch
import optorch

class RosenbrockCost(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x, y):
        return (-x + 1)**2 + 100*(y-x**2)**2

if __name__ == '__main__':
    x = torch.tensor(-1.2, dtype=torch.float64)
    y = torch.tensor(1., dtype=torch.float64)

    print(f'Initial value of x: {x.item()} y: {y.item()}')

    problem = optorch.Problem()
    problem.max_iterations = 200
    problem.add_residual(RosenbrockCost(), x, y)
    problem.solve(verbose=True)

    print(f'Final value of x: {x.item()} y: {y.item()}')
