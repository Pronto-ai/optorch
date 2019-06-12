#!/usr/bin/env python3
import torch
import optorch

class SimpleCost(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        return 10 - x

if __name__ == '__main__':
    x = torch.tensor(0., dtype=torch.float64)
    problem = optorch.Problem()
    problem.add_residual(SimpleCost(), x)
    problem.solve(verbose=True)
    print(f'final x: {x}')
