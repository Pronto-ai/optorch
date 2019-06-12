#!/usr/bin/env python3

import re
import pprint
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any, Tuple
from collections import defaultdict

import torch
import optorch

# each nist problem has 2 starting positions
N_TRIES = 2

@dataclass
class NistProblem:
    name: str
    difficulty: str
    loss: float
    params: Any
    vals: Any

    def n_params(self):
        return len(self.params)
    def n_tries(self):
        return len(self.params[0]['initial_positions'])

    def print(self):
        print(f'=== name:       {self.name}')
        print(f'=== difficulty: {self.difficulty}')
        print(f'=== num params: {self.n_params()}')
        print(f'=== loss:       {self.loss}')
        print('=== params:')
        pprint.pprint(self.params)
        print('=== data:')
        pprint.pprint(self.vals)

def parse_nist_problem(f):
    params = []
    vals = []
    for line in f:
        line = line.strip()
        m = re.match('Dataset Name:\s+(\w+)', line)
        if m:
            name = m.group(1)
        m = re.match('(\w+) Level of Difficulty', line)
        if m:
            difficulty = m.group(1)
        m = re.match('(\d+) Parameters', line)
        if m:
            n_params = int(m.group(1))
        if re.search('Starting values', line):
            next(f); next(f)
            for i in range(n_params):
                line = next(f).strip().split()
                param = {
                    'name': line[0],
                    'initial_positions': [],
                }
                for j in range(N_TRIES):
                    param['initial_positions'].append(float(line[2+j]))
                param['final_position'] = float(line[2+N_TRIES])
                params.append(param)
            continue
        m = re.match('Residual Sum of Squares:\s+([^\s]+)', line)
        if m:
            loss = float(m.group(1))
        if re.search('^Data:\s+y\s+x$', line):
            for line in f:
                line = line.strip().split()
                vals.append((float(line[0]), float(line[1])))
            break

    return NistProblem(name, difficulty, loss, params, vals)

class NistCost(torch.jit.ScriptModule):
    def __init__(self, cost_fn, xs, ys):
        super().__init__()
        self.cost_fn = cost_fn
        self.xs = torch.nn.Parameter(xs)
        self.ys = torch.nn.Parameter(ys)

    @torch.jit.script_method
    def forward(self, params):
        residuals = []
        for i in range(self.xs.shape[0]):
            y_hat = self.cost_fn(self.xs[i], params)
            residuals.append(self.ys[i] - y_hat)
        return torch.stack(residuals)

def n_good(nist_prob, params):
    n = 0
    for i, p in enumerate(params):
        nist_value = nist_prob.params[i]['final_position']
        matching_digits = -torch.log10(abs(nist_value - p))
        if matching_digits > 4:
            n += 1
    return n

if __name__ == '__main__':
    possible = defaultdict(lambda: 0)
    successes = defaultdict(lambda: 0)

    def add_nist_problem(filename, cost_fn):
        with open(Path('dataset_nist') / (filename+'.dat')) as f:
            nist_prob = parse_nist_problem(f)
            nist_prob.print()
            fake_x = torch.rand(1, dtype=torch.float64)
            fake_params = torch.rand(nist_prob.n_params(), dtype=torch.float64)
            traced_cost = torch.jit.trace(cost_fn, (fake_x, fake_params))
            xs = torch.tensor([v[1] for v in nist_prob.vals], dtype=torch.float64)
            ys = torch.tensor([v[0] for v in nist_prob.vals], dtype=torch.float64)
            cost = NistCost(traced_cost, xs, ys)
            for t in range(nist_prob.n_tries()):
                possible[nist_prob.difficulty] += nist_prob.n_params()
                problem = optorch.Problem()
                problem.solver = 'dense_qr'
                problem.num_threads = 12
                problem.max_iterations = 200
                params = torch.tensor([p['initial_positions'][t]
                                       for p in nist_prob.params], dtype=torch.float64)
                problem.add_residual(cost, params)
                problem.solve(verbose=True)
                successes[nist_prob.difficulty] += n_good(nist_prob, params)

    add_nist_problem('Misra1a', lambda x, b: b[0] * (1 - torch.exp(-b[1] * x)))

    chwirut = lambda x, b: torch.exp(-b[0]*x) / (b[1]+b[2]*x)
    add_nist_problem('Chwirut1', chwirut)
    add_nist_problem('Chwirut2', chwirut)

    lanczos = lambda x, b: b[0]*torch.exp(-b[1]*x) + b[2]*torch.exp(-b[3]*x) + b[4]*torch.exp(-b[5]*x)
    add_nist_problem('Lanczos3', lanczos)

    def gauss(x, b):
        y =  b[0] * torch.exp(-b[1]*x)
        y += b[2] * torch.exp(-torch.pow((x - b[3])/b[4], 2))
        y += b[5] * torch.exp(-torch.pow((x - b[6])/b[7], 2))
        return y
    add_nist_problem('Gauss1', gauss)
    add_nist_problem('Gauss2', gauss)

    for difficulty in possible:
        print(f'{difficulty}: {successes[difficulty]}/{possible[difficulty]}')
