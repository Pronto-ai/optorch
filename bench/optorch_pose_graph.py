#!/usr/bin/env python3
"""
Usage:
  ./optorch_pose_graph.py <file> <verts> <iter> [--viewer]
"""
from docopt import docopt

import os
import torch
import optorch
from tqdm import tqdm
from quaternion import qmul, qrot
import numpy as np
import time

from viewer import start_viewer

def xyzw_to_wxyz(lst):
    return [lst[3], lst[0], lst[1], lst[2]]

@torch.jit.script
def qlen(q_):
    q = q_.view(-1, 4)
    return (q**2).sum(dim=1).view(-1, 1)

@torch.jit.script
def qinv(q_):
    q = q_.view(-1, 4)
    conj = torch.stack((q[:,0], -q[:,1], -q[:,2], -q[:,3]), dim=1)
    return conj / qlen(q)

@torch.jit.script
def qdist(q1, q2):
    return torch.acos(2*torch.dot(q1, q2)**2 - 1)

class SE3Cost(torch.jit.ScriptModule):
    def __init__(self, t, r):
        super().__init__()
        self.t = torch.nn.Parameter(t)
        self.r = torch.nn.Parameter(r)

    @torch.jit.script_method
    def forward(self, t0_, r0_, t1_, r1_):
        t0 = t0_.view(-1, 3)
        r0 = r0_.view(-1, 4)
        t1 = t1_.view(-1, 3)
        r1 = r1_.view(-1, 4)

        r0_inv = qinv(r0)

        t_diff = qrot(r0_inv, (t1 - t0))
        r_diff = qmul(r0_inv.clone(), r1)

        self_t = self.t.repeat(t0.shape[0], 1)
        self_r = self.r.repeat(t0.shape[0], 1)

        t_loss = t_diff - self_t
        self_r_inv = qinv(self_r)
        r_loss = qmul(self_r_inv, r_diff)
        norm_r_loss = r_loss / qlen(r_loss)
        cut_norm_r_loss = norm_r_loss[:,1:]

        return torch.cat([t_loss, cut_norm_r_loss], dim=1)

def other_forward(t, r, t0_, r0_, t1_, r1_):
    t0 = t0_.view(-1, 3)
    r0 = r0_.view(-1, 4)
    t1 = t1_.view(-1, 3)
    r1 = r1_.view(-1, 4)

    r0_inv = qinv(r0)

    t_diff = qrot(r0_inv, (t1 - t0))
    r_diff = qmul(r0_inv.clone(), r1)

    self_t = t.repeat(t0.shape[0], 1)
    self_r = r.repeat(t0.shape[0], 1)

    t_loss = t_diff - self_t
    self_r_inv = qinv(self_r)
    r_loss = qmul(self_r_inv, r_diff)
    norm_r_loss = r_loss / qlen(r_loss)
    cut_norm_r_loss = norm_r_loss[:,1:]

    return torch.cat([t_loss, cut_norm_r_loss], dim=1)

def l2_loss(edges):
    total = 0.
    for v0, v1, t, r in tqdm(edges):
        if v0 not in vertices or v1 not in vertices:
            continue
        t0, r0 = vertices[v0]
        t1, r1 = vertices[v1]
        loss = other_forward(t, r, t0, r0, t1, r1)
        total += (loss[0,:3]**2).sum().detach().numpy()
    return total

if __name__ == '__main__':
    args = docopt(__doc__)
    vertices = dict()
    edges = []

    fn = args['<file>']
    i = 0
    with open(fn, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            if line[0] == 'VERTEX_SE3:QUAT':
                if i < int(args['<verts>']):
                    pose = [float(x) for x in line[2:]]
                    t = torch.tensor(pose[:3])
                    r = torch.tensor(xyzw_to_wxyz(pose[3:]))
                    vertices[line[1]] = (t, r)
                i += 1
            elif line[0] == 'EDGE_SE3:QUAT':
                v0, v1 = line[1], line[2]
                edge = [float(x) for x in line[3:10]]
                t = torch.tensor(edge[:3])
                r = torch.tensor(xyzw_to_wxyz(edge[3:]))
                edges.append((v0, v1, t, r))
            else:
                raise ValueError('unknown tag', line[0])

    print(f'{len(vertices)} vertices')

    if args['--viewer']:
        q = start_viewer()
        init_edges = []

    print('building problem...')
    problem = optorch.Problem()
    for v0, v1, t, r in tqdm(edges):
        if v0 not in vertices or v1 not in vertices:
            continue
        t0, r0 = vertices[v0]
        t1, r1 = vertices[v1]
        problem.add_residual(SE3Cost(t, r), t0, r0, t1, r1)
        if args['--viewer']:
            init_edges.append([t0.numpy(), t1.numpy()])

    for t0, r0 in vertices.values():
        problem.set_local_parameterization(r0, 'quat')

    if args['--viewer']:
        init_edges = np.array(init_edges)
        q.put(init_edges)
    print('built problem')

    print(f'initial translation error: {l2_loss(edges)}')

    problem.set_num_threads(12)
    problem.set_max_iterations(int(args['<iter>']))
    t0 = time.time()
    problem.solve(verbose=True, abort=False)
    t1 = time.time()
    print('outside loop time:', (t1-t0))

    if args['--viewer']:
        init_edges = []
        for v0, v1, t, r in edges:
            if v0 not in vertices or v1 not in vertices:
                continue
            t0, r0 = vertices[v0]
            t1, r1 = vertices[v1]
            init_edges.append([t0.numpy(), t1.numpy()])
        init_edges = np.array(init_edges)
        q.put(init_edges)

    final_err = l2_loss(edges)
    print(f'final translation error: {final_err}')
    output_name = os.path.splitext(args['<file>'])[0] + '.optorch_'+args['<verts>']+'.data'
    with open(output_name, 'a') as f:
        f.write(str(t1-t0)+' '+str(final_err)+'\n')
