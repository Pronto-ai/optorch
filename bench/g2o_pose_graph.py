#!/usr/bin/env python3
"""
Usage:
  ./g2o_pose_graph.py <file> <verts> <iter> [--viewer]
"""
from docopt import docopt

import os
import g2o
import numpy as np
import time
from viewer import start_viewer

MAX_ITER = 10
VIEWER = True

def l2_loss(opt):
    total = 0.
    for e in opt.edges():
        e.compute_error()
        total += (e.error()[:3]**2).sum()
    return total

def put_edges(opt, q):
    edges = []
    for e in optimizer.edges():
        v = e.vertices()
        v0 = v[0].estimate().matrix()[:3,-1]
        v1 = v[1].estimate().matrix()[:3,-1]
        edges.append([v0, v1])
    edges = np.array(edges)
    q.put(edges)

if __name__ == '__main__':
    args = docopt(__doc__)

    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)

    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(True)
    optimizer.set_algorithm(solver)

    fn = args['<file>']

    optimizer.load(fn)
    print(f'initial num vertices: {len(optimizer.vertices())}')
    print(f'initial num edges: {len(optimizer.edges())}')

    vertex_ids = list(optimizer.vertices().keys())
    vertex_ids.sort()
    vertex_ids = set(vertex_ids[:int(args['<verts>'])])
    # print('vids:', vertex_ids)

    to_rm = []
    for e in optimizer.edges():
        vid0 = e.vertices()[0].id()
        vid1 = e.vertices()[1].id()
        if vid0 not in vertex_ids or vid1 not in vertex_ids:
            to_rm.append(e)
    for e in to_rm:
        optimizer.remove_edge(e)

    for vid, v in list(optimizer.vertices().items()):
        if vid not in vertex_ids:
            optimizer.remove_vertex(v)

    print('new num vertices:', len(optimizer.vertices()))
    print('new num edges:', len(optimizer.edges()))

    print(f'initial translation error: {l2_loss(optimizer)}')

    if args['--viewer']:
        q = start_viewer()

    optimizer.initialize_optimization()
    i = 0
    cum_time = 0.
    output_name = os.path.splitext(args['<file>'])[0] + '.g2opy_'+args['<verts>']+'.data'
    output = open(output_name, 'w')
    while True:
        t0 = time.time()
        optimizer.optimize(1)
        t1 = time.time()
        cum_time += (t1 - t0)
        if args['--viewer']:
            if i % 20 == 0:
                print('putting edges')
                put_edges(optimizer, q)
        loss = l2_loss(optimizer)
        output.write(str(cum_time) + ' ' + str(loss) + '\n')
        i += 1
        if i >= int(args['<iter>']):
            break
    output.close()

    if args['--viewer']:
        while True:
            time.sleep(1)
