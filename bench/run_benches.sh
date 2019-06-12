#!/bin/bash

VERTS="${VERTS:-250 500 1000}"
SEQ="${SEQ:-$(seq 1 8)}"

for vert in $VERTS; do
  echo "running for verts $vert"
  if [ -z $PLOTONLY ]; then
    ./g2o_pose_graph.py parking-garage.g2o $vert 50
    rm -f parking-garage.optorch_${vert}.data
    for i in $SEQ; do
      MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ./optorch_pose_graph.py parking-garage.g2o $vert $i
    done
  fi
  verts=$vert ./make_plot.plot
done
