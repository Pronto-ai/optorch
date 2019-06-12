#!/usr/bin/env gnuplot
reset

nverts=system("echo $verts")

set terminal png size 400,400
set output 'plot'.nverts.'.png'

# get a good ytic val
stats 'parking-garage.g2opy_'.nverts.'.data' using 1 name 'gx' nooutput
stats 'parking-garage.g2opy_'.nverts.'.data' using 2 name 'gy' nooutput
stats 'parking-garage.optorch_'.nverts.'.data' using 1 name 'ox' nooutput
stats 'parking-garage.optorch_'.nverts.'.data' using 2 name 'oy' nooutput

set xlabel 'log time (s)'
set logscale xy

set xtics 10
set ytics 1+(1-gy_max/gy_min)/2
set autoscale yfix
set autoscale xfix
set offsets 0.1, 0.1, 0.01, 0.01
set lmargin 11

set style line 1 lc rgb "red"
set style line 2 lc rgb "blue"

plot \
'parking-garage.g2opy_'.nverts.'.data' using 1:2 ls 1 title 'g2opy' with linespoints, \
'parking-garage.optorch_'.nverts.'.data' using 1:2 ls 2 title 'optorch' with linespoints
