#!/bin/bash

# Define the range of scale values
scales=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Iterate over all combinations of scale1 and scale2
for scale1 in "${scales[@]}"; do
    for scale2 in "${scales[@]}"; do
        echo "Running with scale1=$scale1 and scale2=$scale2"
        python examples/brushnet/evaluate_brushnet.py --scale1 $scale1 --scale2 $scale2
    done
done
