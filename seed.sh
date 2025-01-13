scales=(2 3 4 5 6 7 8 9 10 11)

# Iterate over all combinations of scale1 and scale2
for scale1 in "${scales[@]}"; do
    echo "Running with scale1=$scale1"
    python examples/brushnet/evaluate_brushnet.py --blended --scale1=0 --scale2=0 --seed "$scale1"
    python examples/brushnet/evaluate_brushnet.py --blended --scale1=0.4 --scale2=0.7 --seed "$scale1"
done