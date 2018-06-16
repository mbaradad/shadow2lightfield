#!/usr/bin/env bash
declare -a arr=("squares_no_parallax" "squares_parallax" "man")

for i in "${arr[@]}"
do
echo "Running experiment $i"
python main.py --amat data/plants_amat.npz --params_file data/$i/params.json --obs data/$i/obs.npz --homography data/$i/homography.npz --background data/$i/background.npz --ignore_region data/$i/ignore_region --out_dir output/$i
done