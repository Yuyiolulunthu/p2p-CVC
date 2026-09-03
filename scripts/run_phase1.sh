
set -e

python -m src.bench --sizes 256 512 1024 --out paper/bench.csv
python -m src.bench --check-resolution | tee paper/table2_resolution.txt

for ENC in resnet34 mit_b0 mobileone_s1 efficientnet-b0 timm-mobilenetv3_large_100; do
  for SEED in 0 1 2; do
    python -m src.train --config configs/phase1.yaml --seed $SEED --set encoder=$ENC
  done
done

python -m src.aggregate --runs runs --out paper/table1.tex
