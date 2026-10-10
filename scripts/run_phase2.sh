set -e
BEST=${1:-mit_b0}   # Phase 1 選出的 encoder

# (a) 古典 thresholding 對照組(Table 5),先跑,很快
python -m src.classical --data data --split splits/split_seed0.json

# (b) 五個變體 x 兩個 encoder(Table 3)
for ENC in $BEST resnet34; do
  for V in v0 v1 v2 v3 v4; do
    python -m src.train --config configs/phase2.yaml --seed 0 \
      --set encoder=$ENC variant=$V
  done
done

# (c) lambda_tv 掃描(Table 4)—— 這是「解掉 noisy background」的實際旋鈕
for TV in 0 0.001 0.01 0.1; do
  python -m src.train --config configs/phase2.yaml --seed 0 \
    --set encoder=$BEST variant=v4 lambda_tv=$TV
done

# (d) 勝出組合補跑 seed 1,2(主表要 3 seeds)
for SEED in 1 2; do
  python -m src.train --config configs/phase2.yaml --seed $SEED --set encoder=$BEST
done

python -m src.aggregate --runs runs --out paper/table3.tex
