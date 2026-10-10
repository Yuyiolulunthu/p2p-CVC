set -e
BEST=${1:-mit_b0}

# (1) crop 訓練 + 三種推論策略(Table 6)
python -m src.train --config configs/phase3.yaml --seed 0 --name crop_train
python -m src.infer_full --ckpt runs/crop_train/best.pt --mode whole tile blend

# (2) Leave-One-Class-Out(Table 7)。12 folds x 2 configs。
for CLS in C01 C02 C03 C04 C05 C06 C07 C08 C09 C10 C11 C12; do
  python -m src.train --config configs/phase3.yaml --loco-class $CLS --seed 0
  python -m src.train --config configs/phase3.yaml --loco-class $CLS --seed 0 \
    --set encoder=resnet34 variant=v0 --name baseline_loco$CLS
done
python -m src.aggregate --runs runs --loco --out paper/table7.tex

# (3) component-wise ablation(Table 8)
python -m src.train --config configs/phase3.yaml --seed 0 --set variant=v0        --name abl_no_thresh
python -m src.train --config configs/phase3.yaml --seed 0 --set lambda_tv=0       --name abl_no_tv
python -m src.train --config configs/phase3.yaml --seed 0 --set crop_margin=0     --name abl_no_crop
python -m src.train --config configs/phase3.yaml --seed 0 --set encoder_weights=null --name abl_no_pretrain
python -m src.aggregate --runs runs --out paper/table8.tex
