#!/bin/bash
# =====================================================================
# CD-SEM segmentation — NCHC 晶創25 (Nano5) job array
#
#   sbatch --array=1-15 scripts/nano5_array.sbatch jobs/phase1.txt
#
# jobs/*.txt 每一行是一個 run 的參數，陣列第 N 個 task 跑第 N 行。
# 一次把整批實驗丟出去，而不是一個一個等。
# =====================================================================
#SBATCH --job-name=cdsem
#SBATCH --account=mst114556
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:50:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

JOBLIST="${1:?用法: sbatch --array=1-N nano5_array.sbatch jobs/phase1.txt}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
[ -n "$LINE" ] || { echo "job list 第 ${SLURM_ARRAY_TASK_ID} 行是空的"; exit 1; }

echo "=========================================="
echo "job      : $SLURM_JOB_NAME  $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "node     : $(hostname)"
echo "args     : $LINE"
echo "started  : $(date)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ---- 環境 ----
source .venv/bin/activate

# 預先抓好的權重。HF_HUB_OFFLINE=1 讓漏抓的立刻報錯，
# 而不是在沒有網路的節點上卡到逾時才失敗。
export TORCH_HOME="$SLURM_SUBMIT_DIR/weights_cache/torch"
export HF_HOME="$SLURM_SUBMIT_DIR/weights_cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# GPU 可能被同節點上還沒結束的 job 佔著。等一下再放棄，
# 而不是 3 秒內失敗、白白浪費一次排程。
python - <<'PYGPU'
import sys, time, torch
for i in range(30):                      # 最多等 10 分鐘
    if torch.cuda.is_available():
        print(f"torch {torch.__version__} | cuda True | {torch.cuda.get_device_name(0)}", flush=True)
        sys.exit(0)
    if i == 0:
        print("GPU 尚未就緒，等待中 ...", flush=True)
    time.sleep(20)
print("等了 10 分鐘仍拿不到 GPU，放棄", file=sys.stderr)
sys.exit(1)
PYGPU

# ---- 資料完整性：Linux 分大小寫，上來第一件事先確認還是 1390 / 12 class ----
python - <<'PY'
from src.data import scan_pairs
from collections import Counter
items = scan_pairs("dataset", verbose=False)
cls = Counter(i["cls"] for i in items)
print(f"pairs={len(items)} classes={len(cls)}")
assert len(items) == 1390, f"配對數不符：{len(items)}，預期 1390"
assert len(cls) == 12, f"class 數不符：{len(cls)}，預期 12"
PY

# ---- 跑 ----
eval "python -m src.train $LINE"

echo "=========================================="
echo "finished : $(date)"
