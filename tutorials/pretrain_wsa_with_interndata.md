### interndata目录

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series

export INTERNDATA_ROOT=/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/InternData-A1-v30

mkdir -p data

# 要求 data/a1 当前不存在
ln -s "$INTERNDATA_ROOT" data/a1
export HF_LEROBOT_HOME="$PWD/data"

ln -s "$PWD/stats" "$HF_LEROBOT_HOME/stats"

### 数据检查
cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series
find -L data/a1 -type d -name data \
  | while read -r d; do
      root="$(dirname "$d")"
      if [[ -d "$root/meta" && -d "$root/videos" ]]; then
        echo "$root"
      fi
    done


bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pretrain

cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series

PROC_PER_NODE=8 \
bash /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/launch/internvla_a1.5_pretrain_interndata.sh
'
```