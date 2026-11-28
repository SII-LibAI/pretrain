"""
python /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/util_scripts/check_retarget.py  \
  --root /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/data/ego/Egoverse-Lerobot-v3/shard_002  \
  --future-frames 8 \
  --frame-stride 4 \
  --action-chunk-size 32 \
  --action-frame-stride 1
"""
#!/usr/bin/env python3
"""Fast, progress-printing timestamp alignment check for LeRobot V3 and retarget action.parquet.


Only compares matching index/episode_index/frame_index/timestamp; it does NOT
confirm true camera-to-pose synchronization or decoded video-frame PTS.
"""

print('[0] 检查脚本已启动（正在导入依赖）', flush=True)
import argparse
import json
from pathlib import Path
import sys
import time

try:
    import numpy as np
    import pyarrow.parquet as pq
except ImportError as e:
    print(f'缺少依赖: {e}; 请先安装 numpy 和 pyarrow', flush=True)
    raise

KEYS = ('index', 'episode_index', 'frame_index', 'timestamp')


def read_head(pf, n):
    existing = pf.schema_arrow.names
    missing = set(KEYS) - set(existing)
    if missing:
        raise RuntimeError(f'文件缺少字段: {sorted(missing)}')
    b = next(pf.iter_batches(batch_size=n, columns=list(KEYS)), None)
    return [] if b is None else b.to_pylist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='LeRobot V3 数据集根目录')
    p.add_argument('--action-file', type=Path, help='action.parquet 的路径（可覆盖默认值）')
    p.add_argument('--base-parquet', type=Path, help='指定要抽样的主数据 parquet（默认 data/ 的第一个）')
    p.add_argument('--samples', type=int, default=256, help='抽样主数据行数，默认256')
    p.add_argument('--max-action-rows', type=int, default=2000000, help='最多扫描 action.parquet 多少行，默认200万')
    p.add_argument('--batch-size', type=int, default=50000)
    p.add_argument('--tolerance-ms', type=float, default=1.0)
    p.add_argument('--future-frames', type=int, default=8)
    p.add_argument('--frame-stride', type=int, default=4)
    p.add_argument('--action-chunk-size', type=int, default=32)
    p.add_argument('--action-frame-stride', type=int, default=1)
    args = p.parse_args()

    if min(args.samples, args.max_action_rows, args.batch_size, args.future_frames,
           args.frame_stride, args.action_chunk_size, args.action_frame_stride) < 1:
        p.error('样本数、扫描上限和各个采样参数必须是正整数')

    t_start = time.monotonic()
    root = args.root.expanduser().resolve()
    action = (args.action_file or (root / 'meta/annotations/action.parquet')).expanduser().resolve()
    data = root / 'data'
    print(f'[1] 数据集: {root}', flush=True)
    print(f'    动作文件: {action}', flush=True)
    if not action.is_file():
        raise FileNotFoundError(f'找不到动作文件: {action}\n可用 --action-file 指定真实位置')
    if args.base_parquet:
        base = args.base_parquet.expanduser().resolve()
    else:
        if not data.is_dir():
            raise FileNotFoundError(f'找不到主数据目录: {data}')
        print('[2] 正在定位主数据 Parquet ...', flush=True)
        base = next(iter(sorted(data.rglob('*.parquet'))), None)
        if base is None:
            raise FileNotFoundError(f'{data} 下没有找到 parquet 文件')
    print(f'    主数据抽样文件: {base}', flush=True)

    print('[3] 读取主数据前几百行（仅4个索引/时间戳字段）...', flush=True)
    pf = pq.ParquetFile(base)
    rows = read_head(pf, args.samples)
    if not rows:
        raise RuntimeError('主数据 parquet 为空')
    wanted = {int(row['index']): row for row in rows}
    if len(wanted) != len(rows):
        print('警告: 抽样主数据 index 有重复', flush=True)
    print(f'    抽样 {len(rows)} 行; index 范围 {min(wanted)}–{max(wanted)}', flush=True)

    print('[4] 扫描 action.parquet 的对应 index（增量读取，不加载动作向量）...', flush=True)
    apf = pq.ParquetFile(action)
    missing = set(wanted)
    matches = {}
    scanned = 0
    missing_cols = set(KEYS) - set(apf.schema_arrow.names)
    if missing_cols:
        raise RuntimeError(f'action.parquet 缺少字段: {sorted(missing_cols)}')
    print(f'    action.parquet 总行数: {apf.metadata.num_rows:,}', flush=True)
    last_report = 0
    for batch in apf.iter_batches(batch_size=args.batch_size, columns=list(KEYS)):
        chunk = batch.to_pydict()
        for i, idx in enumerate(chunk['index']):
            idx = int(idx)
            if idx in missing:
                matches[idx] = {key: chunk[key][i] for key in KEYS}
                missing.discard(idx)
        scanned += len(chunk['index'])
        if scanned - last_report >= 100000 or not missing:
            print(f'    已扫描 {scanned:,} 行; 匹配 {len(matches)}/{len(wanted)}', flush=True)
            last_report = scanned
        if not missing or scanned >= args.max_action_rows:
            break

    print('\n========== 对齐检查结果 ==========', flush=True)
    print(f'主数据抽样: {len(wanted)} 行', flush=True)
    print(f'动作扫描: {scanned:,}/{apf.metadata.num_rows:,} 行', flush=True)
    print(f'匹配 index: {len(matches)}/{len(wanted)}', flush=True)
    if missing:
        print('未找到的部分 index (前10个):', sorted(missing)[:10], flush=True)
        if scanned < apf.metadata.num_rows:
            print('注意：只扫描了部分 action.parquet。未匹配不代表缺失。', flush=True)
            print('若需要继续扫描：提高 --max-action-rows 的值。', flush=True)
        else:
            print('警告：扫描完整个 action.parquet 后仍有 index 未匹配。', flush=True)

    if matches:
        match_keys = sorted(matches)
        ep_bad = [i for i in match_keys if int(wanted[i]['episode_index']) != int(matches[i]['episode_index'])]
        frame_bad = [i for i in match_keys if int(wanted[i]['frame_index']) != int(matches[i]['frame_index'])]
        times_ms = np.array([(float(matches[i]['timestamp']) - float(wanted[i]['timestamp'])) * 1000
                             for i in match_keys], dtype=float)
        print(f'episode_index 不匹配: {len(ep_bad)} 行', flush=True)
        print(f'frame_index 不匹配: {len(frame_bad)} 行', flush=True)
        print(f'时间戳绝对误差: 最大 {np.max(np.abs(times_ms)):.6f} ms; '
              f'P99 {np.percentile(np.abs(times_ms),99):.6f} ms', flush=True)
        print(f'超过 {args.tolerance_ms:g} ms: {np.count_nonzero(np.abs(times_ms) > args.tolerance_ms)} 行', flush=True)
        print('对齐示例（base_ts vs action_ts）:', flush=True)
        for i in match_keys[:min(5,len(match_keys))]:
            b, a = wanted[i], matches[i]
            dt_ms = (float(a['timestamp'])-float(b['timestamp']))*1000
            print(f'  index={i} episode={b["episode_index"]} frame={b["frame_index"]} '
                  f'base={float(b["timestamp"]):.6f}s action={float(a["timestamp"]):.6f}s '
                  f'delta={dt_ms:+.3f}ms', flush=True)
    else:
        print('没有匹配行，不能检验时间戳。可使用 --base-parquet 指定与动作数据相应的主数据文件。', flush=True)

    rgb_offsets = np.arange(1, args.future_frames+1) * args.frame_stride
    act_offsets = np.arange(1, args.action_chunk_size+1) * args.action_frame_stride
    print('\n========== 采样偏移检查 ==========', flush=True)
    print('RGB future 偏移:', rgb_offsets.tolist(), flush=True)
    print('Action 偏移: 从', int(act_offsets[0]), '到', int(act_offsets[-1]),
          '步长', args.action_frame_stride, flush=True)
    print('两者预测终点一致:', bool(rgb_offsets[-1] == act_offsets[-1]), flush=True)

    sample_sorted = sorted(rows, key=lambda x: (x['episode_index'], x['frame_index']))
    gaps = []
    for left, right in zip(sample_sorted[:-1], sample_sorted[1:]):
        if left['episode_index'] == right['episode_index'] and right['frame_index'] - left['frame_index'] == 1:
            gaps.append(float(right['timestamp']) - float(left['timestamp']))
    if gaps:
        print(f'抽样主数据连续帧间隔中位数: {np.median(gaps)*1000:.3f} ms '
              f'(约 {1/np.median(gaps):.2f} FPS)', flush=True)
    info_file = root/'meta/info.json'
    if info_file.is_file():
        try:
            with info_file.open(encoding='utf-8') as f:
                info = json.load(f)
            print('meta/info.json 声明 FPS:', info.get('fps','未知'), flush=True)
        except (OSError, ValueError) as e:
            print('无法读取 info.json:', e, flush=True)
    print(f'完成，耗时 {time.monotonic()-t_start:.1f} 秒。', flush=True)
    print('提示：timestamp 一致并不意味着原始视频帧和 retarget 姿态在物理上同步。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise

