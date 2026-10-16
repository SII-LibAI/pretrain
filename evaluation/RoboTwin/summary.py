
from pathlib import Path

root = Path("/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/evaluation/RoboTwin/egopretrain/robotwin/demo_randomized")
expected_tasks, episodes = 50, 100
expected_ids = set(range(1, episodes + 1))
tasks = sorted(p for p in root.iterdir() if p.is_dir())


total_success = 0
for task in tasks:
    success = {int(p.stem.split("_", 1)[1]) for p in task.glob("success_*.mp4") if p.stat().st_size > 0}
    failure = {int(p.stem.split("_", 1)[1]) for p in task.glob("failure_*.mp4") if p.stat().st_size > 0}
    if success & failure or success | failure != expected_ids:
        print (f"{task.name} 的结果不完整或有重复：成功 {len(success)}，失败 {len(failure)}")
    total_success += len(success)
    print(f"{task.name}: {len(success)}/{episodes} = {len(success) / episodes:.1%}")