"""WSA1.5 RoboTwin entry using the same environment interface as ../inference.py."""
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import sys
import traceback

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "RoboTwin"
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.RoboTwin.wsa1_5.policy import WSA1_5Agent


TASK_NAMES = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=Path("outputs/robotwin/wsa1_5"))
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mode", choices=("auto", "abs", "delta"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-horizon", type=int, default=0, help="0 uses the checkpoint action chunk size")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--robot-type", default="aloha")
    parser.add_argument("--stats-path", type=Path, help="Optional training stats; defaults to checkpoint stats.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--text-device", default="cpu")
    return parser.parse_args()


def require_robotwin():
    if not (ROBOTWIN_ROOT / "envs").exists():
        raise RuntimeError(
            f"Use the same RoboTwin checkout as the original evaluation entry at {ROBOTWIN_ROOT} "
            "and install its dependencies. See evaluation/RoboTwin/wsa1_5/README.md."
        )

    for path in [
        ROBOTWIN_ROOT,
        ROBOTWIN_ROOT / "policy",
        ROBOTWIN_ROOT / "description" / "utils",
    ]:
        if str(path) not in sys.path:
            sys.path.append(str(path))

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    return CONFIGS_PATH, UnStableError, generate_episode_descriptions


def get_embodiment_config(robot_file: str):
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return OmegaConf.load(f)


def make_env(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def build_task_args(task_config: str, task_name: str, configs_path: str):
    # RoboTwin exports the active task-config directory through CONFIGS_PATH.
    # This is `env_cfg/task_config` in current versions and `task_config` in
    # older versions, so avoid hard-coding either repository layout here.
    configs_dir = Path(configs_path)
    with open(configs_dir / f"{task_config}.yml", "r", encoding="utf-8") as f:
        task_args = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    with open(configs_dir / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    with open(configs_dir / "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment_type}")
        return robot_file

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = str(ROBOTWIN_ROOT / get_embodiment_file(embodiment_type[0]))
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = str(ROBOTWIN_ROOT / get_embodiment_file(embodiment_type[0]))
        task_args["right_robot_file"] = str(ROBOTWIN_ROOT / get_embodiment_file(embodiment_type[1]))
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("RoboTwin embodiment list must contain 1 or 3 items.")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    # This entry saves its own canvas replay, so Base_Task must not expect an ffmpeg pipe.
    task_args["eval_video_save_dir"] = None
    return task_args


def save_replay_video(video_path: Path, replay_images: list[np.ndarray], fps: int):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = video_path.with_name(f".{video_path.stem}.tmp{video_path.suffix}")
    imageio.mimwrite(temporary_path, replay_images, fps=fps)
    temporary_path.replace(video_path)


def maybe_close_env(task_env, *, clear_cache: bool = False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except TypeError:
        task_env.close_env()
    except Exception:
        logging.debug("Failed to close RoboTwin env cleanly.", exc_info=True)


def infer_once(args):
    if not 0 <= args.task_idx < len(TASK_NAMES):
        raise IndexError(f"task_idx must be in [0, {len(TASK_NAMES) - 1}]")
    if args.infer_horizon < 0 or args.num_episodes <= 0 or args.fps <= 0 or args.seed < 0:
        raise ValueError("Invalid horizon, episode count, fps or seed")
    configs_path, unstable_error, generate_episode_descriptions = require_robotwin()
    task_name = TASK_NAMES[args.task_idx]
    task_args = build_task_args(args.task_config, task_name, configs_path)
    task_args.setdefault("data_type", {}).update(rgb=True, qpos=True)
    task_env = make_env(task_name)
    agent = WSA1_5Agent.from_checkpoint(args)
    run_episodes(args, task_env, task_args, unstable_error, generate_episode_descriptions,
                 agent.step, agent.reset, agent.action_type)


def run_episodes(args, task_env, task_args, unstable_error, generate_episode_descriptions,
                 policy_step, policy_reset, action_type):
    """Use the same expert-valid seed selection and success/video reporting for both policies."""
    task_name = TASK_NAMES[args.task_idx]
    task_env.suc = 0
    task_env.test_num = 0
    np.random.seed(args.seed)

    seed_start = 100000 * (1 + args.seed)
    seed_cursor = seed_start
    episode_id = 0
    clear_cache_freq = task_args["clear_cache_freq"]
    successful_seed_count = 0

    args.video_dir.mkdir(parents=True, exist_ok=True)

    while successful_seed_count < args.num_episodes:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0
        seed_value = seed_cursor

        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed_value, is_test=True, **task_args)
            episode_info = task_env.play_once()
            maybe_close_env(task_env)
        except unstable_error as exc:
            logging.warning("Skipping unstable seed for task=%s seed=%s: %s", task_name, seed_value, exc)
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue
        except Exception:
            logging.error("Expert rollout failed for task=%s seed=%s", task_name, seed_value)
            logging.error(traceback.format_exc())
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue

        if not (task_env.plan_success and task_env.check_success()):
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed_value, is_test=True, **task_args)
        except unstable_error as exc:
            logging.warning("Skipping unstable rollout initialization for task=%s seed=%s: %s", task_name, seed_value, exc)
            maybe_close_env(task_env)
            seed_cursor += 1
            continue
        except Exception:
            maybe_close_env(task_env)
            raise
        successful_seed_count += 1
        try:
            descriptions = generate_episode_descriptions(task_name, [episode_info["info"]], args.num_episodes)
            instruction = str(np.random.choice(descriptions[0][args.instruction_type]))
            task_env.set_instruction(instruction=instruction)

            policy_reset(seed_value)
            replay_images: list[np.ndarray] = []
            success = False

            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                action, image = policy_step(observation, task_env.get_instruction())
                replay_images.append(image)
                task_env.take_action(action, action_type=action_type)

                if task_env.eval_success:
                    success = True
                    break

            if success:
                task_env.suc += 1
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")

            suffix = "success" if success else "failure"
            save_replay_video(args.video_dir / f"{suffix}_{successful_seed_count}.mp4", replay_images, args.fps)
        finally:
            maybe_close_env(task_env, clear_cache=((successful_seed_count + 1) % clear_cache_freq == 0))
            if getattr(task_env, "render_freq", 0):
                task_env.viewer.close()

        episode_id += 1

        task_env.test_num += 1
        print(
            f"\033[93m{task_name}\033[0m | \033[92m{task_args['task_config']}\033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
            f"\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{seed_cursor}\033[0m\n"
        )
        seed_cursor += 1


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logging.getLogger("curobo").setLevel(logging.WARNING)
    if any(args.video_dir.glob("success_*.mp4")) or any(args.video_dir.glob("failure_*.mp4")):
        raise FileExistsError(f"Evaluation results already exist in {args.video_dir}; choose a new output directory")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Starting WSA1.5 RoboTwin evaluation: task=%s checkpoint=%s", args.task_idx, args.ckpt_path)
    infer_once(args)


if __name__ == "__main__":
    main()
