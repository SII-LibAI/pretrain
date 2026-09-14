#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}

if [[ ${NUM_GPUS} -lt 1 || -z "${GPU_ARR[0]}" ]]; then
    echo "ERROR: GPUS cannot be empty" >&2
    exit 2
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CHECKPOINT OUTPUT_ROOT [TASK_CONFIG] [TASK_START] [TASK_END] [NUM_EPISODES]" >&2
    echo "Example: GPUS=0,1,2,3,4,5,6,7 $0 /path/to/pretrained_model outputs/my_run demo_clean 0 49 100" >&2
    exit 2
fi

PRETRAINED_CKPT="$1"
OUTPUT_ROOT="$2"
TASK_CONFIG="${3:-${TASK_CONFIG:-demo_clean}}"
TASK_START="${4:-${TASK_START:-0}}"
TASK_END="${5:-${TASK_END:-49}}"
NUM_EPISODES="${6:-${NUM_EPISODES:-100}}"

ACTION_MODE="${ACTION_MODE:-abs}"
INFER_HORIZON="${INFER_HORIZON:-30}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
DTYPE="${DTYPE:-bfloat16}"
FORCE_RERUN="${FORCE_RERUN:-false}"

if (( TASK_START < 0 || TASK_END < TASK_START || TASK_END > 49 )); then
    echo "ERROR: invalid task range ${TASK_START}..${TASK_END}; valid range is 0..49" >&2
    exit 2
fi

if (( NUM_EPISODES < 1 )); then
    echo "ERROR: NUM_EPISODES must be positive; got ${NUM_EPISODES}" >&2
    exit 2
fi

if [[ -d "${PRETRAINED_CKPT}" ]]; then
    PRETRAINED_CKPT="$(cd "${PRETRAINED_CKPT}" && pwd)"
fi

task_names=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_burger_fries"
    "place_can_basket"
    "place_cans_plasticbox"
    "place_container_plate"
    "place_dual_shoes"
    "place_empty_cup"
    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle"
    "shake_bottle_horizontally"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
    "stamp_seal"
    "turn_switch"
)

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

LOG_ROOT="${OUTPUT_ROOT}/logs/${TASK_CONFIG}"
mkdir -p "${LOG_ROOT}"

SCHED_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/robotwin_eval.XXXXXX")"
NEXT_TASK_FILE="${SCHED_ROOT}/next_task"
LOCK_DIR="${SCHED_ROOT}/lock"
FAIL_FILE="${SCHED_ROOT}/failed"
printf '%s\n' "${TASK_START}" >"${NEXT_TASK_FILE}"

PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" >/dev/null 2>&1 || true
    done
    rm -rf -- "${SCHED_ROOT}"
}
trap cleanup EXIT INT TERM

claim_task() {
    local task_idx
    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
        sleep 0.1
    done

    task_idx="$(<"${NEXT_TASK_FILE}")"
    if [[ ! "${task_idx}" =~ ^[0-9]+$ ]] || (( task_idx > TASK_END )); then
        rmdir "${LOCK_DIR}"
        return 1
    fi

    printf '%s\n' "$((task_idx + 1))" >"${NEXT_TASK_FILE}"
    rmdir "${LOCK_DIR}"
    printf '%s\n' "${task_idx}"
}

completed_episode_count() {
    local output_path="$1"
    local episode_idx
    local count=0

    for ((episode_idx = 1; episode_idx <= NUM_EPISODES; episode_idx++)); do
        if [[ -s "${output_path}/success_${episode_idx}.mp4" || -s "${output_path}/failure_${episode_idx}.mp4" ]]; then
            count=$((count + 1))
        fi
    done

    printf '%s\n' "${count}"
}

task_is_complete() {
    local output_path="$1"
    local marker="${output_path}/.complete_${NUM_EPISODES}"
    local count

    if [[ -f "${marker}" ]]; then
        return 0
    fi

    count="$(completed_episode_count "${output_path}")"
    if (( count >= NUM_EPISODES )); then
        touch "${marker}"
        return 0
    fi

    return 1
}

worker_loop() {
    local worker_idx="$1"
    local gpu="$2"
    local task_idx
    local task_name
    local output_path
    local log_file
    local existing_count

    while task_idx="$(claim_task)"; do
        task_name="${task_names[$task_idx]}"
        output_path="${OUTPUT_ROOT}/robotwin/${TASK_CONFIG}/${task_name}"
        log_file="${LOG_ROOT}/${task_name}.log"

        if [[ "${FORCE_RERUN}" != "true" ]] && task_is_complete "${output_path}"; then
            echo "SKIP worker=${worker_idx} gpu=${gpu} task=${task_idx}:${task_name} status=complete"
            continue
        fi

        existing_count="$(completed_episode_count "${output_path}")"
        echo "RUN  worker=${worker_idx} gpu=${gpu} task=${task_idx}:${task_name} existing=${existing_count}/${NUM_EPISODES} log=${log_file}"

        if CUDA_VISIBLE_DEVICES="${gpu}" \
            INFERENCE_BACKEND="${INFERENCE_BACKEND}" \
            bash "${SINGLE_EVAL_SCRIPT}" \
                "${PRETRAINED_CKPT}" \
                "${output_path}" \
                "${TASK_CONFIG}" \
                "${task_idx}" \
                "${ACTION_MODE}" \
                "${INFER_HORIZON}" \
                "${NUM_EPISODES}" \
                "${DTYPE}" \
                >"${log_file}" 2>&1; then
            if task_is_complete "${output_path}"; then
                echo "DONE worker=${worker_idx} gpu=${gpu} task=${task_idx}:${task_name}"
            else
                echo "ERROR: task exited successfully but produced fewer than ${NUM_EPISODES} results: ${task_name}" >&2
                printf '%s\n' "task=${task_idx}:${task_name} log=${log_file} reason=incomplete_output" >>"${FAIL_FILE}"
            fi
        else
            echo "ERROR: task failed: ${task_idx}:${task_name}; log=${log_file}" >&2
            printf '%s\n' "task=${task_idx}:${task_name} log=${log_file} reason=process_failed" >>"${FAIL_FILE}"
        fi
    done
}

echo "CHECKPOINT        = ${PRETRAINED_CKPT}"
echo "OUTPUT_ROOT       = ${OUTPUT_ROOT}"
echo "TASK_CONFIG       = ${TASK_CONFIG}"
echo "TASK_RANGE        = ${TASK_START}..${TASK_END}"
echo "NUM_EPISODES      = ${NUM_EPISODES}"
echo "GPUS              = ${GPUS}"
echo "ACTION_MODE       = ${ACTION_MODE}"
echo "INFER_HORIZON     = ${INFER_HORIZON}"
echo "INFERENCE_BACKEND = ${INFERENCE_BACKEND}"
echo "DTYPE             = ${DTYPE}"

for worker_idx in "${!GPU_ARR[@]}"; do
    worker_loop "${worker_idx}" "${GPU_ARR[$worker_idx]}" &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

if [[ -f "${FAIL_FILE}" ]]; then
    echo "Some RoboTwin tasks failed or remained incomplete:" >&2
    cat "${FAIL_FILE}" >&2
    exit 1
fi

echo "All requested RoboTwin tasks are complete. Logs: ${LOG_ROOT}"
