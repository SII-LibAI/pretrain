#!/usr/bin/env python3

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-memory CUDA utilization burner.")
    parser.add_argument("--duty-cycle", type=float, default=0.9)
    parser.add_argument("--cycle-seconds", type=float, default=0.25)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--operations-per-check", type=int, default=20)
    args = parser.parse_args()

    if not 0.0 < args.duty_cycle <= 1.0:
        parser.error("--duty-cycle must be in (0, 1].")
    if args.cycle_seconds <= 0.0:
        parser.error("--cycle-seconds must be positive.")
    if args.matrix_size <= 0:
        parser.error("--matrix-size must be positive.")
    if args.operations_per_check <= 0:
        parser.error("--operations-per-check must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to gpu_burner.py")

    device = torch.device("cuda:0")
    dtype = torch.float16
    left = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    torch.cuda.synchronize(device)

    allocated_mib = torch.cuda.memory_allocated(device) / 1024**2
    print(
        f"gpu_burner device={torch.cuda.get_device_name(device)} "
        f"duty_cycle={args.duty_cycle:.2f} matrix_size={args.matrix_size} "
        f"allocated={allocated_mib:.1f}MiB",
        flush=True,
    )

    work_seconds = args.cycle_seconds * args.duty_cycle
    with torch.inference_mode():
        while True:
            cycle_start = time.monotonic()
            work_deadline = cycle_start + work_seconds

            while time.monotonic() < work_deadline:
                for _ in range(args.operations_per_check):
                    torch.mm(left, right, out=output)
                torch.cuda.synchronize(device)

            sleep_seconds = cycle_start + args.cycle_seconds - time.monotonic()
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
