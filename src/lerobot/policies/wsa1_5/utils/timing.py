"""Stage timing on the execution device, without synchronizing every stage."""
import time

import torch


class StageTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.marks = []
        self.mark(None)

    def mark(self, name):
        if self.device.type == "cuda":
            stamp = torch.cuda.Event(enable_timing=True)
            stamp.record(torch.cuda.current_stream(self.device))
        else:
            stamp = time.perf_counter()
        self.marks.append((name, stamp))

    def values(self):
        if self.device.type == "cuda":
            self.marks[-1][1].synchronize()
        return {name: previous.elapsed_time(current) / 1000 if self.device.type == "cuda" else current - previous
                for (_, previous), (name, current) in zip(self.marks, self.marks[1:])}
