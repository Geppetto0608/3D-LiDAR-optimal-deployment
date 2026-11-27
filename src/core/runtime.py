"""Opt-in work counters and timing for sequential placement experiments."""
from collections import Counter
from contextlib import contextmanager
import resource
import time


class RuntimeMeter:
    def __init__(self):
        self.counts = Counter()
        self.stages = Counter()
        self.trace = []
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()

    def add(self, name, amount=1):
        self.counts[name] += int(amount)

    @staticmethod
    def synchronize():
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.stages[name] += time.perf_counter() - start

    def observe(self, coverage, source):
        self.synchronize()
        self.trace.append(dict(elapsed_sec=time.perf_counter() - self.started,
                               coverage_pct=float(coverage), source=source))

    def report(self):
        import torch
        self.synchronize()
        return dict(elapsed_sec=time.perf_counter() - self.started,
                    cpu_process_sec=time.process_time() - self.started_cpu,
                    stages_sec=dict(self.stages), counts=dict(self.counts),
                    quality_trace=self.trace,
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    torch_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                        if torch.cuda.is_initialized() else 0),
                    torch_peak_reserved_bytes=(torch.cuda.max_memory_reserved()
                        if torch.cuda.is_initialized() else 0),
                    memory_scope='Linux process lifetime RSS; PyTorch allocator only, not total GPU memory')
