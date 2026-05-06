# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utilities for Prometheus metrics collection."""

from dataclasses import dataclass
from enum import Enum

from tokenspeed.runtime.metrics.utils import exponential_buckets
from tokenspeed.runtime.utils.env import envs

TOKENSPEED_TEST_REQUEST_TIME_STATS = envs.TOKENSPEED_TEST_REQUEST_TIME_STATS.get()


@dataclass
class TimeStats:
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    lb_entry_time: float = 0.0
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0

    class RequestType(Enum):
        UNIFIED = "unified"
        PREFILL = "prefill"
        DECODE = "decode"
        INVALID = "invalid"

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time

    def __str__(self) -> str:
        # if unified
        _type = self.get_type()

        if _type == self.RequestType.UNIFIED:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time}"
        if _type == self.RequestType.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                assert (
                    bootstrap_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
            return f"bootstrap_duration={self.format_duration(bootstrap_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.prefill_bootstrap_queue_entry_time}"
        # if decode
        if _type == self.RequestType.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )

            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if TOKENSPEED_TEST_REQUEST_TIME_STATS:
                assert (
                    prealloc_duration >= 0
                    and transfer_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"prealloc_duration={self.format_duration(prealloc_duration)}, transfer_duration={self.format_duration(transfer_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.decode_prealloc_queue_entry_time}"
        return "Invalid Time Stats"

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def get_type(self) -> RequestType:
        """Determine the type of request based on timestamp values."""
        if (
            self.prefill_bootstrap_queue_entry_time == 0.0
            and self.prefill_transfer_queue_entry_time == 0.0
            and self.decode_prealloc_queue_entry_time == 0.0
            and self.decode_transfer_queue_entry_time == 0.0
        ):
            return self.RequestType.UNIFIED
        elif (
            self.prefill_bootstrap_queue_entry_time > 0.0
            and self.prefill_transfer_queue_entry_time > 0.0
        ):
            return self.RequestType.PREFILL
        elif (
            self.decode_prealloc_queue_entry_time > 0.0
            and self.decode_transfer_queue_entry_time > 0.0
            and self.wait_queue_entry_time > 0.0
        ):
            return self.RequestType.DECODE
        else:
            return self.RequestType.INVALID


class SchedulerMetricsCollector:
    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        self.enable_prometheus = "prometheus" in metrics_reporters
        self.labels = labels

        if self.enable_prometheus:
            self._init_prometheus(labels)

    def _init_prometheus(self, labels: dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Histogram

        self.labels = labels

        self.request_latency_seconds = Histogram(
            name="tokenspeed:request_latency_seconds",
            documentation="The latency of each stage of requests.",
            # captures latency in range [1ms - ~1191s]
            buckets=exponential_buckets(start=0.001, width=1.62, length=30),
            labelnames=list(labels.keys()) + ["stage"],
        )

    def observe_request_latency_seconds(self, stage: str, latency: float) -> None:
        if self.enable_prometheus:
            labels_with_stage = {**self.labels, "stage": stage}
            self.request_latency_seconds.labels(**labels_with_stage).observe(latency)


class TokenizerMetricsCollector:
    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        self.enable_prometheus = "prometheus" in metrics_reporters

        if self.enable_prometheus:
            self._init_prometheus(labels)

    def _init_prometheus(self, labels: dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels

        self.prompt_tokens_total = Counter(
            name="tokenspeed:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )

        self.generation_tokens_total = Counter(
            name="tokenspeed:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )

        self.num_requests_total = Counter(
            name="tokenspeed:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.histogram_time_to_first_token = Histogram(
            name="tokenspeed:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.1,
                0.3,
                0.5,
                0.7,
                0.9,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                120,
                160,
            ],
        )

        self.histogram_time_per_output_token = Histogram(
            name="tokenspeed:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.002,
                0.005,
                0.010,
                0.020,
                0.030,
                0.040,
                0.050,
                0.060,
                0.070,
                0.080,
                0.090,
                0.100,
                0.150,
                0.200,
                0.300,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
            ],
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="tokenspeed:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.050,
                0.075,
                0.100,
                0.150,
                0.200,
                0.300,
                0.400,
                0.500,
                0.750,
                1.000,
                2.000,
            ],
        )

        self.histogram_e2e_request_latency = Histogram(
            name="tokenspeed:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=[
                0.1,
                0.2,
                0.4,
                0.8,
                1,
                2,
                5,
                10,
                20,
                40,
                60,
                80,
                100,
                150,
                200,
                250,
                300,
                350,
                500,
                1000,
            ],
        )

    def _log_histogram(self, histogram, data: int | float) -> None:
        if self.enable_prometheus:
            histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        e2e_latency: float,
        tokenized_duration: float,
    ):
        if self.enable_prometheus:
            self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
            self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
            self.num_requests_total.labels(**self.labels).inc(1)
            self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)
            if generation_tokens >= 1:
                self.histogram_time_per_output_token.labels(**self.labels).observe(
                    e2e_latency / generation_tokens
                )

    def observe_time_to_first_token(self, value: float):
        if self.enable_prometheus:
            self.histogram_time_to_first_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(
        self, internval: float, num_new_tokens: int, name: str = "TPOT"
    ):
        adjusted_interval = internval / num_new_tokens

        if self.enable_prometheus:
            # A faster version of the Histogram::observe which observes multiple values at the same time.
            # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
            his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
            his._sum.inc(internval)

            for i, bound in enumerate(his._upper_bounds):
                if adjusted_interval <= bound:
                    his._buckets[i].inc(num_new_tokens)
                    break

    def observe_request_arrival(self, batch_size: int = 1):
        if self.enable_prometheus:
            pass


class ErrorMetricsCollector:
    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        self.enable_prometheus = "prometheus" in metrics_reporters

    def record_error(self, error_message: str) -> None:
        return


class KVTransferMetricsCollector:

    def __init__(self, labels: dict[str, str], metrics_reporters: list[str]) -> None:
        pass

    def log_kv_transfer_timeout(self) -> None:
        return

    def log_kv_transfer_failed(self) -> None:
        return

    def log_kv_transfer_time(self, transfer_time_seconds: float) -> None:
        return
