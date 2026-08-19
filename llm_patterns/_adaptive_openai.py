"""Private adaptive provider-call scheduling for local OpenAI operations."""

# 0) Imports
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")

_STATE_SCHEMA = "adaptive-openai-concurrency-v1"
_INITIAL_START_INTERVAL_SECONDS = 0.05
_MINIMUM_START_INTERVAL_SECONDS = 0.005
_MAXIMUM_START_INTERVAL_SECONDS = 5.0
_MAXIMUM_PERSISTED_PAUSE_SECONDS = 24 * 60 * 60

# 1) Sub functions
# Batch policy
@dataclass(frozen=True)
class _BatchConfig:
    max_concurrency: int
    max_retries: int
    state_path: Path | None


def _batch_config(batch: bool | Mapping[str, object]) -> _BatchConfig | None:
    batch_keys = {"max_concurrency", "max_retries", "state_path"}
    if batch is False:
        return None
    if batch is True:
        return _validated_batch_config(
            max_concurrency=100,
            max_retries=5,
            state_path=None,
        )
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be False, True, or a mapping of batch options")
    unexpected = set(batch) - batch_keys
    if unexpected:
        names = ", ".join(sorted(str(name) for name in unexpected))
        raise ValueError(f"unsupported batch options: {names}")
    return _validated_batch_config(
        max_concurrency=batch.get("max_concurrency", 100),
        max_retries=batch.get("max_retries", 5),
        state_path=batch.get("state_path"),
    )


def _validated_batch_config(
    max_concurrency: object,
    max_retries: object,
    state_path: object,
) -> _BatchConfig:
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise TypeError("max_concurrency must be an integer")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise TypeError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if state_path is None:
        normalized_state_path = None
    elif isinstance(state_path, str):
        if not state_path.strip():
            raise ValueError("state_path must not be blank")
        normalized_state_path = Path(state_path)
    elif isinstance(state_path, Path):
        normalized_state_path = state_path
    else:
        raise TypeError("state_path must be a string, Path, or None")
    return _BatchConfig(
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        state_path=normalized_state_path,
    )


def _batch_inputs(input_text: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(input_text, Mapping):
        raise TypeError(
            "input_text must be a mapping of stable input IDs to text in batch mode"
        )
    if not input_text:
        raise ValueError("input_text must not be empty in batch mode")
    normalized: dict[str, str] = {}
    for input_id, text in input_text.items():
        if not isinstance(input_id, str) or not input_id.strip():
            raise ValueError("batch input IDs must be nonblank strings")
        if input_id != input_id.strip():
            raise ValueError("batch input IDs must not have surrounding whitespace")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every batch input must be a nonblank string")
        normalized[input_id] = text.strip()
    return normalized


class _BatchCancelled(RuntimeError):
    pass


# Shared provider pressure and retry state
class _AdaptiveProviderCaller:
    base_backoff_seconds = 1.0
    jitter_seconds = 0.25

    def __init__(self, scheduler: "_AdaptiveScheduler", max_retries: int) -> None:
        self._scheduler = scheduler
        self._max_retries = max_retries
        self._condition = threading.Condition()
        self._active_calls = 0
        self._next_start_time = time.monotonic()
        self._aborted = False

    def call(self, method: Callable[..., object], **request: object) -> object:
        for retry_number in range(self._max_retries + 1):
            self._acquire_call_slot()
            try:
                response = method(**request)
            except Exception as error:
                fallback_delay = self.base_backoff_seconds * (2**retry_number)
                fallback_delay += random.random() * self.jitter_seconds
                self._release_call_slot(error, fallback_delay)
                if (
                    not _is_retryable_openai_error(error)
                    or retry_number == self._max_retries
                ):
                    raise
                self._wait_before_retry(fallback_delay)
            else:
                self._release_call_slot(None, 0.0)
                return response
        raise RuntimeError("Provider retry loop ended unexpectedly")

    def abort(self) -> None:
        with self._condition:
            self._aborted = True
            self._condition.notify_all()

    def _acquire_call_slot(self) -> None:
        with self._condition:
            while True:
                if self._aborted:
                    raise _BatchCancelled("Adaptive operation batch was cancelled")
                now = time.monotonic()
                allowed_start_time = max(
                    self._next_start_time,
                    self._scheduler.pause_until,
                )
                if (
                    self._active_calls < self._scheduler.current_concurrency
                    and now >= allowed_start_time
                ):
                    self._active_calls += 1
                    self._scheduler.observe_launch(self._active_calls)
                    self._next_start_time = (
                        max(self._next_start_time, now)
                        + self._scheduler.current_start_interval_seconds
                    )
                    return
                timeout = (
                    max(0.0, allowed_start_time - now)
                    if self._active_calls < self._scheduler.current_concurrency
                    else None
                )
                self._condition.wait(timeout=timeout)

    def _release_call_slot(
        self,
        error: Exception | None,
        fallback_delay_seconds: float,
    ) -> None:
        with self._condition:
            self._active_calls -= 1
            if error is None:
                self._scheduler.observe_success()
            elif _is_openai_backpressure_error(error):
                self._scheduler.observe_backpressure(error, fallback_delay_seconds)
            self._condition.notify_all()

    def _wait_before_retry(self, fallback_delay_seconds: float) -> None:
        retry_not_before = time.monotonic() + fallback_delay_seconds
        with self._condition:
            while True:
                if self._aborted:
                    raise _BatchCancelled("Adaptive operation batch was cancelled")
                resume_at = max(retry_not_before, self._scheduler.pause_until)
                remaining = resume_at - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(timeout=remaining)


@dataclass
class _AdaptiveScheduler:
    maximum_concurrency: int
    scope_sha256: str
    state_path: Path | None
    current_concurrency: int
    current_start_interval_seconds: float
    stable_concurrency: int
    stable_start_interval_seconds: float
    pause_until: float = 0.0
    resume_not_before_epoch_seconds: float = 0.0
    cooldown_level: int = 0
    successful_completions: int = 0
    peak_active_count: int = 0

    @classmethod
    def create(
        cls,
        maximum_concurrency: int,
        model: str,
        scope: str,
        state_path: Path | None,
    ) -> "_AdaptiveScheduler":
        initial_concurrency = min(10, maximum_concurrency)
        initial_interval = _INITIAL_START_INTERVAL_SECONDS
        cooldown_decay_seconds = 60 * 60
        scope_sha256 = hashlib.sha256(
            f"openai-responses\x1f{model}\x1f{scope}".encode()
        ).hexdigest()
        persisted = _load_state(state_path, scope_sha256)
        current_epoch = time.time()
        stable_concurrency = initial_concurrency
        stable_interval = initial_interval
        cooldown_level = 0
        resume_epoch = 0.0
        if persisted is not None:
            stable_concurrency = _bounded_int(
                persisted.get("stable_concurrency"),
                minimum=1,
                maximum=maximum_concurrency,
                default=initial_concurrency,
            )
            stable_interval = _bounded_float(
                persisted.get("stable_start_interval_seconds"),
                minimum=_MINIMUM_START_INTERVAL_SECONDS,
                maximum=_MAXIMUM_START_INTERVAL_SECONDS,
                default=initial_interval,
            )
            cooldown_level = _bounded_int(
                persisted.get("cooldown_level"),
                minimum=0,
                maximum=32,
                default=0,
            )
            resume_epoch = _bounded_float(
                persisted.get("resume_not_before_epoch_seconds"),
                minimum=0.0,
                maximum=current_epoch + _MAXIMUM_PERSISTED_PAUSE_SECONDS,
                default=0.0,
            )
            updated_at = _bounded_float(
                persisted.get("updated_at_epoch_seconds"),
                minimum=0.0,
                maximum=current_epoch,
                default=current_epoch,
            )
            if resume_epoch <= current_epoch:
                elapsed_windows = int(
                    max(0.0, current_epoch - updated_at) / cooldown_decay_seconds
                )
                cooldown_level = max(0, cooldown_level - elapsed_windows)
                resume_epoch = 0.0
        return cls(
            maximum_concurrency=maximum_concurrency,
            scope_sha256=scope_sha256,
            state_path=state_path,
            current_concurrency=stable_concurrency,
            current_start_interval_seconds=stable_interval,
            stable_concurrency=stable_concurrency,
            stable_start_interval_seconds=stable_interval,
            pause_until=time.monotonic() + max(0.0, resume_epoch - current_epoch),
            resume_not_before_epoch_seconds=resume_epoch,
            cooldown_level=cooldown_level,
        )

    def observe_launch(self, active_count: int) -> None:
        self.peak_active_count = max(self.peak_active_count, active_count)

    def observe_success(self) -> None:
        if time.monotonic() < self.pause_until:
            return
        self.successful_completions += 1
        if self.successful_completions < max(4, self.current_concurrency):
            return
        concurrency_was_exercised = self.peak_active_count >= self.current_concurrency
        if concurrency_was_exercised:
            self.stable_concurrency = self.current_concurrency
            self.current_concurrency = min(
                self.maximum_concurrency,
                self.current_concurrency + 1,
            )
        self.current_start_interval_seconds = max(
            _MINIMUM_START_INTERVAL_SECONDS,
            self.current_start_interval_seconds * 0.9,
        )
        self.stable_start_interval_seconds = self.current_start_interval_seconds
        self.cooldown_level = max(0, self.cooldown_level - 1)
        self.resume_not_before_epoch_seconds = 0.0
        self.successful_completions = 0
        self.peak_active_count = 0
        self.persist()

    def observe_backpressure(
        self, error: Exception, fallback_delay_seconds: float
    ) -> None:
        provider_pause = _openai_pause_seconds(error)
        now = time.monotonic()
        new_episode = now >= self.pause_until
        if new_episode:
            self.cooldown_level = min(32, self.cooldown_level + 1)
            inferred_pause = min(
                _MAXIMUM_PERSISTED_PAUSE_SECONDS,
                max(fallback_delay_seconds, 2 ** max(0, self.cooldown_level - 1)),
            )
            pause_seconds = (
                provider_pause if provider_pause is not None else inferred_pause
            )
            self.current_concurrency = max(1, self.current_concurrency // 2)
            self.current_start_interval_seconds = min(
                _MAXIMUM_START_INTERVAL_SECONDS,
                max(
                    _INITIAL_START_INTERVAL_SECONDS,
                    self.current_start_interval_seconds * 2,
                ),
            )
            self.stable_concurrency = self.current_concurrency
            self.stable_start_interval_seconds = self.current_start_interval_seconds
        else:
            pause_seconds = provider_pause or fallback_delay_seconds
        self.pause_until = max(self.pause_until, now + pause_seconds)
        self.resume_not_before_epoch_seconds = max(
            self.resume_not_before_epoch_seconds,
            time.time() + self.seconds_until_resume(),
        )
        self.successful_completions = 0
        self.peak_active_count = 0
        self.persist()

    def seconds_until_resume(self) -> float:
        return max(0.0, self.pause_until - time.monotonic())

    def persist(self) -> None:
        if self.state_path is None:
            return
        _persist_state(
            self.state_path,
            {
                "schema": _STATE_SCHEMA,
                "scope_sha256": self.scope_sha256,
                "stable_concurrency": self.stable_concurrency,
                "stable_start_interval_seconds": round(
                    self.stable_start_interval_seconds,
                    6,
                ),
                "cooldown_level": self.cooldown_level,
                "resume_not_before_epoch_seconds": round(
                    self.resume_not_before_epoch_seconds,
                    3,
                ),
                "updated_at_epoch_seconds": round(time.time(), 3),
            },
        )


# OpenAI failure classification
def _is_retryable_openai_error(error: Exception) -> bool:
    if "insufficient_quota" in str(error).casefold():
        return False
    if type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    }:
        return True
    status = _status_code(error)
    return status in {408, 409, 429} or (status is not None and 500 <= status <= 599)


def _is_openai_backpressure_error(error: Exception) -> bool:
    if "insufficient_quota" in str(error).casefold():
        return False
    return type(error).__name__ == "RateLimitError" or _status_code(error) == 429


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response_status = getattr(getattr(error, "response", None), "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _openai_pause_seconds(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    retry_after = _nonnegative_float(getter("retry-after"))
    if retry_after is not None and retry_after > 0:
        return retry_after

    exhausted_resets: list[float] = []
    for remaining_name, reset_name in (
        ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"),
        ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
        (
            "x-ratelimit-remaining-project-tokens",
            "x-ratelimit-reset-project-tokens",
        ),
    ):
        remaining = _nonnegative_float(getter(remaining_name))
        reset_seconds = _duration_seconds(getter(reset_name))
        if remaining == 0 and reset_seconds is not None and reset_seconds > 0:
            exhausted_resets.append(reset_seconds)
    return max(exhausted_resets) if exhausted_resets else None


def _duration_seconds(value: object) -> float | None:
    duration_part = re.compile(r"(?P<number>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    position = 0
    seconds = 0.0
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    for match in duration_part.finditer(normalized):
        if match.start() != position:
            return None
        seconds += float(match.group("number")) * multipliers[match.group("unit")]
        position = match.end()
    return seconds if position == len(normalized) else None


def _nonnegative_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _bounded_int(
    value: object,
    minimum: int,
    maximum: int,
    default: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if minimum <= value <= maximum else default


def _bounded_float(
    value: object,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if minimum <= number <= maximum else default


# Best-effort scheduler state
def _load_state(state_path: Path | None, scope_sha256: str) -> dict[str, object] | None:
    if state_path is None:
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != _STATE_SCHEMA:
        return None
    if payload.get("scope_sha256") != scope_sha256:
        return None
    return payload


def _persist_state(state_path: Path, payload: Mapping[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, state_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# 2) Wrapper function
def _run_adaptive_operations(
    jobs: Sequence[Job],
    run_job: Callable[[Job, "_AdaptiveProviderCaller"], Result],
    model: str,
    scope: str,
    config: _BatchConfig,
    on_result: Callable[[int, Result], None] | None = None,
) -> list[Result]:
    missing = object()
    if not jobs:
        return []
    scheduler = _AdaptiveScheduler.create(
        maximum_concurrency=config.max_concurrency,
        model=model,
        scope=scope,
        state_path=config.state_path,
    )
    caller = _AdaptiveProviderCaller(
        scheduler=scheduler,
        max_retries=config.max_retries,
    )
    results_by_index: list[object] = [missing] * len(jobs)
    active: dict[Future[Result], int] = {}
    next_index = 0
    executor = ThreadPoolExecutor(
        max_workers=config.max_concurrency,
        thread_name_prefix=f"llm-patterns-{scope}",
    )

    def submit_available() -> None:
        nonlocal next_index
        while next_index < len(jobs) and len(active) < config.max_concurrency:
            future = executor.submit(run_job, jobs[next_index], caller)
            active[future] = next_index
            next_index += 1

    failure: BaseException | None = None
    try:
        submit_available()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index = active.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    failure = failure or error
                else:
                    results_by_index[index] = result
                    if on_result is not None:
                        on_result(index, result)
            if failure is not None:
                raise failure
            submit_available()
    except BaseException as error:
        failure = failure or error
        caller.abort()
        for future in active:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if failure is not None:
        for future, index in list(active.items()):
            if future.cancelled():
                continue
            try:
                result = future.result()
            except BaseException:
                continue
            results_by_index[index] = result
            if on_result is not None:
                on_result(index, result)
        raise failure

    scheduler.persist()
    if any(result is missing for result in results_by_index):
        raise RuntimeError("Adaptive operation batch did not produce every result")
    ordered_results = list(results_by_index)
    return ordered_results  # type: ignore[return-value]
