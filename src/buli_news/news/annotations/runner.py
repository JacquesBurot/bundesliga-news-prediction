"""Orchestrate resume-safe, concurrent local news annotation runs."""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from ._validation import require_non_empty_text
from .config import AnnotationConfig
from .ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_NUM_CTX,
    InvalidOllamaAnnotationError,
    get_ollama_model_digest,
    request_ollama_annotation,
)
from .results import (
    append_jsonl_record,
    build_annotation_failure_row,
    build_annotation_id,
    build_annotation_row,
    read_existing_annotation_failures,
    read_existing_annotations,
)
from .tasks import validate_tasks


@dataclass(frozen=True)
class OllamaAnnotationRun:
    """Summary of one resume-safe local annotation run."""

    selected_count: int
    annotated_count: int
    failed_count: int
    skipped_count: int
    deferred_failure_count: int
    model_digest: str


def annotate_news_tasks(
    tasks: list[dict[str, Any]],
    config: AnnotationConfig,
    output_path: Path,
    failure_output_path: Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    limit: int | None = None,
    task_id: str | None = None,
    retry_failures_only: bool = False,
    timeout_seconds: float = 300.0,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    workers: int = 1,
    append_annotation: Callable[[dict[str, Any], Path], None] | None = None,
    append_failure: Callable[[dict[str, Any], Path], None] | None = None,
) -> OllamaAnnotationRun:
    """Run annotations while exclusively owning their append-only outputs."""
    with annotation_output_lock(output_path):
        return _annotate_news_tasks_unlocked(
            tasks=tasks,
            config=config,
            output_path=output_path,
            failure_output_path=failure_output_path,
            model=model,
            base_url=base_url,
            limit=limit,
            task_id=task_id,
            retry_failures_only=retry_failures_only,
            timeout_seconds=timeout_seconds,
            num_ctx=num_ctx,
            workers=workers,
            append_annotation=append_annotation,
            append_failure=append_failure,
        )


def _annotate_news_tasks_unlocked(
    tasks: list[dict[str, Any]],
    config: AnnotationConfig,
    output_path: Path,
    failure_output_path: Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    limit: int | None = None,
    task_id: str | None = None,
    retry_failures_only: bool = False,
    timeout_seconds: float = 300.0,
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
    workers: int = 1,
    append_annotation: Callable[[dict[str, Any], Path], None] | None = None,
    append_failure: Callable[[dict[str, Any], Path], None] | None = None,
) -> OllamaAnnotationRun:
    """Run validated structured annotations through a local Ollama server."""
    validate_tasks(tasks)
    mismatched_schema_ids = sorted(
        {
            task["annotation_schema_id"]
            for task in tasks
            if task["annotation_schema_id"] != config.schema_id
        }
    )
    if mismatched_schema_ids:
        msg = (
            "Annotation tasks were built for different schema IDs: "
            f"{mismatched_schema_ids}; current config uses {config.schema_id!r}."
        )
        raise ValueError(msg)
    require_non_empty_text(model, "Ollama model")
    require_non_empty_text(base_url, "Ollama base URL")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        msg = "Annotation limit must be a positive integer."
        raise ValueError(msg)
    if timeout_seconds <= 0:
        msg = "Ollama timeout must be positive."
        raise ValueError(msg)
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0:
        msg = "Ollama context size must be a positive integer."
        raise ValueError(msg)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        msg = "Annotation worker count must be a positive integer."
        raise ValueError(msg)
    if task_id is not None:
        selected = [task for task in tasks if task["task_id"] == task_id]
        if not selected:
            msg = f"Unknown annotation task ID {task_id!r}."
            raise ValueError(msg)
    else:
        selected = list(tasks)

    existing = read_existing_annotations(output_path)
    existing_ids = {row["annotation_id"] for row in existing}
    existing_failures = read_existing_annotation_failures(failure_output_path)
    failure_sequences = Counter(
        row["annotation_id"] for row in existing_failures
    )
    deferred_annotation_ids = set(failure_sequences)
    append_record = append_annotation or append_jsonl_record
    append_failure_record = append_failure or append_jsonl_record
    client_base_url = base_url.rstrip("/")
    with httpx.Client(base_url=client_base_url, timeout=timeout_seconds) as client:
        model_digest = get_ollama_model_digest(client=client, model=model)

    pending = []
    skipped_count = 0
    deferred_failure_count = 0
    for task in selected:
        annotation_id = build_annotation_id(
            task_id=task["task_id"],
            config_sha256=config.sha256,
            model_digest=model_digest,
        )
        if annotation_id in existing_ids:
            skipped_count += 1
            continue
        is_deferred_failure = annotation_id in deferred_annotation_ids
        if retry_failures_only:
            if not is_deferred_failure:
                continue
        elif is_deferred_failure:
            deferred_failure_count += 1
            continue
        pending.append((task, annotation_id))
        if limit is not None and len(pending) >= limit:
            break

    annotated_count = 0
    failed_count = 0

    def persist_outcome(
        task: dict[str, Any],
        annotation_id: str,
        outcome: tuple[dict[str, Any], dict[str, Any]] | InvalidOllamaAnnotationError,
    ) -> None:
        nonlocal annotated_count, failed_count
        if isinstance(outcome, InvalidOllamaAnnotationError):
            failure_sequences[annotation_id] += 1
            failure_row = build_annotation_failure_row(
                task=task,
                error=outcome,
                config=config,
                model=model,
                model_digest=model_digest,
                annotation_id=annotation_id,
                failure_sequence=failure_sequences[annotation_id],
            )
            append_failure_record(failure_row, failure_output_path)
            deferred_annotation_ids.add(annotation_id)
            failed_count += 1
            return
        response, metadata = outcome
        row = build_annotation_row(
            task=task,
            response=response,
            metadata=metadata,
            config=config,
            model=model,
            model_digest=model_digest,
            annotation_id=annotation_id,
        )
        append_record(row, output_path)
        existing_ids.add(annotation_id)
        annotated_count += 1

    if workers == 1:
        with httpx.Client(base_url=client_base_url, timeout=timeout_seconds) as client:
            for task, annotation_id in pending:
                try:
                    outcome = request_ollama_annotation(
                        client=client,
                        task=task,
                        config=config,
                        model=model,
                        num_ctx=num_ctx,
                    )
                except InvalidOllamaAnnotationError as exc:
                    outcome = exc
                persist_outcome(task, annotation_id, outcome)
    else:
        def request_with_own_client(
            task: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]] | InvalidOllamaAnnotationError:
            with httpx.Client(
                base_url=client_base_url,
                timeout=timeout_seconds,
            ) as client:
                try:
                    return request_ollama_annotation(
                        client=client,
                        task=task,
                        config=config,
                        model=model,
                        num_ctx=num_ctx,
                    )
                except InvalidOllamaAnnotationError as exc:
                    return exc

        pending_iterator = iter(pending)
        in_flight: dict[Future[Any], tuple[dict[str, Any], str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in range(min(workers, len(pending))):
                task, annotation_id = next(pending_iterator)
                future = executor.submit(request_with_own_client, task)
                in_flight[future] = (task, annotation_id)
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    task, annotation_id = in_flight.pop(future)
                    persist_outcome(task, annotation_id, future.result())
                    try:
                        next_task, next_annotation_id = next(pending_iterator)
                    except StopIteration:
                        continue
                    next_future = executor.submit(request_with_own_client, next_task)
                    in_flight[next_future] = (next_task, next_annotation_id)

    return OllamaAnnotationRun(
        selected_count=len(pending),
        annotated_count=annotated_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        deferred_failure_count=deferred_failure_count,
        model_digest=model_digest,
    )


@contextmanager
def annotation_output_lock(output_path: Path):
    """Hold an OS-level lock so two processes cannot append the same output."""
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    if lock_file.seek(0, os.SEEK_END) == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        msg = (
            f"Another annotation process already owns output {output_path}. "
            "Wait for it to finish instead of starting a concurrent resume run."
        )
        raise ValueError(msg) from exc
    try:
        yield
    finally:
        try:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
