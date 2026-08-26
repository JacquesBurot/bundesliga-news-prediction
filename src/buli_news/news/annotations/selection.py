"""Select deterministic, stratified annotation pilots."""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import Counter
from typing import Any

from .tasks import validate_tasks


NEWS_ANNOTATION_PILOT_SELECTION_VERSION = 1


def select_stratified_annotation_tasks(
    tasks: list[dict[str, Any]],
    size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic pilot balanced across context and text length."""
    validate_tasks(tasks)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        msg = "Pilot size must be a positive integer."
        raise ValueError(msg)
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "Pilot seed must be an integer."
        raise ValueError(msg)
    if size >= len(tasks):
        return list(tasks)

    lengths = sorted(len(task["body"]) for task in tasks)
    boundaries = tuple(
        lengths[round((len(lengths) - 1) * quantile)]
        for quantile in (0.25, 0.5, 0.75)
    )

    task_metadata = {
        task["task_id"]: (
            bisect_right(boundaries, len(task["body"])),
            hashlib.sha256(
                f"{seed}:{task['task_id']}".encode("utf-8")
            ).hexdigest(),
        )
        for task in tasks
    }

    team_counts: Counter[int] = Counter()
    side_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    matchday_tasks: dict[int, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        matchday_tasks.setdefault(task["matchday"], {})[task["task_id"]] = task
    matchdays = sorted(matchday_tasks)
    quotas: Counter[int] = Counter()
    if size < len(matchdays):
        if size == 1:
            quotas[matchdays[seed % len(matchdays)]] = 1
        else:
            for index in range(size):
                position = round(index * (len(matchdays) - 1) / (size - 1))
                quotas[matchdays[position]] += 1
    else:
        base, remainder = divmod(size, len(matchdays))
        for matchday in matchdays:
            quotas[matchday] = base
        offset = seed % len(matchdays)
        for index in range(remainder):
            quotas[matchdays[(offset + index) % len(matchdays)]] += 1

    selected = []
    while any(quotas.values()):
        for matchday in matchdays:
            if quotas[matchday] <= 0:
                continue
            candidates = matchday_tasks[matchday]
            task = min(
                candidates.values(),
                key=lambda candidate: (
                    team_counts[candidate["target_team_id"]],
                    content_counts[candidate["content_id"]],
                    length_counts[task_metadata[candidate["task_id"]][0]],
                    side_counts[candidate["side"]],
                    source_counts[candidate["source_host"]],
                    task_metadata[candidate["task_id"]][1],
                ),
            )
            selected.append(task)
            candidates.pop(task["task_id"])
            quotas[matchday] -= 1
            team_counts[task["target_team_id"]] += 1
            side_counts[task["side"]] += 1
            length_counts[task_metadata[task["task_id"]][0]] += 1
            source_counts[task["source_host"]] += 1
            content_counts[task["content_id"]] += 1

    return selected
