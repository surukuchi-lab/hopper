"""
Module: hopper.utils.profiling

Developer: ehtkarim
Date: April 29, 2026

Collects timing events and writes concise runtime profiles for pipeline stages.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


@dataclass
class ProfileEvent:
    name: str
    seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileNote:
    name: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationProfiler:
    """Lightweight timer and pathway logger for Hopper ``.out`` summaries."""

    enabled: bool = True
    events: list[ProfileEvent] = field(default_factory=list)
    notes: list[ProfileNote] = field(default_factory=list)

    @contextmanager
    def step(self, name: str, **details: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.events.append(ProfileEvent(name=name, seconds=perf_counter() - start, details=dict(details)))

    def add(self, name: str, seconds: float, **details: Any) -> None:
        if self.enabled:
            self.events.append(ProfileEvent(name=name, seconds=float(seconds), details=dict(details)))

    def add_note(self, name: str, **details: Any) -> None:
        if self.enabled:
            self.notes.append(ProfileNote(name=name, details=dict(details)))

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.9g}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(SimulationProfiler._format_value(v) for v in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(f"{k}: {SimulationProfiler._format_value(v)}" for k, v in sorted(value.items()))
            return "{" + items + "}"
        return str(value)

    @classmethod
    def _detail_string(cls, details: dict[str, Any]) -> str:
        if not details:
            return ""
        return " " + " ".join(f"{k}={cls._format_value(v)}" for k, v in sorted(details.items()))

    def summary_lines(self) -> list[str]:
        total = sum(ev.seconds for ev in self.events)
        lines = ["Hopper runtime profile", f"total_recorded_s: {total:.9g}", ""]

        if self.notes:
            lines.append("Simulation pathway details")
            for note in self.notes:
                lines.append(f"[{note.name}]")
                for key, value in sorted(note.details.items()):
                    lines.append(f"  {key}: {self._format_value(value)}")
            lines.append("")

        lines.append("Timed steps")
        for idx, ev in enumerate(self.events, start=1):
            frac = 0.0 if total <= 0.0 else ev.seconds / total
            detail = self._detail_string(ev.details)
            lines.append(f"{idx:03d} {ev.name:42s} {ev.seconds:12.6f} s  {frac:7.2%}{detail}")
        if self.events:
            slowest = max(self.events, key=lambda ev: ev.seconds)
            lines.extend(["", f"largest_step: {slowest.name} ({slowest.seconds:.6f} s)"])
        return lines

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.summary_lines()) + "\n")
