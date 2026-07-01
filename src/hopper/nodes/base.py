"""
Module: hopper.nodes.base

Developer: ehtkarim
Date: April 29, 2026

Defines the lightweight node protocol and execution wrapper used by the simulation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


class Node(Protocol):
    name: str
    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]: ...


@dataclass
class BaseNode:
    name: str

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
