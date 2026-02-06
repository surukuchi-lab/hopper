from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class LoopCoil:
    """A single circular current loop centered on the z-axis."""
    radius_m: float
    z0_m: float
    current_A: float
    turns: int = 1
    name: str = ""


def _parse_number_with_units(s: str) -> float:
    """
    Parse a number possibly with a simple unit suffix.

    Supported:
      - distance: m, cm, mm
      - current: A
    Examples: "12.3mm", "0.5 m", "200A"
    """
    s = str(s).strip()
    if s == "":
        raise ValueError("empty string")

    import re

    m = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$", s)
    if not m:
        raise ValueError(f"Could not parse number: {s!r}")
    val = float(m.group(1))
    unit = m.group(2)

    if unit in ("", None):
        return val

    unit = unit.lower()
    if unit == "m":
        return val
    if unit == "cm":
        return val * 1e-2
    if unit == "mm":
        return val * 1e-3
    if unit == "a":
        return val
    # unknown: assume SI
    return val


def _get_attr_or_child(elem: ET.Element, keys: Iterable[str]) -> Optional[str]:
    # attributes
    for k in keys:
        if k in elem.attrib:
            return elem.attrib[k]
    # direct child elements
    for child in elem:
        tag = child.tag.split("}")[-1]  # drop namespace
        if tag in keys and child.text is not None:
            return child.text
    return None


def load_loop_coils_from_xml(xml_path: str | Path) -> List[LoopCoil]:
    """
    Attempt to parse a trap coil XML file into a list of LoopCoil objects.

    This parser is intentionally permissive: it walks all elements and tries to interpret any
    element that contains (radius, z, current) as a loop.

    Because coil XML schemas vary between projects, you may need to adjust the key lists below
    to match your specific trap .xml format.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    radius_keys = ("r", "radius", "R", "radius_m", "coil_radius", "loop_radius")
    z_keys = ("z", "z0", "z_m", "Z", "center_z", "z_center")
    current_keys = ("I", "i", "current", "current_A", "amps", "I_A")
    turns_keys = ("turns", "n_turns", "N", "num_turns")

    coils: List[LoopCoil] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        r_s = _get_attr_or_child(elem, radius_keys)
        z_s = _get_attr_or_child(elem, z_keys)
        i_s = _get_attr_or_child(elem, current_keys)

        if r_s is None or z_s is None or i_s is None:
            continue

        try:
            radius_m = _parse_number_with_units(r_s)
            z0_m = _parse_number_with_units(z_s)
            current_A = _parse_number_with_units(i_s)
        except Exception:
            continue

        turns_s = _get_attr_or_child(elem, turns_keys)
        turns = 1
        if turns_s is not None:
            try:
                turns = int(float(turns_s))
            except Exception:
                turns = 1

        name = elem.attrib.get("name", tag)
        coils.append(LoopCoil(radius_m=radius_m, z0_m=z0_m, current_A=current_A, turns=turns, name=name))

    if not coils:
        tags = sorted({e.tag.split("}")[-1] for e in root.iter()})
        raise ValueError(
            f"No loop-like elements found in {xml_path}. "
            f"Available tags include: {tags[:30]}{'...' if len(tags)>30 else ''}. "
            "You may need to customize the key lists in load_loop_coils_from_xml()."
        )

    return coils
