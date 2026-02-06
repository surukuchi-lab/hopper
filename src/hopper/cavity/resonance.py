from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ResonanceCurve:
    """
    A simple 1D frequency response curve loaded from a ROOT file (via uproot).

    The curve is used as a multiplicative amplitude factor response(fc).

    If no ROOT file is provided or uproot is unavailable, response(fc)=1.
    """
    f_hz: Optional[np.ndarray] = None
    response: Optional[np.ndarray] = None
    normalize_to_peak: bool = True

    def __post_init__(self) -> None:
        if self.f_hz is None or self.response is None:
            self.f_hz = None
            self.response = None
            return

        f = np.asarray(self.f_hz, float)
        r = np.asarray(self.response, float)
        if f.ndim != 1 or r.ndim != 1 or f.size != r.size:
            raise ValueError("ResonanceCurve expects 1D arrays of equal length")
        order = np.argsort(f)
        f = f[order]
        r = r[order]
        if self.normalize_to_peak and r.size > 0:
            mx = float(np.nanmax(np.abs(r)))
            if mx > 0:
                r = r / mx
        self.f_hz = f
        self.response = r

    @classmethod
    def unity(cls) -> "ResonanceCurve":
        return cls(f_hz=None, response=None)

    @classmethod
    def from_root(
        cls,
        root_file: str | Path,
        object_name: Optional[str] = None,
        *,
        normalize_to_peak: bool = True,
    ) -> "ResonanceCurve":
        """
        Load a resonance curve from ROOT.

        Supported objects:
          - TGraph / TGraphErrors (x,y arrays)
          - TH1 (bin centers, bin values)

        If object_name is None, the first suitable object is used.
        """
        try:
            import uproot  # type: ignore
        except Exception as e:
            raise ImportError("uproot is required to load ROOT resonance curves") from e

        root_file = Path(root_file)
        if not root_file.exists():
            raise FileNotFoundError(root_file)

        with uproot.open(root_file) as f:
            obj = None
            if object_name is not None:
                obj = f[object_name]
            else:
                for k in f.keys():
                    try:
                        o = f[k]
                    except Exception:
                        continue
                    cname = getattr(o, "classname", "")
                    if "TGraph" in cname or cname.startswith("TH1"):
                        obj = o
                        break
            if obj is None:
                raise ValueError(f"No suitable TGraph/TH1 found in {root_file}")

            cname = getattr(obj, "classname", "")
            if "TGraph" in cname:
                arr = obj.to_numpy()
                if isinstance(arr, tuple) and len(arr) >= 2:
                    x = np.asarray(arr[0], float)
                    y = np.asarray(arr[1], float)
                else:
                    raise ValueError(f"Could not extract arrays from {cname}")
            elif cname.startswith("TH1"):
                values, edges = obj.to_numpy()
                centers = 0.5 * (edges[:-1] + edges[1:])
                x = np.asarray(centers, float)
                y = np.asarray(values, float)
            else:
                raise ValueError(f"Unsupported ROOT object class: {cname}")

        return cls(f_hz=x, response=y, normalize_to_peak=normalize_to_peak)

    def __call__(self, f_hz: np.ndarray | float) -> np.ndarray:
        """
        Evaluate response at frequency f_hz.

        If no curve loaded, returns ones.
        """
        if self.f_hz is None or self.response is None:
            return np.ones_like(np.asarray(f_hz, float), dtype=float)

        f = np.asarray(f_hz, float)
        f0 = float(self.f_hz[0])
        f1 = float(self.f_hz[-1])
        fq = np.clip(f, f0, f1)
        return np.interp(fq, self.f_hz, self.response)
