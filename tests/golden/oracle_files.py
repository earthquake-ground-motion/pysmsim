"""Readers for the golden files written by the Fortran oracle.

The golden files and how they were made are described in README.md in this
folder. These helpers are for the tests only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysmsim.spectrum.source import (
    SOURCE_MODELS,
    Joyner1984,
    SingleCorner,
    SourceSpectrum,
    TwoCornerAdditive,
    TwoCornerMultiplicative,
)
from pysmsim.spectrum.stress import (
    HoerlStress,
    LogLinearStress,
    Stafford2022Stress,
    StressModel,
)

GOLDEN = Path(__file__).parent / "double"
ORACLE_INPUTS = Path(__file__).parents[2] / "oracle" / "inputs"


@dataclass(frozen=True)
class FactorFile:
    """Contents of a factor_dump output file."""

    header: dict[str, float]
    columns: dict[str, npt.NDArray[np.float64]]


def read_factor_file(path: Path) -> FactorFile:
    """Read a file written by oracle/programs/factor_dump.for."""
    header: dict[str, float] = {}
    names: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("!"):
            key, sep, value = line[1:].partition("=")
            if sep and key.strip() != "params":
                header[key.strip()] = float(value)
        elif not names:
            names = line.split()
        elif line.strip():
            rows.append([float(v) for v in line.split()])
    data = np.array(rows)
    return FactorFile(header=header, columns={n: data[:, i] for i, n in enumerate(names)})


# Source settings of the shipped .params files used for the golden files:
# shear velocity (km/s) and the source model. All use source 1 with
# 0.5/0.5 duration weights.
SHIPPED_SOURCES: dict[str, tuple[float, SourceSpectrum]] = {
    "ena.scf.bt15scr_fff.bs11_atten.bt15_dp.dmb_3kps_amps_aoi_00.k0.006.bt15e_drms": (
        3.6,
        SingleCorner(stress=LogLinearStress(stress_ref=171.9, magnitude_ref=6.5)),
    ),
    "ofr": (3.6, SingleCorner(stress=LogLinearStress(stress_ref=80.0, magnitude_ref=6.5))),
    "stafford.convenience_model": (
        3.5,
        SingleCorner(stress=Stafford2022Stress(c1=5.07, c2=0.6451)),
    ),
    "stafford.optimal_model": (
        3.5,
        SingleCorner(stress=Stafford2022Stress(c1=4.599, c2=0.4624)),
    ),
    "wna.scf.bt15acr_fff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms": (
        3.5,
        SingleCorner(stress=LogLinearStress(stress_ref=250.0, magnitude_ref=6.5)),
    ),
    "wna.scf.xfff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms": (
        3.5,
        SingleCorner(stress=LogLinearStress(stress_ref=250.0, magnitude_ref=6.5)),
    ),
}

# The synthetic source cases are variants of ofr.params
SYNTHETIC_SHEAR_VELOCITY = 3.6

# The Fortran numsource of each class, for reading source_cases.txt
_FORTRAN_SOURCES = dict(enumerate(SOURCE_MODELS.values(), start=1))


def _stress(line: list[str]) -> StressModel:
    """The stress model of an SMSIM stress line (istress_model, then values)."""
    s = [float(v) for v in line[1:]]
    match int(line[0]):
        case 1:
            return LogLinearStress(stress_ref=s[0], dlog10_stress_dm=s[1], magnitude_ref=s[2])
        case 2:
            return HoerlStress(c1=s[0], c2=s[1], c3=s[2])
        case _:
            return Stafford2022Stress(c1=s[0], c2=s[1])


def read_source_cases() -> dict[str, SourceSpectrum]:
    """The synthetic source cases of oracle/inputs/source_cases.txt as source models.

    Each case gives the shape, stress and scaling lines of an SMSIM params
    file; this turns them into pysmsim source objects.
    """
    cases: dict[str, SourceSpectrum] = {}
    for line in (ORACLE_INPUTS / "source_cases.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, shape, stress_line, scaling = (part.split() for part in line.split("|"))
        cls = _FORTRAN_SOURCES[int(shape[0])]
        pf_a, pd_a, pf_b, pd_b = (float(v) for v in shape[1:5])
        stress = _stress(stress_line)
        c = [float(v) for v in scaling]
        source: SourceSpectrum
        if cls is SingleCorner:
            source = SingleCorner(stress=stress, pf_a=pf_a, pd_a=pd_a)
        elif cls is Joyner1984:
            # Log-linear stress line: stressc, dlsdm, amagc, fbdfa
            source = Joyner1984(
                stress=stress,
                magnitude_hinge=float(stress_line[3]),
                fb_over_fa=float(stress_line[4]),
            )
        elif cls is TwoCornerMultiplicative:
            source = TwoCornerMultiplicative(
                stress=stress,
                log10_fa=(c[0], c[1], c[2]),
                pf_a=pf_a,
                pd_a=pd_a,
                pf_b=pf_b,
                pd_b=pd_b,
            )
        elif cls is TwoCornerAdditive:
            source = TwoCornerAdditive(
                stress=stress,
                log10_fa=(c[0], c[1], c[2]),
                log10_eps=(c[3], c[4], c[5]),
                pf_a=pf_a,
                pd_a=pd_a,
                pf_b=pf_b,
                pd_b=pd_b,
            )
        else:  # no parameters
            source = cls()
        cases[name[0]] = source
    return cases
