"""Readers for the golden files written by the Fortran oracle.

The golden files and how they were made are described in README.md in this
folder. These helpers are for the tests only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysmsim.spectrum.attenuation import PiecewiseQ, QModel, Stafford2022Q
from pysmsim.spectrum.filters import (
    Filters,
    HighCutFilter,
    LowCutFilter,
    NoHighCut,
    QuarterCosineHighCut,
    RaisedCosineHighCut,
)
from pysmsim.spectrum.finite_fault import (
    FiniteFaultFactor,
    Generalized,
    NoFiniteFault,
    OneLine,
    RpsCombination,
    Stafford2022FiniteFault,
    TwoLines,
    TwoLinesSmoothed,
)
from pysmsim.spectrum.site import Site
from pysmsim.spectrum.source import (
    SOURCE_MODELS,
    Joyner1984,
    SingleCorner,
    SourceSpectrum,
    TwoCornerAdditive,
    TwoCornerMultiplicative,
)
from pysmsim.spectrum.spreading import (
    Atkinson2011,
    Atkinson2012,
    AtkinsonBoore2014,
    AtkinsonBoore2014R10,
    BooreTrial,
    GeometricalSpreading,
    Hinged,
    SpreadingSegment,
    StaffordGeneralized,
    StaffordOptimal,
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


# --- Path: finite-fault factor, geometrical spreading and Q ----------------


@dataclass(frozen=True)
class PathModels:
    """The path models of one params file."""

    finite_fault: FiniteFaultFactor
    spreading: GeometricalSpreading
    q: QModel


def _finite_fault(line: list[str]) -> FiniteFaultFactor:
    """The finite-fault model of an SMSIM finite-fault line."""
    iflag, nlines = int(line[0]), int(line[1])
    c1, c2, c3, c4, dm = (float(v) for v in line[2:7])
    if iflag == 0:
        return NoFiniteFault()
    combination = RpsCombination.QUADRATURE if iflag == 1 else RpsCombination.SUM
    match nlines:
        case 1:
            return OneLine(c1=c1, c2=c2, combination=combination)
        case 2:
            return TwoLines(c1=c1, c2=c2, c3=c3, c4=c4, combination=combination)
        case 3:
            return TwoLinesSmoothed(c1=c1, c2=c2, c3=c3, c4=c4, delta_m=dm, combination=combination)
        case 999:
            return Generalized(c1=c1, c2=c2, c3=c3, c4=c4, h_ratio=dm, combination=combination)
        case _:
            return Stafford2022FiniteFault(
                h_alpha=c1,
                h_beta=c2,
                h_gamma=c3,
                h_delta=c4,
                h_epsilon=dm,
                combination=combination,
            )


def _spreading(lines: list[list[str]]) -> GeometricalSpreading:
    """The spreading model of SMSIM's spreading lines (option, r1 line, r_ref, nsegs, ...)."""
    option = int(lines[0][0])
    r1_line = [float(v) for v in lines[1][:6]]
    r_ref = float(lines[2][0])
    rest = [[float(v) for v in line] for line in lines[4:]]
    match option:
        case 0:
            segments = tuple(
                SpreadingSegment(r_start=s[0], a=s[1], b=s[2], m_ref=s[3]) for s in rest
            )
            return Hinged(segments=segments, r_ref=r_ref)
        case 1:
            return Atkinson2011()
        case 2:
            r1, lf, hf, far, ft1, ft2 = r1_line
            return BooreTrial(r1=r1, slope_lf=lf, slope_hf=hf, slope_far=far, ft1=ft1, ft2=ft2)
        case 3:
            return Atkinson2012(r_ref=r_ref)
        case 4:
            return AtkinsonBoore2014(h=r1_line[0], r_ref=r_ref)
        case 5:
            return AtkinsonBoore2014R10(h=r1_line[0], r_ref=r_ref)
        case 6:
            g1, rt, g2, grat = rest[0][:4]
            return StaffordGeneralized(gamma1=g1, r_t=rt, gamma2=g2, g_ratio=grat, r_ref=r_ref)
        case _:
            g1, rt, g2 = rest[0][:3]
            return StaffordOptimal(gamma1=g1, r_t=rt, gamma2=g2, r_ref=r_ref)


def _q(line: list[str]) -> QModel:
    """The Q model of an SMSIM Q line (fr1, qr1, s1, ft1, ft2, fr2, qr2, s2, c_q)."""
    fr1, qr1, s1, ft1, ft2, fr2, qr2, s2, c_q = (float(v) for v in line[:9])
    if fr1 < 0:
        return Stafford2022Q(q0=qr1, eta_a=s1, eta_b=ft1, eta_g=ft2, c_q=c_q)
    return PiecewiseQ(fr1=fr1, qr1=qr1, s1=s1, ft1=ft1, ft2=ft2, fr2=fr2, qr2=qr2, s2=s2, c_q=c_q)


def path_models(finite_fault: str, spreading: str, q: str) -> PathModels:
    """Path models from SMSIM params-file lines; spreading lines are separated by ";"."""
    return PathModels(
        finite_fault=_finite_fault(finite_fault.split()),
        spreading=_spreading([line.split() for line in spreading.split(";")]),
        q=_q(q.split()),
    )


# The path lines of the shipped .params files (finite fault | spreading | Q)
SHIPPED_PATHS: dict[str, PathModels] = {
    "ena.scf.bt15scr_fff.bs11_atten.bt15_dp.dmb_3kps_amps_aoi_00.k0.006.bt15e_drms": path_models(
        "1 3 -1.8276 0.4300 -0.5126 0.2350 2.0",
        "0;10.0 0.0 0.0 0.0 0.0 0.0;1.0;2;1.0 -1.0 0.0 6.5;50.0 -0.5 0.0 6.5",
        "1.0 410 0.50 1.0 1.0 1.0 410 0.50 3.5",
    ),
    "ofr": path_models(
        "0 0 0.0 0.0 0.0 0.0 0.0",
        "0;10.0 0.0 0.0 0.0 0.0 0.0;1.0;3;1.0 -1.0 0.0 6.5;70.0 0.0 0.0 6.5;130.0 -0.5 0.0 6.5",
        "0.1 275 -2.0 0.2 0.6 1.0 88.0 0.9 3.6",
    ),
    "stafford.convenience_model": path_models(
        "2 1000 -0.7771 0.4768 1.1513 3.418 7.088",
        "0;10.0 0.0 0.0 0.0 0.0 0.0;1.0;3;1.0 -1.1680 0.0 6.5;25.0 -0.9293 0.0 6.5;"
        "85.0 -0.5 0.0 6.5",
        "1.0 183.7 0.7077 1.0 1.0 1.0 183.7 0.7077 3.5",
    ),
    "stafford.optimal_model": path_models(
        "2 1000 -0.8712 0.4451 1.1513 5.0948 7.2725",
        "7;10.0 0.0 0.0 0.0 0.0 0.0;1.0;0;-1.1611 50.0 -0.5 0.0",
        "-1.0 205.4 0.6884 0.1354 5.1278 0.0 0.0 0.0 3.5",
    ),
    "wna.scf.bt15acr_fff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms": path_models(
        "1 3 -1.7200 0.4300 -0.405 0.2350 2.0",
        "0;10.0 0.0 0.0 0.0 0.0 0.0;1.0;2;1.0 -1.0 0.0 6.5;40.0 -0.5 0.0 6.5",
        "1.0 180 0.45 1.0 1.0 1.0 180 0.45 3.5",
    ),
    "wna.scf.xfff.rea99_atten.bt14_dp.bj97gr760_amps.k0.035.bt15w_drms": path_models(
        "0 0 0.0 0.0 0.0 0.0 0.0",
        "0;10.0 0.0 0.0 0.0 0.0 0.0;1.0;2;1.0 -1.0 0.0 6.5;40.0 -0.5 0.0 6.5",
        "1.0 180 0.45 1.0 1.0 1.0 180 0.45 3.5",
    ),
}


def read_path_cases() -> dict[str, PathModels]:
    """The synthetic path cases of oracle/inputs/path_cases.txt."""
    cases: dict[str, PathModels] = {}
    for line in (ORACLE_INPUTS / "path_cases.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, ff, spreading, q = (part.strip() for part in line.split("|"))
        cases[name] = path_models(ff, spreading, q)
    return cases


# --- Site and filters -----------------------------------------------------


@dataclass(frozen=True)
class SiteModels:
    """The site terms and filters of one params file.

    ``nslope`` is the low-cut value as written in the file, before SMSIM
    rounds it down to a multiple of 4.
    """

    site: Site
    filters: Filters
    nslope: int


def _high_cut(line: list[str]) -> HighCutFilter:
    """The high-cut filter of an SMSIM high-cut line (itype_hcfilt, fhc1, fhc2, eta)."""
    itype = int(line[0])
    f1, f2, eta = (float(v) for v in line[1:4])
    match itype:
        case 0:
            return NoHighCut()
        case 1:
            return RaisedCosineHighCut(f1=f1, f2=f2, eta=eta)
        case _:
            return QuarterCosineHighCut(f1=f1, f2=f2, eta=eta)


def site_models(points: list[list[str]], dimin: str, low_cut: str, high_cut: str) -> SiteModels:
    """Site and filters from SMSIM params-file lines; ``points`` are (f, amp) pairs."""
    fm, kappa, dkappa, mref = (float(v) for v in dimin.split()[:4])
    fcut_s, nslope_s = low_cut.split()[:2]
    nslope = int(nslope_s)
    # SMSIM rounds nslope down to a multiple of 4 (and up to 4)
    order = max(4, 4 * (nslope // 4)) // 2
    site = Site(
        amp_frequencies=tuple(float(p[0]) for p in points),
        amplifications=tuple(float(p[1]) for p in points),
        kappa0=kappa,
        dkappa_dmag=dkappa,
        kappa_magnitude_ref=mref,
        fmax=fm,
    )
    filters = Filters(
        low_cut=LowCutFilter(fcut=float(fcut_s), order=order),
        high_cut=_high_cut(high_cut.split()),
    )
    return SiteModels(site=site, filters=filters, nslope=nslope)


def read_shipped_sites() -> dict[str, SiteModels]:
    """The site and filter lines of the shipped .params files (golden site_lines/)."""
    sites: dict[str, SiteModels] = {}
    for path in sorted((GOLDEN / "site_lines").glob("*.txt")):
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        namps = int(lines[0].split()[0])
        points = [line.split()[:2] for line in lines[1 : namps + 1]]
        dimin, low_cut, high_cut = lines[namps + 1 : namps + 4]
        sites[path.stem] = site_models(points, dimin, low_cut, high_cut)
    return sites


def read_site_cases() -> dict[str, SiteModels]:
    """The synthetic site and filter cases of oracle/inputs/site_cases.txt."""
    cases: dict[str, SiteModels] = {}
    for line in (ORACLE_INPUTS / "site_cases.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, amps, dimin, low_cut, high_cut = (part.strip() for part in line.split("|"))
        points = [p.split() for p in amps.split(";")]
        cases[name] = site_models(points, dimin, low_cut, high_cut)
    return cases
