"""Converting SMSIM ``.params`` and ``tmrsk_loop_rv_drvr`` ``.ctl`` files to TOML.

:func:`read_params` reads an SMSIM parameter file (the format dated
02/25/21, which SMSIM 2023-03-03 reads) into :class:`~pysmsim.params.Params`,
:func:`read_rv_loop_ctl` reads a control file of ``tmrsk_loop_rv_drvr``
(dated 03/03/23, or 01/22/21 as in the example shipped with SMSIM, which
has no mechanism line), and :func:`convert` turns both into the text of a
TOML run file::

    from pysmsim.io.legacy import convert
    text, notes = convert("ofr.params", "tmrsk_loop_rv_drvr.ctl")

The files are read as SMSIM reads them:

- before each group of values, ``skipcmnt`` skips one line whatever it
  holds, and then every line that starts with ``!`` or ``|``; a skipped
  line that is not a comment is reported in the notes, as it usually
  means a missing comment line (SMSIM then reads the wrong values);
- values are read as Fortran list-directed input: separated by blanks or
  commas, continued on the next lines when a line has too few, the rest of
  the last line ignored, ``n*value`` repeated, ``d`` exponents accepted.

Integer codes become model names (see each model's docstring for its
Fortran number). Where SMSIM behaves differently from pysmsim (for
example the stress model of sources 2, 11 and 12, or the rounding of the
low-cut ``nslope``), the notes say so; they are written as comments at the
top of the run file.
"""

from __future__ import annotations

import re
import textwrap
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import tomli_w

from pysmsim.io.config import SCHEMA_VERSION, params_table, parse_config
from pysmsim.params import (
    Medium,
    NoiseDistribution,
    OscillatorCorrection,
    Params,
    PathDuration,
    RVConfig,
    TDConfig,
    WindowShape,
)
from pysmsim.scenario import MAX_SPECTRAL_POINTS, Mechanism, RjbToRrup
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
    JiArchuleta2020,
    Joyner1984,
    SingleCorner,
    SourceSpectrum,
    TwoCornerAdditive,
    TwoCornerMultiplicative,
    WeightedDuration,
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
from pysmsim.spectrum.stress import HoerlStress, LogLinearStress, Stafford2022Stress, StressModel

__all__ = [
    "CTL_DATES",
    "PARAMS_DATE",
    "LegacyFormatError",
    "RVLoopControl",
    "convert",
    "read_params",
    "read_rv_loop_ctl",
]

PARAMS_DATE = "02/25/21"
"""Date line of the parameter files SMSIM 2023-03-03 reads."""

CTL_DATES = ("03/03/23", "01/22/21")
"""Date lines of the ``tmrsk_loop_rv_drvr`` control files this module reads.

03/03/23 is SMSIM 2023-03-03's; 01/22/21 is that of the example shipped
with it, which has no mechanism line (strike-slip is then assumed).
"""

_SOURCE_NUMBERS = dict(enumerate(SOURCE_MODELS.values(), start=1))
"""Source model class of each Fortran ``numsource`` (1 to 13)."""

_BT15_FILES = {"bt15_wna_acr": "bt15_wna_acr", "bt15_ena_scr": "bt15_ena_scr"}
"""Bundled BT15 table for the start of a ``.pars`` file name."""


class LegacyFormatError(ValueError):
    """An SMSIM input file cannot be read or converted."""


# --- Reading SMSIM files as Fortran does --------------------------------------------


_REPEAT = re.compile(r"^(\d+)\*(.*)$")


class _FortranReader:
    """Reads a text file as SMSIM's ``skipcmnt`` and Fortran ``read`` do."""

    def __init__(self, text: str, name: str) -> None:
        self.lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.name = name
        self.pos = 0
        self.notes: list[str] = []

    def _where(self) -> str:
        return f"{self.name}, line {self.pos}"

    def skipcmnt(self) -> None:
        """Skip one line, then every line starting with ``!`` or ``|``."""
        first = True
        while self.pos < len(self.lines):
            stripped = self.lines[self.pos].strip()
            if first or stripped[:1] in ("!", "|"):
                if first and stripped and stripped[:1] not in ("!", "|"):
                    self.notes.append(
                        f"{self.name}, line {self.pos + 1}: SMSIM skips this line, which is not "
                        f"a comment ({stripped[:40]!r}); a comment line is probably missing "
                        "above it, so the values that follow may be the wrong ones"
                    )
                first = False
                self.pos += 1
            else:
                return

    def line(self) -> str:
        """Read a whole line (Fortran ``read(nu, '(a)')``)."""
        if self.pos >= len(self.lines):
            raise LegacyFormatError(f"{self.name}: the file ends too early")
        self.pos += 1
        return self.lines[self.pos - 1].strip()

    def _tokens(self, what: str) -> Iterator[str]:
        """Yield the values of one list-directed read, continued on the next lines."""
        while True:
            if self.pos >= len(self.lines):
                raise LegacyFormatError(f"{self.name}: the file ends while reading {what}")
            text = self.lines[self.pos]
            self.pos += 1
            for token in text.replace(",", " ").split():
                if token.startswith("/"):
                    raise LegacyFormatError(
                        f"{self._where()}: '/' ends the values for {what} early; give every value"
                    )
                if token.startswith("!"):  # a comment after the values
                    break
                match = _REPEAT.match(token)
                if match:
                    yield from [match.group(2)] * int(match.group(1))
                else:
                    yield token

    def values(self, count: int, what: str) -> list[str]:
        """Read ``count`` values by list-directed input (Fortran ``read(nu, *)``)."""
        tokens = self._tokens(what)
        return [next(tokens) for _ in range(count)]

    def counted(self, what: str) -> list[float]:
        """Read ``n`` then ``n`` values in one statement (``read(nu,*) n, (x(i), i=1,n)``)."""
        tokens = self._tokens(what)
        count = self._int(next(tokens), f"the number of {what}")
        if count < 1:
            raise LegacyFormatError(f"{self._where()}: the number of {what} must be at least 1")
        return [self._float(next(tokens), what) for _ in range(count)]

    def floats(self, count: int, what: str) -> list[float]:
        return [self._float(v, what) for v in self.values(count, what)]

    def ints(self, count: int, what: str) -> list[int]:
        return [self._int(v, what) for v in self.values(count, what)]

    def _float(self, value: str, what: str) -> float:
        try:
            return float(value.lower().replace("d", "e"))
        except ValueError:
            raise LegacyFormatError(
                f"{self._where()}: {value!r} is not a number (reading {what})"
            ) from None

    def _int(self, value: str, what: str) -> int:
        number = self._float(value, what)
        if not number.is_integer():
            raise LegacyFormatError(
                f"{self._where()}: {value!r} is not a whole number (reading {what})"
            )
        return int(number)


def _check_date(reader: _FortranReader, expected: Sequence[str], kind: str) -> str:
    reader.skipcmnt()
    date = reader.line().split()[0] if reader.pos < len(reader.lines) else ""
    if date not in expected:
        raise LegacyFormatError(
            f"{reader.name}: this {kind} has the date {date!r}; pysmsim reads "
            f"{' or '.join(expected)} (the format of SMSIM 2023-03-03)"
        )
    return date


# --- Parameter files ---------------------------------------------------------------


def _stress(code: int, s: list[float]) -> StressModel:
    match code:
        case 1:
            return LogLinearStress(stress_ref=s[0], dlog10_stress_dm=s[1], magnitude_ref=s[2])
        case 2:
            return HoerlStress(c1=s[0], c2=s[1], c3=s[2])
        case 3:
            return Stafford2022Stress(c1=s[0], c2=s[1])
    raise LegacyFormatError(f"stress model {code} is not one of SMSIM's (1, 2 or 3)")


def _source(
    numsource: int,
    shape: list[float],
    stress_code: int,
    s: list[float],
    scaling: list[float],
    weights: list[float],
    notes: list[str],
) -> SourceSpectrum:
    cls = _SOURCE_NUMBERS.get(numsource)
    if cls is None:
        raise LegacyFormatError(f"source number {numsource} is not one of SMSIM's (1 to 13)")
    pf_a, pd_a, pf_b, pd_b = shape
    wa, wb = weights
    if numsource in (2, 11, 12) and stress_code != 1:
        notes.append(
            f"source {numsource} with stress model {stress_code}: SMSIM uses the log-linear "
            "stress formula with these coefficients whatever the stress model, pysmsim uses "
            f"stress model {stress_code} (see the quirks in the documentation)"
        )
    if cls is SingleCorner:
        return SingleCorner(
            stress=_stress(stress_code, s),
            pf_a=pf_a,
            pd_a=pd_a,
            duration_weight_fa=wa,
            duration_weight_fb=wb,
        )
    if cls is Joyner1984:
        return Joyner1984(
            stress=_stress(stress_code, s),
            magnitude_hinge=s[2],
            fb_over_fa=s[3],
            duration_weight_fa=wa,
            duration_weight_fb=wb,
        )
    if cls is TwoCornerMultiplicative:
        return TwoCornerMultiplicative(
            stress=_stress(stress_code, s),
            log10_fa=(scaling[0], scaling[1], scaling[2]),
            pf_a=pf_a,
            pd_a=pd_a,
            pf_b=pf_b,
            pd_b=pd_b,
            duration_weight_fa=wa,
            duration_weight_fb=wb,
        )
    if cls is TwoCornerAdditive:
        return TwoCornerAdditive(
            stress=_stress(stress_code, s),
            log10_fa=(scaling[0], scaling[1], scaling[2]),
            log10_eps=(scaling[3], scaling[4], scaling[5]),
            pf_a=pf_a,
            pd_a=pd_a,
            pf_b=pf_b,
            pd_b=pd_b,
            duration_weight_fa=wa,
            duration_weight_fb=wb,
        )
    if cls is JiArchuleta2020:
        return JiArchuleta2020()
    if not issubclass(cls, WeightedDuration):  # pragma: no cover - only Ji and Archuleta
        raise LegacyFormatError(f"source {numsource} has no duration weights")
    return cls(duration_weight_fa=wa, duration_weight_fb=wb)


def _finite_fault(iflag: int, nlines: int, c: list[float]) -> FiniteFaultFactor:
    if iflag == 0 or nlines == 0:
        return NoFiniteFault()
    if iflag not in (1, 2):
        raise LegacyFormatError(f"iflag_f_ff = {iflag} is not 0, 1 or 2")
    combination = RpsCombination.QUADRATURE if iflag == 1 else RpsCombination.SUM
    c1, c2, c3, c4, dm = c
    match nlines:
        case 1:
            return OneLine(c1=c1, c2=c2, combination=combination)
        case 2:
            return TwoLines(c1=c1, c2=c2, c3=c3, c4=c4, combination=combination)
        case 3:
            return TwoLinesSmoothed(c1=c1, c2=c2, c3=c3, c4=c4, delta_m=dm, combination=combination)
        case 999:
            return Generalized(c1=c1, c2=c2, c3=c3, c4=c4, h_ratio=dm, combination=combination)
        case 1000:
            return Stafford2022FiniteFault(
                h_alpha=c1, h_beta=c2, h_gamma=c3, h_delta=c4, h_epsilon=dm, combination=combination
            )
    raise LegacyFormatError(f"nlines_f_ff = {nlines} is not 0, 1, 2, 3, 999 or 1000")


def _read_spreading(reader: _FortranReader) -> GeometricalSpreading:
    reader.skipcmnt()
    (option,) = reader.ints(1, "the geometrical spreading option")
    reader.skipcmnt()
    r1_line = reader.floats(6, "the frequency-dependent spreading parameters")
    reader.skipcmnt()
    (r_ref,) = reader.floats(1, "r_ref")
    (nsegs,) = reader.ints(1, "the number of spreading segments")
    match option:
        case 0:
            segments = []
            for _ in range(nsegs):
                r_start, a, b, m_ref = reader.floats(4, "a spreading segment")
                segments.append(SpreadingSegment(r_start=r_start, a=a, b=b, m_ref=m_ref))
            return Hinged(segments=tuple(segments), r_ref=r_ref)
        case 6:
            g1, rt, g2, grat = reader.floats(4, "the option-6 spreading parameters")
            return StaffordGeneralized(gamma1=g1, r_t=rt, gamma2=g2, g_ratio=grat, r_ref=r_ref)
        case 7:
            g1, rt, g2 = reader.floats(3, "the option-7 spreading parameters")
            return StaffordOptimal(gamma1=g1, r_t=rt, gamma2=g2, r_ref=r_ref)
    # Options 1 to 5 still read the segment lines
    for _ in range(nsegs):
        reader.floats(4, "a spreading segment")
    match option:
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
    raise LegacyFormatError(f"geometrical spreading option {option} is not 0 to 7")


def _q(p: list[float], notes: list[str]) -> QModel:
    fr1, qr1, s1, ft1, ft2, fr2, qr2, s2, c_q = p
    if fr1 < 0:
        if fr1 != -1.0:
            notes.append(
                f"Q: fr1 = {fr1:g} selects the Stafford et al. (2022) Q; SMSIM uses Rrup in its "
                "path attenuation only when fr1 is exactly -1 (pysmsim always does)"
            )
        return Stafford2022Q(q0=qr1, eta_a=s1, eta_b=ft1, eta_g=ft2, c_q=c_q)
    return PiecewiseQ(fr1=fr1, qr1=qr1, s1=s1, ft1=ft1, ft2=ft2, fr2=fr2, qr2=qr2, s2=s2, c_q=c_q)


def _high_cut(itype: int, f1: float, f2: float, eta: float) -> HighCutFilter:
    match itype:
        case 0:
            return NoHighCut()
        case 1:
            return RaisedCosineHighCut(f1=f1, f2=f2, eta=eta)
        case 2:
            return QuarterCosineHighCut(f1=f1, f2=f2, eta=eta)
    raise LegacyFormatError(f"itype_hcfilt = {itype} is not 0, 1 or 2")


def _bt15_table(file_name: str) -> str | None:
    base = PureWindowsPath(file_name.strip()).name  # handles \ and /
    for prefix, table in _BT15_FILES.items():
        if base.lower().startswith(prefix):
            return table
    return None


def _read_params(reader: _FortranReader) -> Params:
    notes = reader.notes
    _check_date(reader, (PARAMS_DATE,), "parameter file")
    reader.skipcmnt()
    title = reader.line()
    reader.skipcmnt()
    rho, beta, prtitn, rtp, fs = reader.floats(5, "rho, beta, prtitn, radpat, fs")
    medium = Medium(
        density=rho, shear_velocity=beta, partition=prtitn, radiation=rtp, free_surface=fs
    )

    reader.skipcmnt()
    numsource_s, *shape_s = reader.values(5, "the spectral shape")
    numsource = reader._int(numsource_s, "the source number")
    shape = [reader._float(v, "the spectral shape") for v in shape_s]
    reader.skipcmnt()
    stress_code_s, *stress_s = reader.values(6, "the stress specification")
    stress_code = reader._int(stress_code_s, "the stress model")
    stress = [reader._float(v, "the stress specification") for v in stress_s]
    reader.skipcmnt()
    scaling = reader.floats(6, "the spectral scaling")

    reader.skipcmnt()
    iflag_s, nlines_s, *ff_s = reader.values(7, "the finite-fault factor")
    finite_fault = _finite_fault(
        reader._int(iflag_s, "iflag_f_ff"),
        reader._int(nlines_s, "nlines_f_ff"),
        [reader._float(v, "the finite-fault coefficients") for v in ff_s],
    )

    spreading = _read_spreading(reader)
    reader.skipcmnt()
    q = _q(reader.floats(9, "the Q model"), notes)
    reader.skipcmnt()
    weights = reader.floats(2, "the source-duration weights")
    source = _source(numsource, shape, stress_code, stress, scaling, weights, notes)

    reader.skipcmnt()
    (nknots,) = reader.ints(1, "the number of path-duration knots")
    knots = [reader.floats(2, "a path-duration knot") for _ in range(nknots)]
    (slope,) = reader.floats(1, "the slope of the last path-duration segment")
    path_duration = PathDuration(
        distances=tuple(k[0] for k in knots),
        durations=tuple(k[1] for k in knots),
        slope_beyond=slope,
    )

    reader.skipcmnt()
    (namps,) = reader.ints(1, "the number of crustal amplifications")
    amps = [reader.floats(2, "a crustal amplification") for _ in range(namps)]
    if not amps:
        notes.append("no crustal amplification given: written as a constant amplification of 1")
        amps = [[1.0, 1.0]]
    reader.skipcmnt()
    fm, akappa, dkappadmag, amagkref = reader.floats(4, "fmax, kappa, dkappadmag, amagkref")
    site = Site(
        amp_frequencies=tuple(a[0] for a in amps),
        amplifications=tuple(a[1] for a in amps),
        kappa0=akappa,
        dkappa_dmag=dkappadmag,
        kappa_magnitude_ref=amagkref,
        fmax=fm,
    )

    reader.skipcmnt()
    fcut_s, nslope_s = reader.values(2, "fcut and nslope")
    fcut = reader._float(fcut_s, "fcut")
    nslope_in = reader._int(nslope_s, "nslope")
    nslope = 4 if nslope_in < 4 else 4 * (nslope_in // 4)
    if nslope != nslope_in:
        notes.append(
            f"low-cut filter: nslope = {nslope_in} is not 4, 8, 12, ...; SMSIM uses {nslope}, "
            f"and so does this file (order {nslope // 2})"
        )
    reader.skipcmnt()
    itype_s, *hc_s = reader.values(4, "the high-cut filter")
    high_cut = _high_cut(
        reader._int(itype_s, "itype_hcfilt"),
        *(reader._float(v, "the high-cut filter") for v in hc_s),
    )
    filters = Filters(low_cut=LowCutFilter(fcut=fcut, order=nslope // 2), high_cut=high_cut)

    reader.skipcmnt()
    zup, eps_int, amp_cutoff, osc_s = reader.values(4, "the RV parameters")
    osc = reader._int(osc_s, "osc_crrctn")
    corrections = list(OscillatorCorrection)
    if not 0 <= osc < len(corrections):
        raise LegacyFormatError(f"osc_crrctn = {osc} is not 0 to 5")
    correction = corrections[osc]
    reader.skipcmnt()
    wna_file = reader.line()
    reader.skipcmnt()
    ena_file = reader.line()
    tables = [_bt15_table(wna_file), _bt15_table(ena_file)]
    used = {3: (0,), 4: (1,), 5: (0, 1)}.get(osc, ())
    for i in used:
        if tables[i] is None:
            raise LegacyFormatError(
                f"osc_crrctn = {osc} uses the table {(wna_file, ena_file)[i]!r}, but pysmsim "
                "only has the two BT15 tables of SMSIM (bt15_wna_acr_... and bt15_ena_scr_...)"
            )
    bt15 = (tables[0] or "bt15_wna_acr", tables[1] or "bt15_ena_scr")
    rv = RVConfig(
        zup=reader._float(zup, "zup"),
        eps_int=reader._float(eps_int, "eps_int"),
        amp_cutoff=reader._float(amp_cutoff, "amp_cutoff"),
        osc_correction=correction,
        bt15_tables=bt15,
    )

    td = TDConfig()
    try:
        reader.skipcmnt()
        indxwind, taper, eps_w, eta_w, f_tb2te, f_te_xtnd = reader.values(6, "the window")
        reader.skipcmnt()
        dur_fctr, dt, tshift, seed, nsims, iran_type = reader.values(6, "the timing")
    except LegacyFormatError:
        notes.append("no time-domain lines: the [td] section has pysmsim's defaults")
    else:
        windows = {0: WindowShape.BOX, 1: WindowShape.EXPONENTIAL}
        noises = {0: NoiseDistribution.GAUSSIAN, 1: NoiseDistribution.UNIFORM}
        window = windows.get(reader._int(indxwind, "indxwind"))
        noise = noises.get(reader._int(iran_type, "iran_type"))
        if window is None or noise is None:
            raise LegacyFormatError("indxwind and iran_type must be 0 or 1")
        td = TDConfig(
            window=window,
            taper=reader._float(taper, "taper"),
            eps_w=reader._float(eps_w, "eps_w"),
            eta_w=reader._float(eta_w, "eta_w"),
            f_tb2te=reader._float(f_tb2te, "f_tb2te"),
            f_te_xtnd=reader._float(f_te_xtnd, "f_te_xtnd"),
            dur_fctr=reader._float(dur_fctr, "dur_fctr"),
            dt=reader._float(dt, "dt"),
            tshift=reader._float(tshift, "tshift"),
            seed=reader._int(seed, "seed"),
            nsims=reader._int(nsims, "nsims"),
            noise=noise,
        )

    return Params(
        title=title,
        medium=medium,
        source=source,
        finite_fault=finite_fault,
        spreading=spreading,
        q=q,
        path_duration=path_duration,
        site=site,
        filters=filters,
        rv=rv,
        td=td,
    )


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="latin-1")
    except OSError as err:
        raise LegacyFormatError(f"{path}: cannot read the file: {err.strerror or err}") from err


def read_params(path: str | Path) -> tuple[Params, list[str]]:
    """Read an SMSIM parameter file (dated 02/25/21).

    Parameters
    ----------
    path : str or Path
        The ``.params`` file.

    Returns
    -------
    tuple of Params and list of str
        The parameters, and notes on where SMSIM would behave differently
        or the file looks wrong.

    Raises
    ------
    LegacyFormatError
        If the file cannot be read, has another date, or holds a value
        SMSIM does not accept.
    """
    reader = _FortranReader(_read_text(path), Path(path).name)
    try:
        params = _read_params(reader)
    except LegacyFormatError:
        raise
    except (ValueError, TypeError) as err:  # a consistency check in a dataclass
        raise LegacyFormatError(f"{reader.name}: {err}") from err
    return params, reader.notes


# --- Control files of tmrsk_loop_rv_drvr ---------------------------------------------


@dataclass(frozen=True, slots=True)
class RVLoopControl:
    """The scenarios of a ``tmrsk_loop_rv_drvr`` control file.

    ``scenarios`` is the ``[scenarios]`` table of a run file: log-spaced
    values stay ``{start, stop, count}`` tables, as in the control file.
    """

    params_file: str
    col_file: str
    scenarios: dict[str, Any]
    notes: list[str] = field(default_factory=list)


def _spaced(reader: _FortranReader, flag: int, what: str) -> Any:
    """Log-spaced (flag 0) or individual (other flags) values."""
    if flag != 0:
        return reader.counted(what)
    n, start, stop = reader.values(3, f"the log-spaced {what}")
    count = reader._int(n, f"the number of {what}")
    first = reader._float(start, what)
    last = reader._float(stop, what)
    if count < 1 or first <= 0.0:
        raise LegacyFormatError(
            f"{reader._where()}: log-spaced {what} need a count of at least 1 and a start above 0"
        )
    if count == 1:
        return [first]
    return {"start": first, "stop": last, "count": count}


def _read_rv_loop_ctl(reader: _FortranReader) -> RVLoopControl:
    notes = reader.notes
    date = _check_date(reader, CTL_DATES, "control file")
    reader.skipcmnt()
    reader.line()  # summary file
    reader.skipcmnt()
    reader.line()  # comment
    reader.skipcmnt()
    params_file = reader.line()
    reader.skipcmnt()
    col_file = reader.line()
    reader.skipcmnt()
    reader.line()  # less output (Y/N)
    reader.skipcmnt()
    reader.line()  # column-label tag
    reader.skipcmnt()
    (damping,) = reader.floats(1, "the damping")
    reader.skipcmnt()
    reader.line()  # loop order: pysmsim's is fixed (stress, kappa_0, magnitude, distance)
    reader.skipcmnt()
    (iperflag,) = reader.ints(1, "the period flag")
    reader.skipcmnt()
    pgv_pga = reader.line()[:1].upper() == "Y"
    scenarios: dict[str, Any] = {}

    reader.skipcmnt()
    if iperflag == 0:
        scenarios["periods"] = _spaced(reader, 0, "periods")
        scenarios["pga"] = scenarios["pgv"] = pgv_pga
    else:
        periods = reader.counted("periods")
        scenarios["periods"] = [t for t in periods if t > 0]
        scenarios["pga"] = 0.0 in periods
        scenarios["pgv"] = -1.0 in periods
        others = sorted({t for t in periods if t < 0 and t != -1.0})
        if others:
            notes.append(
                f"periods {', '.join(f'{t:g}' for t in others)} (Wood-Anderson and WWSSN "
                "instruments) are left out: pysmsim's RV method does not compute them yet"
            )
        if not scenarios["periods"]:
            scenarios["periods"] = []
    if isinstance(scenarios["periods"], dict):
        count = scenarios["periods"]["count"]
    else:
        count = len(scenarios["periods"])
    if count > MAX_SPECTRAL_POINTS:
        raise LegacyFormatError(
            f"{reader.name}: {count} periods; a pysmsim run may have at most "
            f"{MAX_SPECTRAL_POINTS}, so reduce the number of periods"
        )

    reader.skipcmnt()
    (imagflag,) = reader.ints(1, "the magnitude flag")
    reader.skipcmnt()
    if imagflag == 1:
        scenarios["magnitudes"] = reader.counted("magnitudes")
    else:
        m_start, dm, m_stop = reader.floats(3, "m_start, delta_m, m_stop")
        scenarios["magnitudes"] = {"start": m_start, "stop": m_stop, "step": dm}

    reader.skipcmnt()
    (ik0flag,) = reader.ints(1, "the kappa_0 flag")
    reader.skipcmnt()
    if ik0flag < 0:
        reader.line()  # a placeholder line, skipped by SMSIM
    else:
        scenarios["kappa0"] = _spaced(reader, ik0flag, "kappa_0 values")

    reader.skipcmnt()
    (idistflag,) = reader.ints(1, "the distance flag")
    reader.skipcmnt()
    scenarios["distances"] = _spaced(reader, idistflag, "distances")

    reader.skipcmnt()
    (case_h,) = reader.ints(1, "the Rjb-to-Rrup case")
    conversions = {0: RjbToRrup.NONE, 1: RjbToRrup.MAGNITUDE_FUNCTION, 2: RjbToRrup.CY14_ZTOR}
    if case_h not in conversions:
        raise LegacyFormatError(f"{reader.name}: h4rjb2rrup case {case_h} is not 0, 1 or 2")
    mechanism = Mechanism.STRIKE_SLIP
    if date != "01/22/21":
        reader.skipcmnt()
        (mech,) = reader.floats(1, "the mechanism")
        mechanisms = {1: Mechanism.STRIKE_SLIP, 2: Mechanism.NORMAL, 3: Mechanism.REVERSE}
        if mech not in mechanisms:
            raise LegacyFormatError(f"{reader.name}: mechanism {mech:g} is not 1, 2 or 3")
        mechanism = mechanisms[int(mech)]
    reader.skipcmnt()
    m1, h1, c1, c2, c3, m2, h2 = reader.floats(7, "the Rjb-to-Rrup depth function")

    reader.skipcmnt()
    (istressflag,) = reader.ints(1, "the stress flag")
    reader.skipcmnt()
    if istressflag >= 0:
        scenarios["stress"] = _spaced(reader, istressflag, "stresses")

    table: dict[str, Any] = {
        "magnitudes": scenarios["magnitudes"],
        "distances": scenarios["distances"],
        "distance_metric": "rjb",
        "rjb_to_rrup": conversions[case_h].value,
        "mechanism": mechanism.value,
        "depth_function": {"m1": m1, "h1": h1, "c1": c1, "c2": c2, "c3": c3, "m2": m2, "h2": h2},
        "periods": scenarios["periods"],
        "pga": scenarios["pga"],
        "pgv": scenarios["pgv"],
        "damping": damping,
    }
    for name in ("kappa0", "stress"):
        if name in scenarios:
            table[name] = scenarios[name]
    return RVLoopControl(params_file=params_file, col_file=col_file, scenarios=table, notes=notes)


def read_rv_loop_ctl(path: str | Path) -> RVLoopControl:
    """Read a control file of SMSIM's ``tmrsk_loop_rv_drvr``.

    Parameters
    ----------
    path : str or Path
        The ``.ctl`` file, dated 03/03/23 (or 01/22/21, without the
        mechanism line).

    Returns
    -------
    RVLoopControl

    Raises
    ------
    LegacyFormatError
        If the file cannot be read, has another date, or holds a value the
        driver does not accept.
    """
    reader = _FortranReader(_read_text(path), Path(path).name)
    return _read_rv_loop_ctl(reader)


# --- The run file ----------------------------------------------------------------------

_DEFAULT_SCENARIOS: dict[str, Any] = {
    "magnitudes": {"start": 4.0, "stop": 8.0, "step": 1.0},
    "distances": [10.0, 50.0, 100.0],
    "periods": {"start": 0.01, "stop": 10.0, "count": 31},
}


def convert(params_path: str | Path, ctl_path: str | Path | None = None) -> tuple[str, list[str]]:
    """Convert SMSIM input files into the text of a TOML run file.

    Parameters
    ----------
    params_path : str or Path
        The ``.params`` file.
    ctl_path : str or Path, optional
        A ``tmrsk_loop_rv_drvr`` control file for the ``[scenarios]``
        section. Without one, the section holds example scenarios to edit.

    Returns
    -------
    tuple of str and list of str
        The run file, and the notes (also written at its top as comments).

    Raises
    ------
    LegacyFormatError
        If a file cannot be converted.
    """
    params, notes = read_params(params_path)
    notes = list(notes)
    if ctl_path is None:
        scenarios = dict(_DEFAULT_SCENARIOS)
        output = {"path": f"{Path(params_path).stem}.csv", "format": "csv"}
        notes.append(
            "no control file given: [scenarios] holds example values; edit them before running"
        )
    else:
        control = read_rv_loop_ctl(ctl_path)
        notes.extend(control.notes)
        scenarios = control.scenarios
        col_name = PureWindowsPath(control.col_file).name or "results.col"
        output = {"path": f"{Path(col_name).stem}.csv", "format": "csv"}
        if "kappa0" not in scenarios and params.site.dkappa_dmag != 0.0:
            notes.append(
                "kappa depends on magnitude (dkappadmag is not 0) and the control file takes "
                "kappa_0 from the parameter file: SMSIM's loop drivers ignore dkappadmag, "
                "pysmsim keeps it unless run with --fortran-compat"
            )
    doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    doc.update(params_table(params))
    doc["scenarios"] = scenarios
    doc["output"] = output
    sources = [Path(params_path).name] + ([] if ctl_path is None else [Path(ctl_path).name])
    header = _wrap(f"Converted by pysmsim from {' and '.join(sources)}.")
    if notes:
        header += ["#", "# Notes:"]
        for note in notes:
            header.extend(_wrap(f"- {note}"))
    text = "\n".join(header) + "\n\n" + _format_toml(doc)
    try:
        config = parse_config(tomllib.loads(text))
    except ValueError as err:  # for example, a control file with no periods and no PGA or PGV
        raise LegacyFormatError(f"the converted run file is not valid: {err}") from err
    if config.params != params:  # pragma: no cover - a bug in the converter
        raise LegacyFormatError("the converted parameters do not read back the same")
    return text, notes


def _wrap(text: str, width: int = 76) -> list[str]:
    """``text`` as comment lines, continued lines indented under the bullet."""
    return textwrap.wrap(
        text,
        width=width,
        initial_indent="# ",
        subsequent_indent="#   ",
        break_long_words=False,
        break_on_hyphens=False,
    )


# --- Writing TOML in the layout of the example run file ----------------------------

_WIDTH = 88


def _scalar(value: Any) -> str:
    return tomli_w.dumps({"v": value}).strip().removeprefix("v = ")


def _inline(table: dict[str, Any]) -> str:
    return "{ " + ", ".join(f"{k} = {_scalar(v)}" for k, v in table.items()) + " }"


def _is_section(key: str, value: Any) -> bool:
    """Whether a table is written as a [section] rather than on one line."""
    if not isinstance(value, dict):
        return False
    nested = any(isinstance(v, dict | list) for v in value.values())
    return nested or len(f"{key} = {_inline(value)}") > _WIDTH


def _entry(key: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        return [f"{key} = {_inline(value)}"]
    if not isinstance(value, list):
        return [f"{key} = {_scalar(value)}"]
    if value and isinstance(value[0], dict):
        return [f"{key} = [", *(f"    {_inline(v)}," for v in value), "]"]
    items = [_scalar(v) for v in value]
    one_line = f"{key} = [{', '.join(items)}]"
    if len(one_line) <= _WIDTH:
        return [one_line]
    lines = [f"{key} = ["]
    line = "   "
    for item in items:
        if len(line) + len(item) + 2 > _WIDTH:
            lines.append(line.rstrip())
            line = "   "
        line += f" {item},"
    lines += [line.rstrip(), "]"]
    return lines


def _section(name: str, table: dict[str, Any], out: list[str]) -> None:
    direct = [(k, v) for k, v in table.items() if not _is_section(k, v)]
    if direct or all(not _is_section(k, v) for k, v in table.items()):
        out += ["", f"[{name}]"]
        for key, value in direct:
            out += _entry(key, value)
    for key, value in table.items():
        if _is_section(key, value):
            _section(f"{name}.{key}", value, out)


def _format_toml(doc: dict[str, Any]) -> str:
    """``doc`` as TOML: short arrays and tables on one line, as in the example run file."""
    out: list[str] = []
    for key, value in doc.items():
        if not isinstance(value, dict):
            out += _entry(key, value)
    for key, value in doc.items():
        if isinstance(value, dict):
            _section(key, value, out)
    return "\n".join(out) + "\n"
