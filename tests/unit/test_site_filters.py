"""Unit tests of the site terms and the filters."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pysmsim.spectrum.filters import (
    HIGH_CUT_MODELS,
    Filters,
    HighCutFilter,
    LowCutFilter,
    NoHighCut,
    QuarterCosineHighCut,
    RaisedCosineHighCut,
)
from pysmsim.spectrum.site import Site

FREQS = np.concatenate(([0.0], np.logspace(-3, 2.5, 120)))

SITE = Site(
    amp_frequencies=(0.1, 1.0, 2.0, 5.0, 10.0),
    amplifications=(1.0, 1.5, 2.0, 2.5, 3.0),
    kappa0=0.04,
    dkappa_dmag=-0.01,
    kappa_magnitude_ref=6.0,
    fmax=15.0,
)

# --- Site ---------------------------------------------------------------------


def test_amplification_kernel_matches_numpy_twin() -> None:
    famp = np.asarray(SITE.amp_frequencies)
    amp = np.asarray(SITE.amplifications)
    k = Site.amplification_kernel
    np.testing.assert_allclose(k.compiled(FREQS, famp, amp), k.numpy(FREQS, famp, amp), rtol=1e-14)


def test_diminution_kernel_matches_numpy_twin() -> None:
    k = Site.diminution_kernel
    for fmax in (0.0, 15.0):
        np.testing.assert_allclose(
            k.compiled(FREQS, 0.03, fmax), k.numpy(FREQS, 0.03, fmax), rtol=1e-14
        )


def test_amplification_passes_through_the_table_and_is_constant_outside() -> None:
    np.testing.assert_allclose(SITE.amplification(SITE.amp_frequencies), SITE.amplifications)
    assert SITE.amplification([0.0, 0.01, 0.1]) == pytest.approx([1.0, 1.0, 1.0])
    assert SITE.amplification([10.0, 50.0]) == pytest.approx([3.0, 3.0])


def test_amplification_is_log_linear_between_points() -> None:
    # Halfway between 1 and 2 Hz: the geometric mean of 1.5 and 2.0
    assert SITE.amplification(1.5) == pytest.approx(math.sqrt(1.5 * 2.0), rel=1e-14)


def test_one_point_table_is_constant() -> None:
    site = Site(amp_frequencies=(1.0,), amplifications=(2.0,))
    assert site.amplification(FREQS) == pytest.approx(np.full(FREQS.shape, 2.0))


def test_kappa_depends_on_magnitude() -> None:
    assert SITE.kappa(6.0) == 0.04
    assert SITE.kappa(8.0) == pytest.approx(0.02)


def test_diminution() -> None:
    f = np.array([0.0, 1.0, 15.0])
    kappa = SITE.kappa(7.0)
    expected = np.exp(-math.pi * kappa * f) / np.sqrt(1.0 + (f / 15.0) ** 8)
    np.testing.assert_allclose(SITE.diminution(f, 7.0), expected, rtol=1e-14)
    # 1/sqrt(2) from fmax at f = fmax
    no_kappa = Site(amp_frequencies=(1.0,), amplifications=(1.0,), fmax=15.0)
    assert no_kappa.diminution(15.0, 6.0) == pytest.approx(1.0 / math.sqrt(2.0))
    no_fmax = Site(amp_frequencies=(1.0,), amplifications=(1.0,), kappa0=0.02)
    assert no_fmax.diminution(100.0, 6.0) == pytest.approx(math.exp(-2.0 * math.pi))


def test_site_keeps_the_shape_of_f() -> None:
    f = np.ones((2, 3))
    assert SITE.amplification(f).shape == (2, 3)
    assert SITE.diminution(f, 6.0).shape == (2, 3)
    assert SITE.amplification(1.0).shape == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"amp_frequencies": (1.0, 2.0), "amplifications": (1.0,)}, "same length"),
        ({"amp_frequencies": (), "amplifications": ()}, "at least one point"),
        ({"amp_frequencies": (2.0, 1.0), "amplifications": (1.0, 1.0)}, "strictly increasing"),
        ({"amp_frequencies": (1.0, 2.0), "amplifications": (1.0, 0.0)}, "positive"),
        ({"amp_frequencies": (1.0,), "amplifications": (1.0,), "fmax": -1.0}, "fmax"),
    ],
)
def test_site_checks(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Site(**kwargs)  # type: ignore[arg-type]


def test_site_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        SITE.kappa0 = 0.0  # type: ignore[misc]


# --- Low-cut filter -----------------------------------------------------------


def test_low_cut_kernel_matches_numpy_twin() -> None:
    k = LowCutFilter.kernel
    for fcut in (0.0, 0.1):
        np.testing.assert_allclose(k.compiled(FREQS, fcut, 4.0), k.numpy(FREQS, fcut, 4.0))


def test_low_cut_filter() -> None:
    lc = LowCutFilter(fcut=0.1, order=4)
    assert lc.factor(0.1) == pytest.approx(0.5)
    assert lc.factor(0.0) == 0.0
    assert lc.factor(0.05) == pytest.approx(1.0 / (1.0 + 2.0**8))
    assert lc.factor(100.0) == pytest.approx(1.0)
    assert LowCutFilter().factor(FREQS) == pytest.approx(np.ones(FREQS.shape))


def test_low_cut_checks() -> None:
    with pytest.raises(ValueError, match="even number"):
        LowCutFilter(fcut=0.1, order=3)
    with pytest.raises(ValueError, match="negative"):
        LowCutFilter(fcut=-0.1)


# --- High-cut filters ---------------------------------------------------------

TAPERS = [RaisedCosineHighCut(f1=10.0, f2=30.0, eta=2.0), QuarterCosineHighCut(f1=10.0, f2=30.0)]


def test_high_cut_registry() -> None:
    assert list(HIGH_CUT_MODELS) == ["none", "raised_cosine", "quarter_cosine"]
    assert all(issubclass(c, HighCutFilter) for c in HIGH_CUT_MODELS.values())


@pytest.mark.parametrize("hc", TAPERS, ids=lambda h: h.name)
def test_high_cut_kernel_matches_numpy_twin(hc: RaisedCosineHighCut) -> None:
    args = (hc.f1, hc.f2, hc.eta)
    np.testing.assert_allclose(hc.kernel.compiled(FREQS, *args), hc.kernel.numpy(FREQS, *args))


@pytest.mark.parametrize("hc", TAPERS, ids=lambda h: h.name)
def test_high_cut_taper(hc: RaisedCosineHighCut) -> None:
    out = hc.factor([0.0, 10.0, 20.0, 30.0, 40.0])
    assert out[0] == out[1] == 1.0
    assert out[3] == out[4] == 0.0
    assert 0.0 < out[2] < 1.0
    inside = hc.factor(np.linspace(10.0, 30.0, 50))
    assert np.all(np.diff(inside) <= 0.0)


def test_high_cut_values_at_the_middle() -> None:
    raised, quarter = TAPERS
    assert raised.factor(20.0) == pytest.approx(0.25)  # (1/2)**2
    assert quarter.factor(20.0) == pytest.approx(math.cos(math.pi / 4.0))


def test_no_high_cut() -> None:
    assert NoHighCut().factor(np.ones((2, 3))) == pytest.approx(np.ones((2, 3)))


@pytest.mark.parametrize("cls", [RaisedCosineHighCut, QuarterCosineHighCut])
def test_high_cut_checks(cls: type[RaisedCosineHighCut]) -> None:
    with pytest.raises(ValueError, match="0 < f1 < f2"):
        cls(f1=50.0, f2=25.0)
    with pytest.raises(ValueError, match="eta"):
        cls(f1=25.0, f2=50.0, eta=0.0)


def test_filters_multiply() -> None:
    filters = Filters(low_cut=LowCutFilter(fcut=0.1), high_cut=TAPERS[0])
    np.testing.assert_allclose(
        filters.factor(FREQS), filters.low_cut.factor(FREQS) * filters.high_cut.factor(FREQS)
    )
    assert Filters().factor(FREQS[1:]) == pytest.approx(np.ones(FREQS.size - 1))
