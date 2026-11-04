from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pysmsim.spectrum.source import (
    SOURCE_MODELS,
    AtkinsonSilva2000,
    JiArchuleta2020,
    Joyner1984,
    SingleCorner,
    SourceSpectrum,
    TwoCornerAdditive,
    TwoCornerMultiplicative,
    corner_frequency,
    seismic_moment,
)
from pysmsim.spectrum.stress import LogLinearStress

FREQS = np.logspace(-3, 2, 101)
STRESS = LogLinearStress(stress_ref=80.0, magnitude_ref=6.5)
LOG10_FA = (2.181, -0.496, 0.0)


def _example(cls: type[SourceSpectrum]) -> SourceSpectrum:
    """One example of each source model, with the settings it needs."""
    if cls is SingleCorner:
        return SingleCorner(stress=STRESS)
    if cls is Joyner1984:
        return Joyner1984(stress=STRESS, magnitude_hinge=6.5)
    if cls is TwoCornerMultiplicative:
        return TwoCornerMultiplicative(stress=STRESS, log10_fa=LOG10_FA)
    if cls is TwoCornerAdditive:
        return TwoCornerAdditive(stress=STRESS, log10_fa=LOG10_FA, log10_eps=(0.605, -0.255, 0.0))
    return cls()


EXAMPLES = [_example(cls) for cls in SOURCE_MODELS.values()]
IDS = list(SOURCE_MODELS)


def test_registry_names_match_classes() -> None:
    assert len(SOURCE_MODELS) == 13
    for name, cls in SOURCE_MODELS.items():
        assert cls.name == name


def test_moment_and_corner_frequency() -> None:
    assert seismic_moment(6.0) == pytest.approx(10**25.05)
    # Doubling the stress raises fc by 2**(1/3)
    fc1 = corner_frequency(1e25, 100.0, 3.5)
    assert corner_frequency(1e25, 200.0, 3.5) / fc1 == pytest.approx(2 ** (1 / 3))


@pytest.mark.parametrize("source", EXAMPLES, ids=IDS)
@pytest.mark.parametrize("magnitude", [2.0, 4.0, 6.0, 8.0])
def test_shape_is_one_at_zero_and_decreasing(source: SourceSpectrum, magnitude: float) -> None:
    terms = source.terms(magnitude, 3.5)
    shape = source.shape(np.concatenate([[0.0], FREQS]), terms)
    assert shape[0] == pytest.approx(1.0)
    assert np.all(np.diff(shape) <= 1e-15)


@pytest.mark.parametrize("source", EXAMPLES, ids=IDS)
def test_compiled_kernel_matches_numpy_twin(source: SourceSpectrum) -> None:
    terms = source.terms(6.0, 3.5)
    args = source.shape_args(terms)
    np.testing.assert_allclose(
        source.kernel.compiled(FREQS, *args), source.kernel.numpy(FREQS, *args), rtol=1e-14
    )


@pytest.mark.parametrize("source", EXAMPLES, ids=IDS)
def test_models_are_frozen(source: SourceSpectrum) -> None:
    with pytest.raises((FrozenInstanceError, AttributeError)):
        source.extra = 1.0  # type: ignore[attr-defined]


def test_single_corner_is_omega_squared() -> None:
    source = SingleCorner(stress=STRESS)
    terms = source.terms(6.0, 3.5)
    assert terms.fa == terms.fb
    assert source.shape(terms.fa, terms) == pytest.approx(0.5)
    f = 1000 * terms.fa
    assert source.shape(f, terms) == pytest.approx((terms.fa / f) ** 2, rel=1e-5)


def test_two_corner_models_match_single_corner_at_high_frequency() -> None:
    # Models 11 and 12 are built so that the high-frequency level equals that
    # of the single-corner model with the same stress (Boore et al., 2014)
    single = SingleCorner(stress=STRESS)
    f = np.array([1000.0])
    expected = single.shape(f, single.terms(6.0, 3.5))
    for source in (_example(TwoCornerMultiplicative), _example(TwoCornerAdditive)):
        level = source.shape(f, source.terms(6.0, 3.5))
        assert level == pytest.approx(expected, rel=1e-3)


def test_shape_keeps_input_shape() -> None:
    source = SingleCorner(stress=STRESS)
    terms = source.terms(6.0, 3.5)
    assert source.shape(1.0, terms).shape == ()
    assert source.shape(np.ones((2, 3)), terms).shape == (2, 3)


def test_additive_two_corner_rejects_low_stress() -> None:
    source = TwoCornerAdditive(
        stress=LogLinearStress(stress_ref=0.01),
        log10_fa=LOG10_FA,
        log10_eps=(0.605, -0.255, 0.0),
    )
    with pytest.raises(ValueError, match="fb is undefined"):
        source.terms(6.0, 3.5)


def test_fixed_corner_models_have_no_stress() -> None:
    assert AtkinsonSilva2000().terms(6.0, 3.5).stress is None
    assert not hasattr(AtkinsonSilva2000(), "stress")


def test_duration() -> None:
    source = SingleCorner(stress=STRESS, duration_weight_fa=1.0, duration_weight_fb=0.0)
    terms = source.terms(6.0, 3.5)
    assert source.duration(terms) == pytest.approx(1.0 / terms.fa)
    ja = JiArchuleta2020()
    assert ja.duration(ja.terms(6.0, 3.5)) == pytest.approx(10 ** (0.488 * 6.0 - 2.244))
