# Comparison with SMSIM

pysmsim follows the Fortran of SMSIM 2023-03-03, and its tests compare it
with SMSIM's own programs (the "oracle"), built twice: as written, which is
mostly single precision, and with every REAL promoted to double precision.
The double-precision build is the main reference.

## How closely the results agree

| Quantity | Agreement with the double-precision SMSIM build |
| --- | --- |
| Fourier spectrum and each of its factors (source, path, site, filters, responses) | to 1e-12 |
| One series simulated from SMSIM's own white noise | to about 1e-9 of the series' peak |
| RV peak motions | within 1e-3 in almost every case (see below) |
| RV, the example run file (1,212 rows) | within 6e-4 |
| TD averages over 800 simulations | within the scatter of the simulations (see below) |

### Random vibration: the spectral moments

SMSIM integrates the spectral moments with the Numerical Recipes `odeint`
at a relative accuracy `eps_int` of 1e-5, but the error of its steps adds
up, so its moments are accurate only to a few parts in 10,000 (2.6e-3 in
the worst test case). pysmsim integrates them with adaptive Gauss-Kronrod
quadrature and agrees with an independent integration to about 1e-6.

The peak factor depends on the bandwidth sqrt(1 - m1²/(m0·m2)), which
magnifies small errors in the moments of narrow-band responses
(oscillators with long periods or low damping). So a few RV results (5 of
about 4,300 in the tests) differ from SMSIM's by more than 1e-3, at most
4.3e-3. In these cases pysmsim is the more accurate; it does not reproduce
`odeint`.

The single-precision SMSIM build, which is what the distributed programs
are, itself differs from the double-precision build by up to about 2e-3
(for example in the `.col` file of SMSIM's example run).

### Time domain: the random numbers

pysmsim draws its noise from NumPy's generator, not from SMSIM's `ran1`
and `gasdev`, so single simulations cannot match SMSIM's. The tests
compare the two ways:

- with SMSIM's own white noise fed into pysmsim, one series agrees with
  SMSIM's to about 1e-9 of its peak, for both windows, both noise
  distributions and a low-cut filter;
- the means of 800 simulations agree with SMSIM's within 4 standard errors
  (all but 3% within 3) and within 5% for every scenario and measure.

## Where pysmsim deliberately differs

pysmsim fixes some of SMSIM's quirks by default. The ones that change the
results can be switched back with `fortran_compat=True` (`pysmsim run
--fortran-compat`), which the tests use to compare with SMSIM:

| SMSIM | pysmsim by default | `fortran_compat` |
| --- | --- | --- |
| The RV drivers compute the upper integration frequency fup from kappa at the reference magnitude, so `dkappa_dmag` does not change fup | fup from kappa at the scenario's magnitude | SMSIM's fup |
| The DK80 peak factor is a Riemann sum whose step count doubles until it changes by less than `eps_int`; it agrees with the exact integral to about 1e-4 but loses more than 1e-3 with very many zero crossings | the exact integral | SMSIM's sum |
| The loop drivers set `dkappa_dmag` to 0 for every kappa_0, including the one from the params file | kappa depends on magnitude unless the grid gives kappa_0 values | kappa independent of magnitude |
| Source model 12 computes the magnitude where eps = 1 as -c1/c2, ignoring the reference magnitude | the reference magnitude is used | the `fortran_compat` key of `[source]` |
| Spreading options 4 and 5 use 1.570796 for π/2 (an effect below 1e-6) | π/2 | the `fortran_compat` key of `[spreading]` |

Other differences are not switchable, because SMSIM's behaviour is an
error rather than a convention:

- Source models 2, 11 and 12 always use the log-linear stress formula,
  whatever stress model the params file names (with `hoerl` this gives
  zero stress); pysmsim uses the named model. `pysmsim convert` notes it.
- `get_params` rounds a low-cut `nslope` down to a multiple of 4; pysmsim
  rejects other values.
- Failures return sentinel values (-999.9) instead of stopping; pysmsim
  raises an error with a message.
- With neither kappa nor fmax, SMSIM's fup is 0 and there is no motion;
  pysmsim raises an error.
- Spreading option 2 sets the shared `r_ref` to 1, which changes later
  scenarios of a run; options 1 and 2 use r_ref = 1 km in pysmsim.
- SMSIM's single-scenario programs (`gm_rv_drvr`, `gm_td_drvr`,
  `FAS_DRVR`) never set the rupture distance used by the Stafford et al.
  (2022) spreading and Q; pysmsim uses it. The oracle patches this for the
  tests.
- `q()` keeps the Q parameters of its first call; pysmsim has no such
  cache, so Q can change between runs in one session.
- The RV method is not offered for the Wood-Anderson and WWSSN short-period
  seismometers, to which SMSIM applies an oscillator correction with a
  made-up frequency. Their Fourier spectra are available.

## Inputs

pysmsim reads its own TOML run files, which replace SMSIM's `.params`,
`.ctl` and BT15 `.pars` files: settings are named, options are words
rather than numbers, and each model holds only its own parameters.
`pysmsim convert` turns SMSIM's files (the 02/25/21 params format) into a
run file and notes where SMSIM would have behaved differently with them.
