"""The time-domain (TD) method: simulated accelerograms and their peaks.

- :mod:`pysmsim.td.window`: the length of the series and the window
  (Fortran ``get_npts_for_td_calcs``, ``wind_box``, ``wind_exp``);
- :mod:`pysmsim.td.noise`: the white noise;
- :mod:`pysmsim.td.acc_ts`: one acceleration series, the windowed noise
  shaped by the Fourier spectrum (Fortran ``acc_ts``);
- :mod:`pysmsim.td.intensity`: velocity, displacement, Arias intensity and
  the significant durations;
- :mod:`pysmsim.td.response`: peak oscillator responses (Fortran
  ``rdrvaa``);
- :mod:`pysmsim.td.gm_td`: ``nsims`` simulations of one scenario and the
  statistics of their peaks (Fortran ``gm_td``).
"""
