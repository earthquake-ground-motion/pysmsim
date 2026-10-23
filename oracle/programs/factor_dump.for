! ----------------- BEGIN FACTOR_DUMP ---------------------------------
      Program Factor_Dump

! Test-only program for the pysmsim golden tests (not part of SMSIM).
!
! Prints each factor of the SMSIM Fourier amplitude spectrum, as computed
! by function spect_amp, for one magnitude and distance on a log-spaced
! frequency grid.  It calls the unmodified SMSIM subroutines (included
! from smsim_subs_for_rv_programs.for and smsim_util_subs.for) in the
! same order as fmrsk_loop_fas_drvr, so the values are those that the
! SMSIM programs use.
!
! Usage:
!   factor_dump PARAMS M RRUP OUTFILE [NFREQ FMIN FMAX]
!
!   PARAMS   SMSIM .params file (any .pars files it names must be found)
!   M        moment magnitude
!   RRUP     rupture distance (km); the point-source distance Rps (rmod)
!            is computed from it with rmod_calc, as in the drivers
!   OUTFILE  output file
!   NFREQ, FMIN, FMAX  frequency grid (default 201 points, 0.01-100 Hz)
!
! The output starts with comment lines (beginning with "!") giving the
! scenario and the frequency-independent terms, followed by one row per
! frequency.  The columns are:
!
!   f          frequency (Hz)
!   source     spect_shape: source displacement spectrum shape (1 at f=0)
!   gsprd      geometrical spreading (gsprd_freq for i_gsprd_option 1-5,
!              otherwise the frequency-independent gsprd)
!   q          Q(f), from function q
!   q_path     exp(-pi*f*R_q/(Q*c_q)), with R_q = Rrup if fr1 = -1,
!              otherwise Rps
!   kappa      exp(-pi*kappa(M)*f)
!   fmax       1/sqrt(1+(f/fm)**8), or 1 if fm = 0
!   dimin      function dimin (= q_path*kappa*fmax)
!   site       crustal_amp_factor (linear f, log A, as in spect_amp)
!   lowcut     buttrlcf: low-cut filter
!   highcut    high_cut_filter
!   product    const*am0*source*gsprd*dimin*site*lowcut*highcut
!   fds        spect_amp with iaorins = 1, idva = 0: Fourier displacement
!              spectrum (cm-s)
!   fas        (2*pi*f)**2*fds: Fourier acceleration spectrum (cm/s)
!
! "product" and "fds" should agree to rounding; the difference shows that
! no factor is missing.

! Dates: 10/23/26 - Written for pysmsim

      character f_params*200, f_out*200, arg*200
      logical tdflag
      integer nfreq, ifreq, nu_out, nargs
      real fmin_grid, fmax_grid, dlogf, f
      real r_q, fac_source, fac_gsprd, fac_q, fac_q_path,
     :     fac_kappa, fac_fmax, fac_dimin, fac_site, fac_lowcut,
     :     fac_highcut, fac_product, fds_out, fas_out,
     :     dur_source, dur_path

      include 'smsim.fi'

      nargs = command_argument_count()
      if (nargs /= 4 .and. nargs /= 7) then
        write(*,'(a)') ' Usage: factor_dump PARAMS M RRUP OUTFILE'//
     :                 ' [NFREQ FMIN FMAX]'
        stop 2
      end if
      call get_command_argument(1, f_params)
      call get_command_argument(2, arg)
      read(arg, *) amag
      call get_command_argument(3, arg)
      read(arg, *) rrup
      call get_command_argument(4, f_out)
      nfreq = 201
      fmin_grid = 0.01
      fmax_grid = 100.0
      if (nargs == 7) then
        call get_command_argument(5, arg)
        read(arg, *) nfreq
        call get_command_argument(6, arg)
        read(arg, *) fmin_grid
        call get_command_argument(7, arg)
        read(arg, *) fmax_grid
      end if

! Read the params file, as in fmrsk_loop_fas_drvr
      tdflag = .false.
      call get_params( f_params, tdflag )

! Scenario, as in the loop drivers (Rjb and h4rjb2rrup are not used)
      am0 = 10.**(1.5*amag + 16.05)
      rjb = -1.0
      h4rjb2rrup = -1.0
      r = rrup
      rmod = rmod_calc(r, amag,
     :                  iflag_f_ff, nlines_f_ff,
     :                  c1_log10_f_ff, c2_log10_f_ff, mh_f_ff,
     :                  c3_log10_f_ff, c4_log10_f_ff,
     :                  m1t_f_ff, m2t_f_ff,
     :                  c0t_f_ff, c1t_f_ff, c2t_f_ff, c3t_f_ff,
     :                                                          f_ff)

! Fourier displacement spectrum, no instrument
      iaorins = 1
      idva = 0

! The same set-up as SMSIMFAS
      pi = 4.0*atan(1.0)
      twopi = 2.0 * pi
      call spect_scale()
      call const_am0_gsprd()

      dur_source = dursource(w_fa, fa, w_fb, fb)
      dur_path = durpath(rmod, nknots, rdur, dur, slast,
     :                   numsource, amag)

      call get_lun(nu_out)
      open(unit=nu_out, file=f_out, status='unknown')

      write(nu_out,'(a)') '! factor_dump: SMSIM spectral factors'
      write(nu_out,'(a)') '! params = '//trim(f_params)
      write(nu_out,'(a,1x,i0)')      '! numsource =', numsource
      write(nu_out,'(a,1x,i0)')      '! i_gsprd_option =',
     :                               i_gsprd_option
      write(nu_out,'(a,1x,es23.15)') '! m =', amag
      write(nu_out,'(a,1x,es23.15)') '! rrup =', rrup
      write(nu_out,'(a,1x,es23.15)') '! f_ff =', f_ff
      write(nu_out,'(a,1x,es23.15)') '! rps =', rmod
      write(nu_out,'(a,1x,es23.15)') '! m0 =', am0
      write(nu_out,'(a,1x,es23.15)') '! stress =', stress
      write(nu_out,'(a,1x,es23.15)') '! fa =', fa
      write(nu_out,'(a,1x,es23.15)') '! fb =', fb
      write(nu_out,'(a,1x,es23.15)') '! am0b_m0 =', am0b_m0
      write(nu_out,'(a,1x,es23.15)') '! const =', const
      write(nu_out,'(a,1x,es23.15)') '! gsprd =', gsprd
      write(nu_out,'(a,1x,es23.15)') '! freq_indep_factor =',
     :                               freq_indep_factor
      write(nu_out,'(a,1x,es23.15)') '! kappa =', kappa_f(amag)
      write(nu_out,'(a,1x,es23.15)') '! dur_source =', dur_source
      write(nu_out,'(a,1x,es23.15)') '! dur_path =', dur_path
      write(nu_out,'(a)') '!'
      write(nu_out,'(14(1x,a23))') 'f', 'source', 'gsprd', 'q',
     :  'q_path', 'kappa', 'fmax', 'dimin', 'site', 'lowcut',
     :  'highcut', 'product', 'fds', 'fas'

      if (fr1 == -1.0) then
        r_q = rrup
      else
        r_q = rmod
      end if

      if (nfreq > 1) then
        dlogf = alog10(fmax_grid/fmin_grid)/real(nfreq - 1)
      else
        dlogf = 0.0
      end if

      DO ifreq = 1, nfreq
        f = fmin_grid*10.0**(real(ifreq - 1)*dlogf)

        fac_source = spect_shape(f, fa, fb, pf_a, pd_a, pf_b, pd_b,
     :                           am0b_m0, numsource)
        if (i_gsprd_option > 0 .and. i_gsprd_option < 6) then
          fac_gsprd = gsprd_freq(f, rmod, r_ref, i_gsprd_option,
     :      r1_dmb_gsprd,
     :      pgsprd_r_le_r1_lf, pgsprd_r_le_r1_hf, pgsprd_r_gt_r1,
     :      ft1_dmb_gsprd, ft2_dmb_gsprd)
        else
          fac_gsprd = gsprd
        end if
        fac_q = q(f)
        fac_q_path = exp(-pi*f*r_q/(c_q*fac_q))
        fac_kappa = exp(-pi*kappa_f(amag)*f)
        if (fm == 0.0) then
          fac_fmax = 1.0
        else
          fac_fmax = 1.0/sqrt(1.0 + (f/fm)**8.0)
        end if
        fac_dimin = dimin(f)
        fac_site = crustal_amp_factor(f, namps, famp, amp, 2)
        fac_lowcut = buttrlcf(f, fcut, norder)
        fac_highcut = high_cut_filter(f)

        fac_product = const*am0*fac_source*fac_gsprd*fac_dimin*
     :                fac_site*fac_lowcut*fac_highcut
        fds_out = spect_amp(f)
        fas_out = (twopi*f)**2*fds_out

        write(nu_out,'(14(1x,es23.15))') f, fac_source, fac_gsprd,
     :    fac_q, fac_q_path, fac_kappa, fac_fmax, fac_dimin,
     :    fac_site, fac_lowcut, fac_highcut, fac_product,
     :    fds_out, fas_out
      END DO

      close(unit=nu_out)

      stop
      end
! ----------------- END FACTOR_DUMP -----------------------------------

      include 'smsim_subs_for_rv_programs.for'
      include 'smsim_util_subs.for'
