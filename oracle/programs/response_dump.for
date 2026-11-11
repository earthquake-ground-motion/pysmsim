! ----------------- BEGIN RESPONSE_DUMP -------------------------------
      Program Response_Dump

! Test-only program for the pysmsim golden tests (not part of SMSIM).
!
! Prints SMSIM's spect_amp for one magnitude and distance with each of
! the responses it supports (the iaorins and idva settings), on a
! log-spaced frequency grid.  It sets up the scenario like factor_dump and
! calls the unmodified SMSIM subroutines (included from
! smsim_subs_for_rv_programs.for and smsim_util_subs.for).
!
! Usage:
!   response_dump PARAMS M RRUP OUTFILE NFREQ FMIN FMAX DAMP T1 [T2 ...]
!
!   PARAMS   SMSIM .params file (any .pars files it names must be found)
!   M        moment magnitude
!   RRUP     rupture distance (km)
!   OUTFILE  output file
!   NFREQ, FMIN, FMAX  frequency grid
!   DAMP     oscillator damping (fraction of critical)
!   T1 ...   oscillator periods (s), up to 10
!
! The output starts with comment lines giving the scenario, then one row
! per frequency with the columns
!
!   f          frequency (Hz)
!   dis        iaorins = 1, idva = 0: ground displacement (cm-s)
!   vel        iaorins = 1, idva = 1: ground velocity (cm)
!   acc        iaorins = 1, idva = 2: ground acceleration (cm/s)
!   osc_<T>    iaorins = 2, idva = 0: pseudo-velocity response of an
!              oscillator with period T (cm), one column per period
!   wa         iaorins = 3: Wood-Anderson seismometer (poles and zeros)
!   wwssn_sp   iaorins = 4: WWSSN short-period seismometer
!
! Dates: 11/05/26 - Written for pysmsim

      character f_params*200, f_out*200, arg*200, lab*23
      character labels(20)*23
      logical tdflag
      integer nfreq, ifreq, nu_out, nargs, nper, ip, ncol, k
      real fmin_grid, fmax_grid, dlogf, f, per_in(10), out(20)

      include 'smsim.fi'

      nargs = command_argument_count()
      if (nargs < 9) then
        write(*,'(a)') ' Usage: response_dump PARAMS M RRUP OUTFILE'//
     :                 ' NFREQ FMIN FMAX DAMP T1 [T2 ...]'
        stop 2
      end if
      call get_command_argument(1, f_params)
      call get_command_argument(2, arg)
      read(arg, *) amag
      call get_command_argument(3, arg)
      read(arg, *) rrup
      call get_command_argument(4, f_out)
      call get_command_argument(5, arg)
      read(arg, *) nfreq
      call get_command_argument(6, arg)
      read(arg, *) fmin_grid
      call get_command_argument(7, arg)
      read(arg, *) fmax_grid
      call get_command_argument(8, arg)
      read(arg, *) damp
      nper = min(nargs - 8, 10)
      do ip = 1, nper
        call get_command_argument(8 + ip, arg)
        read(arg, *) per_in(ip)
      end do

      tdflag = .false.
      call get_params( f_params, tdflag )

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

      pi = 4.0*atan(1.0)
      twopi = 2.0 * pi
      call spect_scale()
      call const_am0_gsprd()

      ncol = 0
      ncol = ncol + 1
      labels(ncol) = 'f'
      ncol = ncol + 1
      labels(ncol) = 'dis'
      ncol = ncol + 1
      labels(ncol) = 'vel'
      ncol = ncol + 1
      labels(ncol) = 'acc'
      do ip = 1, nper
        ncol = ncol + 1
        lab = ' '
        write(lab, '(a,es9.3)') 'osc_', per_in(ip)
        labels(ncol) = lab
      end do
      ncol = ncol + 1
      labels(ncol) = 'wa'
      ncol = ncol + 1
      labels(ncol) = 'wwssn_sp'

      call get_lun(nu_out)
      open(unit=nu_out, file=f_out, status='unknown')
      write(nu_out,'(a)') '! response_dump: SMSIM spect_amp responses'
      write(nu_out,'(a)') '! params = '//trim(f_params)
      write(nu_out,'(a,1x,es23.15)') '! m =', amag
      write(nu_out,'(a,1x,es23.15)') '! rrup =', rrup
      write(nu_out,'(a,1x,es23.15)') '! rps =', rmod
      write(nu_out,'(a,1x,es23.15)') '! damp =', damp
      write(nu_out,'(a)') '!'
      write(nu_out,'(20(1x,a23))') (adjustr(labels(k)), k = 1, ncol)

      if (nfreq > 1) then
        dlogf = alog10(fmax_grid/fmin_grid)/real(nfreq - 1)
      else
        dlogf = 0.0
      end if

      DO ifreq = 1, nfreq
        f = fmin_grid*10.0**(real(ifreq - 1)*dlogf)
        k = 1
        out(k) = f
        iaorins = 1
        do idva = 0, 2
          k = k + 1
          out(k) = spect_amp(f)
        end do
        iaorins = 2
        do ip = 1, nper
          idva = 0
          fosc = 1.0/per_in(ip)
          k = k + 1
          out(k) = spect_amp(f)
        end do
        iaorins = 3
        idva = 0
        k = k + 1
        out(k) = spect_amp(f)
        iaorins = 4
        idva = 0
        k = k + 1
        out(k) = spect_amp(f)
        write(nu_out,'(20(1x,es23.15))') (out(k), k = 1, ncol)
      END DO

      close(unit=nu_out)

      stop
      end

      include 'smsim_subs_for_rv_programs.for'
      include 'smsim_util_subs.for'
! ----------------- END RESPONSE_DUMP ---------------------------------
