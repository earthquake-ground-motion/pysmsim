! ----------------- BEGIN TD_DUMP -------------------------------------
      Program TD_Dump

! Test-only program for the pysmsim golden tests (not part of SMSIM).
! Writes the white noise and the acceleration time series of one
! time-domain simulation, so that pysmsim can be fed the same noise
! (SMSIM's random-number generator cannot be reproduced in Python).
! It calls the unmodified SMSIM subroutines (included from td_subs.for and
! smsim_util_subs.for) in the same order as tmrsk_loop_td_drvr and gm_td.

! Usage:
!   td_dump PARAMS M RRUP OUTFILE
!   PARAMS   SMSIM .params file, with the time-domain lines
!   M        moment magnitude
!   RRUP     rupture distance (km); Rps (rmod) is computed with rmod_calc
!   OUTFILE  output file

! The noise is drawn exactly as acc_ts draws it, starting from
! iseed = -abs(seed): gasdev resets its saved value and ran1 its shuffle
! table when the seed is negative, so drawing the noise here and then
! calling acc_ts with the same starting seed gives acc_ts the same noise.

! The output starts with comment lines (beginning with "!") giving the
! indices and durations from Get_Npts_for_TD_Calcs and the measures of the
! simulation that gm_td computes (PGA, PGV, PGD, the integral of a^2, the
! 5-75% and 5-95% durations and the oscillator displacement at five
! periods, 5% damping).  Then one row per point of the series:
!   noise   the white noise (nonzero from nstart to nstop), before the
!           window
!   acc     the acceleration from acc_ts (cm/s^2)

! Dates: 11/12/26 - Written for pysmsim

      character f_params*200, f_out*200, arg*200
      logical tdflag, use_straightline_interp
      integer nu_out, nargs, i, nper, j05, j75, j95
      real noise(:), acc(:), vel(:), dis(:), a_sq_int(:)
      allocatable :: noise, acc, vel, dis, a_sq_int
      real per(5), rd(5), omega, damp, v0, d0, rv, aa
      real amin, amax, pga, pgv, pgd, a_sq_int_max
      integer nstart, nstop, npw2, ntaper
      real te

      include 'smsim.fi'

      data per /0.01, 0.1, 0.2, 1.0, 2.0/
      nper = 5
      damp = 0.05

      nargs = command_argument_count()
      if (nargs /= 4) then
        write(*,'(a)') ' Usage: td_dump PARAMS M RRUP OUTFILE'
        stop 2
      end if
      call get_command_argument(1, f_params)
      call get_command_argument(2, arg)
      read(arg, *) amag
      call get_command_argument(3, arg)
      read(arg, *) rrup
      call get_command_argument(4, f_out)

      tdflag = .true.
      call get_params( f_params, tdflag )

! Scenario, as in the loop drivers
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
      new_mr = .true.

      call Get_Npts_for_TD_Calcs(nstart, nstop, npw2, te, ntaper)

      allocate(noise(npw2), acc(npw2), vel(npw2), dis(npw2),
     :         a_sq_int(npw2))
      noise = 0.0

! The noise, as acc_ts draws it
      iseed = -abs(seed)
      if (iran_type .eq. 0) then
        do i = nstart, nstop
          noise(i) = gasdev(iseed)
        end do
      else
        do i = nstart, nstop
          noise(i) = ran1(iseed) - 0.5
        end do
      end if

! One simulation, as in gm_td
      iseed = -abs(seed)
      call acc_ts(acc, nstart, nstop, npw2, te, ntaper)

      v0 = 0.0
      d0 = 0.0
      call acc2vd(acc, npw2, dt, .false., v0, d0, vel, dis)
      call accsqint(acc, npw2, dt, .false., a_sq_int)
      a_sq_int_max = a_sq_int(npw2)
      call locate(a_sq_int,npw2,0.05*a_sq_int_max,j05)
      call locate(a_sq_int,npw2,0.75*a_sq_int_max,j75)
      call locate(a_sq_int,npw2,0.95*a_sq_int_max,j95)

      call mnmax(acc, 1, npw2, 1, amin, amax)
      pga = amax1(abs(amin), abs(amax))
      call mnmax(vel, 1, npw2, 1, amin, amax)
      pgv = amax1(abs(amin), abs(amax))
      call mnmax(dis, 1, npw2, 1, amin, amax)
      pgd = amax1(abs(amin), abs(amax))

      use_straightline_interp = .false.
      do i = 1, nper
        omega = twopi/per(i)
        call rs_calc(acc, npw2, omega, damp, dt,
     :               v0, d0, use_straightline_interp, rd(i), rv, aa)
      end do

      call get_lun(nu_out)
      open(unit=nu_out, file=f_out, status='unknown')
      write(nu_out,'(a)') '! td_dump: one SMSIM time-domain simulation'
      write(nu_out,'(a)') '! params = '//trim(f_params)
      write(nu_out,'(a,1x,es23.15)') '! m =', amag
      write(nu_out,'(a,1x,es23.15)') '! rrup =', rrup
      write(nu_out,'(a,1x,es23.15)') '! rps =', rmod
      write(nu_out,'(a,1x,es23.15)') '! fa =', fa
      write(nu_out,'(a,1x,es23.15)') '! fb =', fb
      write(nu_out,'(a,1x,es23.15)') '! dur_source =', dursource_calc
      write(nu_out,'(a,1x,es23.15)') '! dur_path =', durpath_calc
      write(nu_out,'(a,1x,es23.15)') '! dur_ex =', durex
      write(nu_out,'(a,1x,es23.15)') '! dt =', dt
      write(nu_out,'(a,1x,es23.15)') '! tshift =', tshift
      write(nu_out,'(a,1x,es23.15)') '! te =', te
      write(nu_out,'(a,1x,i0)')      '! nstart =', nstart
      write(nu_out,'(a,1x,i0)')      '! nstop =', nstop
      write(nu_out,'(a,1x,i0)')      '! npw2 =', npw2
      write(nu_out,'(a,1x,i0)')      '! ntaper =', ntaper
      write(nu_out,'(a,1x,es23.15)') '! pga =', pga
      write(nu_out,'(a,1x,es23.15)') '! pgv =', pgv
      write(nu_out,'(a,1x,es23.15)') '! pgd =', pgd
      write(nu_out,'(a,1x,es23.15)') '! a_sq_int =', a_sq_int_max
      write(nu_out,'(a,1x,i0)')      '! j05 =', j05
      write(nu_out,'(a,1x,i0)')      '! j75 =', j75
      write(nu_out,'(a,1x,i0)')      '! j95 =', j95
      write(nu_out,'(a,1x,es23.15)') '! damping =', damp
      do i = 1, nper
        write(nu_out,'(a,1x,es23.15,1x,es23.15)') '! period, rd =',
     :    per(i), rd(i)
      end do
      write(nu_out,'(a)') '!'
      write(nu_out,'(2(1x,a16))') 'noise', 'acc'
      do i = 1, npw2
        write(nu_out,'(2(1x,es16.9))') noise(i), acc(i)
      end do
      close(unit=nu_out)

      stop
      end
! ----------------- END TD_DUMP ---------------------------------------

      include 'td_subs.for'
      include 'smsim_util_subs.for'
