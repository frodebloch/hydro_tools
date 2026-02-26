C
C====  Sheet cavitation routines for XFOIL  =====
C
C     CAVINIT    - Initialize cavitation variables
C     CAVREGION  - Determine cavitated panel regions from Cp distribution
C     CAVTHICK   - Compute cavity thickness from source strengths
C     CAVCLOSE_FM - Franc-Michel short cavity closure model
C     CAVCLOSE_RJ - Re-entrant jet cavity closure model
C     CAVDRAG    - Compute cavity pressure drag
C     CAVSHOW    - Display cavity information
C     CAVSYS     - BL Newton system for cavitated stations
C     CAVMASS    - Compute cavity mass source and augment MASS array
C     CAVINV     - Inviscid-only cavitation prediction (no BL solve)
C     CAVINV_FB  - Inviscid cavitation with displacement feedback
C


      SUBROUTINE CAVINIT
C-----------------------------------------------------------
C     Initializes all cavitation variables to safe defaults.
C     Called from INIT in xfoil.f at startup.
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- cavitation off by default
      LCAV    = .FALSE.
      LCAVZ   = .FALSE.
      LCAVCONV = .FALSE.
C
C---- default closure model: Franc-Michel
      ICAVMOD = 1
C
C---- zero sigma
      SIGMA  = 0.0
      QCAV   = 0.0
      SIGMIN = 0.0
C
C---- zero cavity extent
      DO IS=1, 2
        ICAV1(IS) = 0
        ICAV2(IS) = 0
        NCAVS(IS) = 0
        XCAV1(IS) = 0.0
        XCAV2(IS) = 0.0
        SCAVL(IS) = 0.0
      ENDDO
      NCAVP = 0
C
C---- zero cavity thicknesses
      DO IS=1, 2
        DO IBL=1, IVX
          HCAV(IBL,IS)  = 0.0
          LCAVP(IBL,IS) = .FALSE.
          MCAV(IBL,IS)  = 0.0
        ENDDO
      ENDDO
      DO I=1, IQX
        HCAVP(I) = 0.0
      ENDDO
C
C---- zero output quantities
      CLCAV  = 0.0
      CDCAV  = 0.0
      CDCAV_P = 0.0
      CDCAV_J = 0.0
C
C---- closure model parameters
      FCLTAPER = 0.50
      DO IS=1, 2
        HCLOSE(IS) = 0.0
        HCMAX(IS)  = 0.0
        ICCLOSE(IS) = 0
      ENDDO
C
C---- iteration counters
      NCAVITER = 0
      RCAVEXT  = 0.0
C
C---- plot-copy variables (for CPCAV drawing after BL state restore)
      LCAVZP = .FALSE.
      DO IS=1, 2
        NCAVSP(IS) = 0
        ICAV1P(IS) = 0
        ICAV2P(IS) = 0
      ENDDO
C
C---- default fluid properties (water at 20C)
      PVAP = 2337.0
      PINF = 101325.0
      RHOL = 998.2
C
      RETURN
      END ! CAVINIT



      SUBROUTINE CAVREGION
C-----------------------------------------------------------
C     Determines which panels are cavitated based on
C     the current Cp distribution and cavitation number.
C
C     A panel is cavitated if  Cp(i) < -sigma.
C
C     Cavitation is tracked per BL-side: side 1 = suction
C     (top), side 2 = pressure (bottom).
C
C     Panels within 2 stations of stagnation point are 
C     excluded.  Minimum 3 contiguous cavitated panels 
C     required for a sheet cavity; fewer are ignored.
C
C     Sets:  ICAV1, ICAV2, NCAVS, XCAV1, XCAV2, SCAVL,
C            LCAVP, NCAVP
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- threshold Cp for cavitation (with small tolerance so that
C     stations at exactly Cp=-sigma still count as cavitated)
      CPTOL = 0.001
      CPCAV = -SIGMA + CPTOL
C
C---- stagnation exclusion zone: 2 BL stations from stagnation
C     IBL=2 is at the stagnation point, IBL=3 is its nearest 
C     neighbor — both have unreliable Cp in viscous BL.
C     IBL=4+ are allowed so that leading-edge suction peaks
C     (e.g. on thin elliptic airfoils) are not masked.
      IBLEXCL = 2
C
C---- diagnostic output (commented for production)
cc      WRITE(*,8000) SIGMA, CPCAV, LVISC, LVCONV
cc 8000 FORMAT(' CAVREGION: sigma=',F8.4,'  Cpcav=',F8.4,
cc     &       '  LVISC=',L1,'  LVCONV=',L1)
C
C---- initialize
      NCAVP = 0
      DO IS=1, 2
        ICAV1(IS) = 0
        ICAV2(IS) = 0
        NCAVS(IS) = 0
        XCAV1(IS) = 0.0
        XCAV2(IS) = 0.0
        SCAVL(IS) = 0.0
        DO IBL=1, NBL(IS)
          LCAVP(IBL,IS) = .FALSE.
        ENDDO
      ENDDO
C
C---- scan each side for cavitated regions
      DO 100 IS=1, 2
C
C------ find first and last cavitated BL station on this side
        IBLFIRST = 0
        IBLLAST  = 0
C
C------ track minimum Cp on each side for diagnostics
        CPMINV = 999.0
        CPMINI = 999.0
C
        DO IBL=2, IBLTE(IS)
C
C-------- skip stations in stagnation exclusion zone
          IF(IBL .LE. IBLEXCL+1) THEN
            GO TO 20
          ENDIF
C
C-------- get panel index for this BL station
          I = IPAN(IBL,IS)
C
C-------- track min Cp (both viscous and inviscid)
          IF(CPV(I) .LT. CPMINV) CPMINV = CPV(I)
          IF(CPI(I) .LT. CPMINI) CPMINI = CPI(I)
C
C-------- check Cp at this panel node
C         Use viscous Cp if available, otherwise inviscid
          IF(LVISC .AND. LVCONV) THEN
            CPTEST = CPV(I)
          ELSE
            CPTEST = CPI(I)
          ENDIF
C
          IF(CPTEST .LT. CPCAV) THEN
C---------- this panel is cavitated
            IF(IBLFIRST.EQ.0) IBLFIRST = IBL
            IBLLAST = IBL
          ENDIF
C
   20     CONTINUE
        ENDDO
C
C------ diagnostic: report min Cp per side (commented for production)
cc      WRITE(*,8010) IS, CPMINV, CPMINI, IBLFIRST, IBLLAST,
cc     &                IBLTE(IS)
cc 8010   FORMAT('  Side',I2,': Cpmin_v=',F9.4,'  Cpmin_i=',F9.4,
cc     &         '  first=',I4,'  last=',I4,'  IBLTE=',I4)
C
C------ check minimum 3 contiguous panels
        IF(IBLFIRST.GT.0) THEN
          NSPAN = IBLLAST - IBLFIRST + 1
          IF(NSPAN .LT. 3) THEN
C---------- too few panels, skip this side
            WRITE(*,*) '  CAVREGION: too few panels, NSPAN=', NSPAN
            GO TO 100
          ENDIF
C
C---------- set cavity extent
          ICAV1(IS) = IBLFIRST
          ICAV2(IS) = IBLLAST
          NCAVS(IS) = NSPAN
C
C---------- get x/c coordinates of cavity endpoints
          I1 = IPAN(IBLFIRST,IS)
          I2 = IPAN(IBLLAST,IS)
          XCAV1(IS) = X(I1) / CHORD
          XCAV2(IS) = X(I2) / CHORD
C
C---------- compute cavity arc length
          SCAVL(IS) = 0.0
          DO IBL = IBLFIRST+1, IBLLAST
            IM1 = IPAN(IBL-1,IS)
            IP  = IPAN(IBL,IS)
            DS = SQRT( (X(IP)-X(IM1))**2 + (Y(IP)-Y(IM1))**2 )
            SCAVL(IS) = SCAVL(IS) + DS
          ENDDO
C
C---------- mark cavitated BL stations
          DO IBL = IBLFIRST, IBLLAST
            LCAVP(IBL,IS) = .TRUE.
          ENDDO
C
C---------- count total cavitated panels
          NCAVP = NCAVP + NSPAN
        ENDIF
C
  100 CONTINUE
C
C---- set flag for whether any cavity exists
      LCAVZ = (NCAVP .GT. 0)
C
      RETURN
      END ! CAVREGION



      SUBROUTINE CAVTHICK
C-----------------------------------------------------------
C     Computes cavity thickness distribution from inviscid
C     mass deficit, with closure-model shaping.
C
C     The cavity thickness h(s) is determined by integrating
C     the mass-source difference between the body inviscid
C     velocity and the cavity velocity QCAV:
C
C       h(s) = (1/QCAV) * integral[ (UINV_body(s') - QCAV) ds' ]
C
C     where the integral runs from detachment to station s.
C     
C     For ICAVMOD=1 (Franc-Michel):
C       The last FCLTAPER fraction of the cavity length has
C       a smooth taper h -> 0 at closure.  The taper uses a
C       cos^2 profile: h_tapered = h_raw * cos^2(pi/2 * xi)
C       where xi runs from 0 (start of taper) to 1 (closure).
C       This ensures h=0 and dh/ds=0 at the closure point.
C
C     For ICAVMOD=2 (re-entrant jet):
C       No taper — the cavity remains open at closure.
C       The cavity thickness at the last station is the
C       "jet thickness" that determines the jet momentum drag.
C
C     Sets:
C       HCAV(IBL,IS)  = h(s)       (stored for plotting/drag)
C       HCAVP(I)      = h(s)       (panel-indexed, for CPCAV)
C       HCMAX(IS)     = max thickness on each side
C       HCLOSE(IS)    = closure thickness on each side
C       ICCLOSE(IS)   = BL station where FM taper begins
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
      REAL PI
      PI = 4.0 * ATAN(1.0)
C
      DO IS=1, 2
        HCMAX(IS)  = 0.0
        HCLOSE(IS) = 0.0
        ICCLOSE(IS) = 0
        IF(NCAVS(IS).LE.0) GO TO 100
C
C------ Phase 1: integrate raw mass deficit from detachment
        HCUM = 0.0
        DO IBL = ICAV1(IS), ICAV2(IS)
          I = IPAN(IBL,IS)
C
C-------- get arc-length step from previous station
          IF(IBL .EQ. ICAV1(IS)) THEN
            DS = 0.0
          ELSE
            IM1 = IPAN(IBL-1,IS)
            DS = SQRT((X(I)-X(IM1))**2 + (Y(I)-Y(IM1))**2)
          ENDIF
C
C-------- base inviscid edge velocity at this station
C         Use raw panel velocity QINV (uncontaminated by V-I coupling)
          UBODY = ABS(QINV(I))
C
C-------- integrate: dh = (Ubody - QCAV)/QCAV * ds
C         Use trapezoidal rule for smoother integration
          IF(IBL .EQ. ICAV1(IS)) THEN
            HCUM = 0.0
          ELSE
            IM1BL = IBL - 1
            IM1   = IPAN(IM1BL,IS)
            UBODM = ABS(QINV(IM1))
            DSRC  = 0.5*((UBODM - QCAV) + (UBODY - QCAV)) / QCAV
            HCUM  = HCUM + DSRC * DS
          ENDIF
C
C-------- enforce h >= 0 (cavity cannot have negative thickness)
          IF(HCUM .LT. 0.0) HCUM = 0.0
C
C-------- store raw cavity thickness
          HCAV(IBL,IS) = HCUM
          HCAVP(I)     = HCUM
C
C-------- track maximum thickness
          IF(HCUM .GT. HCMAX(IS)) HCMAX(IS) = HCUM
C
        ENDDO
C
C------ Phase 2: apply closure model shaping
C
        IF(ICAVMOD .EQ. 1) THEN
C
C======== Franc-Michel closure: smooth taper to zero ========
C
C-------- determine taper start station
C         Taper covers the last FCLTAPER fraction of cavity arc length
          STAPER = FCLTAPER * SCAVL(IS)
C
C-------- accumulate arc length from closure end backwards to
C         find the station where the taper begins
          SCUM = 0.0
          ICCLOSE(IS) = ICAV2(IS)
          DO IBL = ICAV2(IS), ICAV1(IS)+1, -1
            I   = IPAN(IBL,IS)
            IM1 = IPAN(IBL-1,IS)
            DS = SQRT((X(I)-X(IM1))**2 + (Y(I)-Y(IM1))**2)
            SCUM = SCUM + DS
            IF(SCUM .GE. STAPER) THEN
              ICCLOSE(IS) = IBL
              GO TO 30
            ENDIF
          ENDDO
C-------- if we get here, taper covers entire cavity
          ICCLOSE(IS) = ICAV1(IS) + 1
C
   30     CONTINUE
C
C-------- compute total taper-region arc length (from ICCLOSE to ICAV2)
          STOTAL = 0.0
          DO IBL = ICCLOSE(IS)+1, ICAV2(IS)
            I   = IPAN(IBL,IS)
            IM1 = IPAN(IBL-1,IS)
            DS = SQRT((X(I)-X(IM1))**2 + (Y(I)-Y(IM1))**2)
            STOTAL = STOTAL + DS
          ENDDO
C
C-------- apply cos^2 taper within the closure region
C         h_tapered = h_raw * cos^2(pi/2 * xi)
C         where xi = (s - s_taper_start) / (s_closure - s_taper_start)
C         xi=0 at taper start, xi=1 at closure
          IF(STOTAL .GT. 0.0) THEN
            SCUM = 0.0
            HRAW_AT_TAPER = HCAV(ICCLOSE(IS),IS)
            DO IBL = ICCLOSE(IS)+1, ICAV2(IS)
              I   = IPAN(IBL,IS)
              IM1 = IPAN(IBL-1,IS)
              DS = SQRT((X(I)-X(IM1))**2 + (Y(I)-Y(IM1))**2)
              SCUM = SCUM + DS
              XI = SCUM / STOTAL
              IF(XI .GT. 1.0) XI = 1.0
C
C------------ cos^2 taper function: smooth h->0 with dh/ds->0 at xi=1
              FTAPER = COS(0.5*PI*XI)**2
C
C------------ apply taper, blending from raw thickness
              HCAV(IBL,IS) = HCAV(IBL,IS) * FTAPER
              HCAVP(I)     = HCAV(IBL,IS)
            ENDDO
          ENDIF
C
C-------- closure thickness should be ~0 for FM model
          HCLOSE(IS) = HCAV(ICAV2(IS),IS)
C
        ELSE
C
C======== Re-entrant jet: no taper — cavity stays open ========
C
C-------- closure thickness is the raw cavity thickness at ICAV2
          HCLOSE(IS) = HCAV(ICAV2(IS),IS)
          ICCLOSE(IS) = ICAV2(IS)
C
        ENDIF
C
  100   CONTINUE
      ENDDO
C
      RETURN
      END ! CAVTHICK



      SUBROUTINE CAVCLOSE_FM(IS)
C-----------------------------------------------------------
C     Franc-Michel short cavity closure model.
C
C     This routine is called AFTER CAVTHICK has computed and
C     tapered the cavity thickness.  Its job is to compute the
C     Cp recovery profile in the taper zone, which is needed
C     by CAVDRAG to get the correct pressure drag.
C
C     In the taper zone (ICCLOSE to ICAV2):
C       - The cavity thickness tapers from h_raw to 0 via cos^2
C       - The pressure is no longer the constant Cp = -sigma
C       - The Cp recovers from -sigma toward the body Cp
C       - Blended Cp: Cp(xi) = -sigma*cos^2 + Cp_body*sin^2
C         where xi is the taper coordinate (0 at taper start,
C         1 at closure)
C
C     The taper-zone Cp is NOT stored in CPV (which holds the
C     viscous Cp for display).  Instead, CAVDRAG directly
C     computes the pressure drag using the blended Cp.
C
C     Input:
C       IS     -  side index (1=suction, 2=pressure)
C
C     Output:
C       HCLOSE(IS) - closure thickness (should be ~0)
C
C     Note: HCLOSE, ICCLOSE already set by CAVTHICK.
C     This routine validates them and reports diagnostics.
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
      IF(NCAVS(IS).LE.0 .OR. ICAV2(IS).LE.0) RETURN
C
C---- closure thickness was already set by CAVTHICK (should be ~0)
C     Validate it is small relative to max cavity thickness
      IF(HCMAX(IS) .GT. 0.0) THEN
        HRAT = HCLOSE(IS) / HCMAX(IS)
        IF(HRAT .GT. 0.01) THEN
          WRITE(*,2000) IS, HCLOSE(IS)/CHORD, HRAT*100.0
 2000     FORMAT(' CAVCLOSE_FM: side',I2,
     &           ' h_close/c =',E12.4,
     &           ' (',F5.1,'% of h_max) - taper residual')
        ENDIF
      ENDIF
C
C---- validate taper start station
      IF(ICCLOSE(IS).LT.ICAV1(IS) .OR.
     &   ICCLOSE(IS).GT.ICAV2(IS)) THEN
        WRITE(*,*) 'CAVCLOSE_FM: invalid ICCLOSE, side', IS
        ICCLOSE(IS) = ICAV2(IS)
      ENDIF
C
      RETURN
      END ! CAVCLOSE_FM



      SUBROUTINE CAVCLOSE_RJ(IS)
C-----------------------------------------------------------
C     Open cavity / re-entrant jet closure model.
C
C     The cavity remains open at closure.  A re-entrant jet
C     at the cavity velocity carries momentum back upstream
C     inside the cavity.  The jet thickness equals the cavity
C     thickness at the closure station.
C
C     Momentum balance at closure:
C       The jet has velocity Qj = Qcav and thickness h_jet.
C       The momentum flux of the jet produces a drag force:
C         F_jet = rho * Qcav^2 * h_jet
C       In drag coefficient form:
C         CDjet = 2*(1+sigma) * h_jet/c   per side
C
C     This routine computes:
C       - Jet thickness = HCLOSE(IS) = HCAV(ICAV2,IS)
C       - Jet velocity  = QCAV
C       - Jet volume flux per unit span = QCAV * h_jet
C       - Jet momentum flux = rho * QCAV^2 * h_jet
C
C     The actual drag computation is done in CAVDRAG.
C
C     Input:
C       IS     -  side index (1=suction, 2=pressure)
C
C     Output:
C       HCLOSE(IS) - closure thickness (= jet thickness)
C
C     Note: HCLOSE already set by CAVTHICK.
C     This routine validates and reports jet parameters.
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
      IF(NCAVS(IS).LE.0 .OR. ICAV2(IS).LE.0) RETURN
C
C---- closure thickness = jet thickness (already set by CAVTHICK)
C     Validate it is positive and physically reasonable
      IF(HCLOSE(IS) .LE. 0.0) THEN
        WRITE(*,*) 'CAVCLOSE_RJ: zero jet thickness, side', IS
        RETURN
      ENDIF
C
C---- jet parameters
      QJET   = QCAV
      HJET   = HCLOSE(IS)
      QFLUX  = QJET * HJET
      CDJET1 = 2.0*(1.0 + SIGMA) * HJET / CHORD
C
C---- report jet parameters if significant
      IF(HJET/CHORD .GT. 1.0E-6) THEN
        WRITE(*,2100) IS, HJET/CHORD, QJET/QINF,
     &                QFLUX/(QINF*CHORD), CDJET1
 2100   FORMAT(' RJ closure side',I2,':',
     &        ' h_jet/c =',E10.3,
     &        ' Qj/Qinf =',F7.4,
     &        ' Qflux/(Qinf*c) =',E10.3,
     &        ' CDjet =',E10.3)
      ENDIF
C
      RETURN
      END ! CAVCLOSE_RJ



      SUBROUTINE CAVDRAG
C-----------------------------------------------------------
C     Computes cavity pressure drag coefficient.
C
C     The drag has two components:
C
C     1. Pressure drag from the cavity Cp on the surface:
C        CDp = -integral( Cp * dy ) / chord
C
C        For the FM model, the Cp varies in the taper zone:
C          - From ICAV1 to ICCLOSE:  Cp = -sigma (constant)
C          - From ICCLOSE to ICAV2:  Cp blended from -sigma
C            to the body Cp via cos^2(pi/2*xi)
C
C        For the RJ model:
C          - Cp = -sigma over the entire cavity (no taper)
C
C     2. Jet momentum drag (RJ model only):
C        CDjet = 2*(1+sigma) * h_jet/c   per side
C
C     Stores: CDCAV, CDCAV_P, CDCAV_J, CLCAV
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
      REAL PI
      PI = 4.0 * ATAN(1.0)
C
      CDCAV   = 0.0
      CDCAV_P = 0.0
      CDCAV_J = 0.0
C
      DO IS=1, 2
        IF(NCAVS(IS).LE.0) GO TO 100
C
C------ pressure drag from cavity Cp on surface
C       CDp = -integral( Cp * dy ) / chord
C       This is a body-frame integral (not wind-axis projected).
C       Wind-axis projection (as in CLCALC) is NOT used here because
C       CDcav_p is an open-surface integral over the cavitated region
C       only, where the sin(alfa)*dx cross-terms do not cancel as they
C       do in a closed contour integral. The full wind-axis pressure
C       drag including cavity effects is computed by CLCALC via the
C       QVIS/GAM overrides at cavity stations.
C       BL stations on both sides go LE->TE, but the contour
C       integral convention (counterclockwise) goes TE->LE on the bottom.
C       This reversal means DY for IS=2 has the wrong sign.
C       SGNDY corrects for this: +1 for top (IS=1), -1 for bottom (IS=2).
C
        IF(IS.EQ.1) THEN
          SGNDY =  1.0
        ELSE
          SGNDY = -1.0
        ENDIF
        CDSIDE = 0.0
C
        IF(ICAVMOD .EQ. 1) THEN
C
C======== FM model: split into constant-Cp and taper regions ========
C
C-------- Region 1: constant Cp = -sigma (ICAV1 to ICCLOSE)
          DO IBL = ICAV1(IS), ICCLOSE(IS)
            IF(.NOT.LCAVP(IBL,IS)) GO TO 40
C
            I = IPAN(IBL,IS)
C
C---------- panel-averaged dy contribution
            IF(IBL.LT.IBLTE(IS) .AND. IBL.LT.ICAV2(IS)) THEN
              IP1 = IPAN(MIN(IBL+1,ICAV2(IS)),IS)
              DY = Y(IP1) - Y(I)
            ELSE
              IM1 = IPAN(MAX(IBL-1,ICAV1(IS)),IS)
              DY = Y(I) - Y(IM1)
            ENDIF
C
C---------- Cp = -sigma in the constant region
            CDSIDE = CDSIDE + SIGMA * DY * SGNDY
C
   40       CONTINUE
          ENDDO
C
C-------- Region 2: taper zone (ICCLOSE+1 to ICAV2)
C         In the FM taper zone, the cavity closes and pressure
C         recovers.  The Cp transitions from -sigma at the taper
C         start toward 0 (freestream) at closure.  The recovery
C         follows the same cos^2 profile as the thickness taper:
C           Cp(xi) = -sigma * cos^2(pi/2 * xi)
C         This gives Cp=-sigma at xi=0, Cp=0 at xi=1,
C         with dCp/ds=0 at closure (smooth recompression).
C
C-------- compute taper-zone arc length
          STOTAL = 0.0
          IF(ICCLOSE(IS) .LT. ICAV2(IS)) THEN
            DO IBL = ICCLOSE(IS)+1, ICAV2(IS)
              I   = IPAN(IBL,IS)
              IM1 = IPAN(IBL-1,IS)
              DS = SQRT((X(I)-X(IM1))**2 + (Y(I)-Y(IM1))**2)
              STOTAL = STOTAL + DS
            ENDDO
          ENDIF
C
          IF(STOTAL .GT. 0.0) THEN
            SCUM = 0.0
            DO IBL = ICCLOSE(IS)+1, ICAV2(IS)
              IF(.NOT.LCAVP(IBL,IS)) GO TO 45
C
              I   = IPAN(IBL,IS)
              IM1 = IPAN(IBL-1,IS)
              DS = SQRT((X(I)-X(IM1))**2 + (Y(I)-Y(IM1))**2)
              SCUM = SCUM + DS
              XI = SCUM / STOTAL
              IF(XI .GT. 1.0) XI = 1.0
C
C------------ Cp in taper zone: recovers from -sigma toward 0
C             via cos^2 profile (same as thickness taper)
              CPTAPER = -SIGMA * COS(0.5*PI*XI)**2
C
C------------ panel-averaged dy contribution
              IF(IBL.LT.IBLTE(IS) .AND. IBL.LT.ICAV2(IS)) THEN
                IP1 = IPAN(MIN(IBL+1,ICAV2(IS)),IS)
                DY = Y(IP1) - Y(I)
              ELSE
                IM1P = IPAN(MAX(IBL-1,ICAV1(IS)),IS)
                DY = Y(I) - Y(IM1P)
              ENDIF
C
C------------ add taper-zone pressure drag (note: CDp = -Cp * DY / chord)
              CDSIDE = CDSIDE + (-CPTAPER) * DY * SGNDY
C
   45         CONTINUE
            ENDDO
          ENDIF
C
        ELSE
C
C======== RJ model: constant Cp = -sigma over entire cavity ========
C
          DO IBL = ICAV1(IS), ICAV2(IS)
            IF(.NOT.LCAVP(IBL,IS)) GO TO 50
C
            I = IPAN(IBL,IS)
C
C---------- panel-averaged dy contribution
            IF(IBL.LT.IBLTE(IS) .AND. IBL.LT.ICAV2(IS)) THEN
              IP1 = IPAN(MIN(IBL+1,ICAV2(IS)),IS)
              DY = Y(IP1) - Y(I)
            ELSE
              IM1 = IPAN(MAX(IBL-1,ICAV1(IS)),IS)
              DY = Y(I) - Y(IM1)
            ENDIF
C
C---------- Cp = -sigma
            CDSIDE = CDSIDE + SIGMA * DY * SGNDY
C
   50       CONTINUE
          ENDDO
C
        ENDIF
C
C------ normalize by chord
        CDSIDE = CDSIDE / CHORD
C
C------ accumulate pressure drag
        CDCAV_P = CDCAV_P + CDSIDE
C
C------ for re-entrant jet model, add jet momentum drag per side
        IF(ICAVMOD.EQ.2) THEN
          HJET = HCLOSE(IS)
          IF(HJET .GT. 0.0) THEN
            CDJET1 = 2.0*(1.0 + SIGMA) * HJET / CHORD
            CDCAV_J = CDCAV_J + CDJET1
          ENDIF
        ENDIF
C
  100   CONTINUE
      ENDDO
C
C---- total cavity drag = pressure + jet
      CDCAV = CDCAV_P + CDCAV_J
C
C---- compute total cavity length / chord
      CLCAV = 0.0
      DO IS=1, 2
        CLCAV = CLCAV + SCAVL(IS) / CHORD
      ENDDO
C
      RETURN
      END ! CAVDRAG



      SUBROUTINE CAVSHOW
C-----------------------------------------------------------
C     Displays current cavity information with closure-model
C     specific details.
C     Uses plot-copy variables (LCAVZP, NCAVSP) as fallback
C     when LCAVZ has been cleared after VISCAL or CAVINV.
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- local: whether we are using plot-copy fallback
      LOGICAL LPCOPY
C
      IF(.NOT.LCAV) THEN
        WRITE(*,*) 'Cavitation modeling is not active.'
        RETURN
      ENDIF
C
      WRITE(*,*)
      WRITE(*,1000) SIGMA, QCAV/QINF, -SIGMA
 1000 FORMAT(' Cavitation number sigma  =', F10.4,
     &      /' Cavity speed  Qcav/Qinf  =', F10.5,
     &      /' Cavity pressure  Cp,cav  =', F10.4)
C
      IF(ICAVMOD.EQ.1) THEN
        WRITE(*,1010) FCLTAPER*100.0
 1010   FORMAT(' Closure model: Franc-Michel short cavity',
     &        /' Taper fraction:', F5.1,'% of cavity length')
      ELSE
        WRITE(*,*) ' Closure model: Open cavity / re-entrant jet'
      ENDIF
C
C---- check for cavity: prefer live LCAVZ, fall back to plot-copy
      LPCOPY = .FALSE.
      IF(.NOT.LCAVZ) THEN
        IF(LCAVZP) THEN
          LPCOPY = .TRUE.
        ELSE
          WRITE(*,*)
          WRITE(*,*) ' No cavity present in current solution.'
          RETURN
        ENDIF
      ENDIF
C
      WRITE(*,*)
      DO IS=1, 2
        IF(LPCOPY) THEN
          NCAVDSP = NCAVSP(IS)
        ELSE
          NCAVDSP = NCAVS(IS)
        ENDIF
        IF(NCAVDSP.LE.0) GO TO 100
C
        IF(IS.EQ.1) THEN
          WRITE(*,1100)
 1100     FORMAT(' Suction side (top):')
        ELSE
          WRITE(*,1200)
 1200     FORMAT(' Pressure side (bottom):')
        ENDIF
C
        WRITE(*,1300) NCAVDSP, XCAV1(IS), XCAV2(IS),
     &                SCAVL(IS)/CHORD, HCMAX(IS)/CHORD
 1300   FORMAT('   Cavitated stations:', I5,
     &        /'   Detachment  x/c   =', F10.5,
     &        /'   Closure     x/c   =', F10.5,
     &        /'   Cavity length/c   =', F10.5,
     &        /'   Max thickness/c   =', F10.6)
C
C------ closure-model-specific output
        IF(ICAVMOD .EQ. 1) THEN
C-------- FM model: show taper info
          IF(ICCLOSE(IS).GT.0) THEN
            ITPR = IPAN(ICCLOSE(IS),IS)
            XTAPER = X(ITPR) / CHORD
            WRITE(*,1310) XTAPER, HCLOSE(IS)/CHORD
 1310       FORMAT('   Taper start x/c   =', F10.5,
     &            /'   Closure h/c       =', E12.4,
     &             ' (should be ~0)')
          ENDIF
        ELSE
C-------- RJ model: show jet info
          WRITE(*,1320) HCLOSE(IS)/CHORD
 1320     FORMAT('   Jet thickness/c   =', E12.4)
          IF(HCLOSE(IS).GT.0.0) THEN
            CDJET1 = 2.0*(1.0+SIGMA)*HCLOSE(IS)/CHORD
            WRITE(*,1330) CDJET1
 1330       FORMAT('   Jet CDdrag (side) =', E12.4)
          ENDIF
        ENDIF
C
  100   CONTINUE
      ENDDO
C
C---- drag summary
      WRITE(*,*)
      WRITE(*,1400) CLCAV
 1400 FORMAT(' Total cavity length/c =', F10.5)
C
      IF(ICAVMOD .EQ. 1) THEN
        WRITE(*,1410) CDCAV_P, CDCAV
 1410   FORMAT(' Pressure drag CDcav_p =', F10.6,
     &        /' Total cavity CDcav    =', F10.6)
      ELSE
        WRITE(*,1420) CDCAV_P, CDCAV_J, CDCAV
 1420   FORMAT(' Pressure drag CDcav_p =', F10.6,
     &        /' Jet drag      CDcav_j =', F10.6,
     &        /' Total cavity CDcav    =', F10.6)
      ENDIF
C
      IF(NCAVITER.GT.0) THEN
        WRITE(*,1500) NCAVITER, RCAVEXT
 1500   FORMAT(' Cavity iterations     =', I5,
     &        /' Extent RMS change     =', E12.4)
      ENDIF
C
C---- diagnostic dump: BL variables on suction side
C     Only show when live cavity flags are set (not plot-copy fallback),
C     since BL state (DSTR, THET, UEDG, LCAVP) is restored/cleared
C     after VISCAL and CAVINV.
C     Also skip in inviscid mode where BL variables are all zero.
      IF(.NOT.LPCOPY .AND. LVISC) THEN
        WRITE(*,*)
        WRITE(*,*) '  IBL   x/c        DSTR/c      THET/c',
     &             '      UEDG    LCAVP'
        IS = 1
        DO IBL = 2, MIN(NBL(IS), IBLTE(IS)+5)
          I = IPAN(IBL,IS)
          XOC = X(I)/CHORD
          DOC = DSTR(IBL,IS)/CHORD
          TOC = THET(IBL,IS)/CHORD
          UE  = UEDG(IBL,IS)
          IF(LCAVP(IBL,IS)) THEN
            WRITE(*,1600) IBL, XOC, DOC, TOC, UE, ' *CAV*'
          ELSE
            WRITE(*,1600) IBL, XOC, DOC, TOC, UE, '      '
          ENDIF
 1600     FORMAT(I5, F9.5, 2E13.4, F10.5, A6)
        ENDDO
      ENDIF
C
      RETURN
      END ! CAVSHOW



      SUBROUTINE CAVSYS(IBL, IS)
C-----------------------------------------------------------
C     Sets up the BL Newton system for a cavitated station.
C
C     On the cavity surface there is no wall — the BL has
C     detached and becomes a free shear layer between the
C     outer flow (at Ue = QCAV) and the stagnant cavity vapor.
C
C     Physically:
C       - Cp = -sigma = const  =>  dUe/ds = 0
C       - No wall  =>  Cf = 0  (no skin friction)
C       - No wall dissipation (only outer-layer turbulent mixing)
C
C     Implementation: use the standard turbulent BL equations
C     (BLVAR + BLMID + BLDIF) but with Cf = 0 and wall
C     dissipation removed.  This is analogous to the wake
C     treatment (ITYP=3) but without doubling the dissipation.
C
C     The BL equations then naturally produce:
C       - dTheta/ds ~ 0  (momentum: pressure gradient and Cf both zero)
C       - DSTR grows slowly (free shear layer spreading)
C       - Continuous BL variables at cavity boundaries
C
C     Input:
C       IBL  - current BL station index
C       IS   - side index (1=suction, 2=pressure)
C-----------------------------------------------------------
      IMPLICIT REAL(M)
      INCLUDE 'XBL.INC'
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- compute secondary BL variables using turbulent correlations
      CALL BLVAR(2)
C
C---- guard against pathological secondary variables
C     At cavity stations that were laminar, the BL may be very thin.
C     Clamp HS2 and RT2 to physically reasonable lower bounds.
C     HS2 (density shape parameter) is typically O(1) for turbulent BL.
C     Using too-small HS2 makes dissipation DD ~ 1/HS2 blow up.
      IF(HS2 .LT. 0.15) THEN
        HS2 = 0.15
        HS2_U2 = 0.0
        HS2_T2 = 0.0
        HS2_D2 = 0.0
        HS2_MS = 0.0
        HS2_RE = 0.0
      ENDIF
      IF(RT2 .LT. 40.0) THEN
        RT2 = 40.0
        RT2_U2 = 0.0
        RT2_T2 = 0.0
        RT2_MS = 0.0
        RT2_RE = 0.0
      ENDIF
C
      CALL BLMID(2)
C
C==== Override skin friction: Cf = 0 (no wall) ====
      CF2    = 0.0
      CF2_HK2 = 0.0
      CF2_RT2 = 0.0
      CF2_M2  = 0.0
      CF2_U2 = 0.0
      CF2_T2 = 0.0
      CF2_D2 = 0.0
      CF2_MS = 0.0
      CF2_RE = 0.0
C
C---- also zero midpoint skin friction
      CFM    = 0.0
      CFM_U1 = 0.0
      CFM_T1 = 0.0
      CFM_D1 = 0.0
      CFM_U2 = 0.0
      CFM_T2 = 0.0
      CFM_D2 = 0.0
      CFM_MS = 0.0
      CFM_RE = 0.0
C
C==== Override wall dissipation: keep only outer-layer turbulent part ====
C     BLVAR(2) computed DI2 = wall_part + outer_part.
C     Recompute DI2 as outer-layer only (same as lines 1007-1036 of BLVAR):
C
C---- outer-layer turbulent dissipation:  S2^2 * (0.995-Us) * 2/H*
      DD     =  S2**2 * (0.995-US2) * 2.0/HS2
      DD_HS2 = -S2**2 * (0.995-US2) * 2.0/HS2**2
      DD_US2 = -S2**2               * 2.0/HS2
      DD_S2  =  S2*2.0* (0.995-US2) * 2.0/HS2
C
      DI2    = DD
      DI2_S2 = DD_S2
      DI2_U2 = DD_HS2*HS2_U2 + DD_US2*US2_U2
      DI2_T2 = DD_HS2*HS2_T2 + DD_US2*US2_T2
      DI2_D2 = DD_HS2*HS2_D2 + DD_US2*US2_D2
      DI2_MS = DD_HS2*HS2_MS + DD_US2*US2_MS
      DI2_RE = DD_HS2*HS2_RE + DD_US2*US2_RE
C
C---- add laminar stress contribution to outer layer CD
      DD     =  0.15*(0.995-US2)**2 / RT2  * 2.0/HS2
      DD_US2 = -0.15*(0.995-US2)*2. / RT2  * 2.0/HS2
      DD_HS2 = -DD/HS2
      DD_RT2 = -DD/RT2
C
      DI2    = DI2    + DD
      DI2_U2 = DI2_U2 + DD_HS2*HS2_U2 + DD_US2*US2_U2 + DD_RT2*RT2_U2
      DI2_T2 = DI2_T2 + DD_HS2*HS2_T2 + DD_US2*US2_T2 + DD_RT2*RT2_T2
      DI2_D2 = DI2_D2 + DD_HS2*HS2_D2 + DD_US2*US2_D2
      DI2_MS = DI2_MS + DD_HS2*HS2_MS + DD_US2*US2_MS + DD_RT2*RT2_MS
      DI2_RE = DI2_RE + DD_HS2*HS2_RE + DD_US2*US2_RE + DD_RT2*RT2_RE
C
C---- cap total dissipation to prevent Newton system instability
C     Typical turbulent DI2 is O(0.01).  For cavity free shear layers
C     with thin BL, DI2 can spike.  Cap at 0.20 (with linearization zeroed).
      IF(DI2 .GT. 0.20) THEN
        DI2    = 0.20
        DI2_S2 = 0.0
        DI2_U2 = 0.0
        DI2_T2 = 0.0
        DI2_D2 = 0.0
        DI2_MS = 0.0
        DI2_RE = 0.0
      ENDIF
C
C---- for similarity station
      IF(SIMI) THEN
       DO ICOM=1, NCOM
         COM1(ICOM) = COM2(ICOM)
       ENDDO
      ENDIF
C
C---- build the standard turbulent finite-difference system with Cf=0
      CALL BLDIF(2)
C
C==== Apply compressible-to-incompressible Ue conversion ====
C     (Same as end of BLSYS — needed for correct VM coupling)
      DO K=1, 4
        RES_U1 = VS1(K,4)
        RES_U2 = VS2(K,4)
        RES_MS = VSM(K)
        VS1(K,4) = RES_U1*U1_UEI
        VS2(K,4) =                RES_U2*U2_UEI
        VSM(K)   = RES_U1*U1_MS + RES_U2*U2_MS  + RES_MS
      ENDDO
C
      RETURN
      END ! CAVSYS


      SUBROUTINE CAVMASS(RLXCAV)
C-----------------------------------------------------------
C     Computes cavity mass source MCAV and augments MASS.
C
C     The cavity thickness acts as an additional displacement
C     surface that modifies the inviscid flow through the
C     DIJ influence matrix coupling:
C
C       MASS_total(IBL,IS) = DSTR*UEDG + RLXCAV*HCAV*QCAV
C
C     The HCAV*QCAV term is the cavity mass source.  It
C     represents the flow displaced by the cavity volume.
C
C     RLXCAV is a relaxation factor (0 to 1) that ramps up
C     over outer iterations to avoid destabilizing the BL.
C     The cavity mass source can be 10-20x larger than the
C     BL mass defect, so sudden introduction would cause
C     MRCHDU convergence failures.
C
C     This subroutine:
C       1. Computes MCAV(IBL,IS) = RLXCAV * HCAV(IBL,IS) * QCAV
C          at cavity stations (0 elsewhere)
C       2. Augments MASS(IBL,IS) += MCAV(IBL,IS) at cavity
C          stations
C
C     Should be called AFTER CAVTHICK and BEFORE QVFUE/GAMQV
C     so the augmented MASS propagates to panel velocities.
C
C     Input:
C       RLXCAV - relaxation factor (0.0 to 1.0)
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- zero MCAV first
      DO IS=1, 2
        DO IBL=1, IVX
          MCAV(IBL,IS) = 0.0
        ENDDO
      ENDDO
C
      IF(.NOT.LCAVZ) RETURN
      IF(RLXCAV .LE. 0.0) RETURN
C
cc      WRITE(*,*) 'CAVMASS called: rlx=',RLXCAV,
cc     &           ' LCAVZ=',LCAVZ,' NCAVS=',NCAVS(1),NCAVS(2)
cc      WRITE(*,*) '  ICAV1/2=',ICAV1(1),ICAV2(1),
cc     &           ' QCAV=',QCAV
cc      WRITE(*,*) '  LCAVP check:',
cc     &           LCAVP(ICAV1(1),1),LCAVP(ICAV2(1),1),
cc     &           LCAVP(20,1)
cc      WRITE(*,*) '  HCAV check:',
cc     &           HCAV(ICAV1(1),1),HCAV(20,1),HCAV(ICAV2(1),1)
C
C---- compute cavity mass source at cavity stations
      MCAVMAX = 0.0
      MCAVSUM = 0.0
      DO IS=1, 2
        IF(NCAVS(IS).LE.0) GO TO 100
C
        DO IBL = ICAV1(IS), ICAV2(IS)
          IF(.NOT.LCAVP(IBL,IS)) GO TO 50
C
C-------- cavity mass source = rlx * h_cav * Q_cav
C         QCAV is the cavity surface velocity (constant on cavity)
          MCAV(IBL,IS) = RLXCAV * HCAV(IBL,IS) * QCAV
C
C-------- augment MASS with cavity source
          MASS(IBL,IS) = MASS(IBL,IS) + MCAV(IBL,IS)
C
C-------- track max MCAV for diagnostics
          IF(ABS(MCAV(IBL,IS)).GT.MCAVMAX) THEN
            MCAVMAX = ABS(MCAV(IBL,IS))
          ENDIF
          MCAVSUM = MCAVSUM + MCAV(IBL,IS)
C
   50     CONTINUE
        ENDDO
C
  100   CONTINUE
      ENDDO
C
      IF(MCAVMAX.GT.0.0) THEN
cc        WRITE(*,2200) RLXCAV, MCAVMAX, MCAVSUM
cc 2200   FORMAT('  CAVMASS: rlx=',F5.2,
cc     &         ' max=',E10.3,' sum=',E10.3)
      ENDIF
C
      RETURN
      END ! CAVMASS



      SUBROUTINE CAVINV
C-----------------------------------------------------------
C     Inviscid-only cavitation prediction.
C
C     Called from the OPER command loop when LCAV is active
C     and the solver is in inviscid mode (.NOT.LVISC).
C     Uses the panel solution (GAM, QINV, CPI) to predict
C     cavity extent, thickness, and pressure drag without
C     any boundary-layer solve.
C
C     Steps:
C       1. Set up BL-to-panel mapping (STFIND + IBLPAN)
C          so CAVREGION etc. can use BL station indexing
C       2. Compute CPI from the current panel solution
C       3. Call CAVREGION to detect cavitated panels
C       4. Call CAVTHICK, closure model, CAVDRAG
C       5. Call CAVSHOW to display results
C       6. Save plot-copy variables for CPCAV overlay
C       7. Clean up working cavity arrays
C
C     Requires: SIGMA > 0, panel solution available (LQAIJ)
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- bail out if cavitation not properly set up
      IF(.NOT.LCAV .OR. SIGMA.LE.0.0) RETURN
C
C---- bail out if no panel solution available
      IF(.NOT.LQAIJ) THEN
        WRITE(*,*) 'CAVINV: no panel solution — skipping'
        RETURN
      ENDIF
C
C---- set up BL-to-panel index mapping (needed by CAVREGION etc.)
C     STFIND locates stagnation point, IBLPAN builds IPAN array.
C     These are harmless in inviscid mode and will be redone by
C     VISCAL if the user later switches to viscous mode.
      CALL STFIND
      CALL IBLPAN
C
C---- compute inviscid Cp from panel velocities
      CALL CPCALC(N,QINV,QINF,MINF,CPI)
C
C---- compute cavity speed
      QCAV = QINF * SQRT(1.0 + SIGMA)
C
C---- detect cavitated regions from Cp distribution
C     (CAVREGION uses CPI since LVISC=F or LVCONV=F)
      CALL CAVREGION
C
      IF(.NOT.LCAVZ) THEN
C------ no cavitation detected — clear plot-copy and return
        LCAVZP = .FALSE.
        DO IS=1, 2
          NCAVSP(IS) = 0
          ICAV1P(IS) = 0
          ICAV2P(IS) = 0
        ENDDO
        RETURN
      ENDIF
C
C---- compute cavity thickness from mass-deficit integration
      CALL CAVTHICK
C
C---- apply closure model
      DO IS=1, 2
        IF(NCAVS(IS).GT.0) THEN
          IF(ICAVMOD.EQ.1) THEN
            CALL CAVCLOSE_FM(IS)
          ELSE
            CALL CAVCLOSE_RJ(IS)
          ENDIF
        ENDIF
      ENDDO
C
C---- compute cavity drag
      CALL CAVDRAG
C
C---- display cavity information
      CALL CAVSHOW
C
C---- save plot-copy variables for CPCAV drawing
      LCAVZP = LCAVZ
      DO IS=1, 2
        NCAVSP(IS) = NCAVS(IS)
        ICAV1P(IS) = ICAV1(IS)
        ICAV2P(IS) = ICAV2(IS)
      ENDDO
C
C---- clean up working arrays (extent info preserved in plot-copy)
      DO IS=1, 2
        NCAVS(IS) = 0
        ICAV1(IS) = 0
        ICAV2(IS) = 0
        DO IBL=1, IBLTE(IS)
          LCAVP(IBL,IS) = .FALSE.
        ENDDO
      ENDDO
      NCAVP = 0
      LCAVZ = .FALSE.
C
      RETURN
      END ! CAVINV



      SUBROUTINE CAVINV_FB
C-----------------------------------------------------------
C     Inviscid cavitation prediction with displacement feedback.
C
C     Like CAVINV, this works without a boundary-layer solve.
C     Unlike CAVINV, the cavity thickness feeds back into the
C     inviscid panel solution through the DIJ source-influence
C     matrix, so the pressure distribution outside the cavity
C     is modified by the cavity displacement.
C
C     This provides a converged cavity solution for cases where
C     the viscous V-I iteration may fail (e.g. large cavities).
C
C     The feedback mechanism is the same as VISCAL Pass 2:
C       MASS(IBL,IS) = HCAV(IBL,IS) * QCAV   at cavity stations
C       UEDG = UINV + DIJ * MASS              via UESET
C       QVIS, GAM updated via QVFUE, GAMQV
C       CPI recomputed -> CAVREGION -> CAVTHICK -> repeat
C
C     The cavity extent is iterated until converged (or max
C     iterations reached).  Relaxation is applied to MASS to
C     avoid oscillation.
C
C     Called from OPERi when LCAV is active and .NOT.LVISC.
C
C     Requires: SIGMA > 0, panel solution available (LQAIJ)
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- iteration limits
      INTEGER NCAVMX, NFBMAX
      DATA NCAVMX / 30 /
      DATA NFBMAX  /  5 /
C
C---- saved cavity extent for convergence check
      INTEGER ICAV1S(ISX), ICAV2S(ISX)
      INTEGER NCAVPREV
      LOGICAL LCONV
      INTEGER IC2, NBLEND
      REAL UECAV, FRAC, WGT
C
C---- bail out if cavitation not properly set up
      IF(.NOT.LCAV .OR. SIGMA.LE.0.0) RETURN
C
C---- bail out if no panel solution available
      IF(.NOT.LQAIJ) THEN
        WRITE(*,*) 'CAVINV_FB: no panel solution -- skipping'
        RETURN
      ENDIF
C
      WRITE(*,*)
      WRITE(*,*) 'Inviscid cavity analysis with feedback ...'
C
C---- ensure NW=0 if wake not set up (inviscid mode)
      IF(.NOT.LWAKE) NW = 0
C
C---- set up BL-to-panel index mapping
      CALL STFIND
      CALL IBLPAN
C
C---- set arc length array
      CALL XICALC
C
C---- set inviscid BL edge velocity UINV from panel solution
      CALL UICALC
C
C---- set up source influence matrix DIJ (airfoil + wake if present)
      IF(.NOT.LWDIJ .OR. .NOT.LADIJ) CALL QDCALC
C
C---- initialize UEDG and MASS to inviscid values (no displacement)
      DO IS=1, 2
        DO IBL=1, NBL(IS)
          UEDG(IBL,IS) = UINV(IBL,IS)
          MASS(IBL,IS) = 0.0
        ENDDO
      ENDDO
C
C---- compute cavity speed
      QCAV = QINF * SQRT(1.0 + SIGMA)
C
C---- compute inviscid Cp from panel velocities
      CALL CPCALC(N,QINV,QINF,MINF,CPI)
C
C---- initial cavity detection (no feedback yet)
      CALL CAVREGION
C
      IF(.NOT.LCAVZ) THEN
C------ no cavitation detected at this sigma -- skip feedback
        WRITE(*,*) '  No cavitation at sigma =', SIGMA
        GO TO 900
      ENDIF
C
      WRITE(*,1000) SIGMA, QCAV/QINF
 1000 FORMAT('  sigma =', F8.4, '  Qcav/Qinf =', F8.4)
C
C==== Outer loop: iterate on cavity extent ====
C
      DO 500 ICAVIT=1, NCAVMX
C
C------ save current extent for convergence check
        NCAVPREV = NCAVP
        DO IS=1, 2
          ICAV1S(IS) = ICAV1(IS)
          ICAV2S(IS) = ICAV2(IS)
        ENDDO
C
C------ compute cavity thickness from inviscid Cp
        CALL CAVTHICK
C
C------ feedback sub-loop: ramp cavity displacement into panel solution
C       Same ramped relaxation as VISCAL Pass 2
        DO 400 IFBIT=1, NFBMAX
C
          RLXCAV = FLOAT(IFBIT) / FLOAT(NFBMAX)
C
C-------- set MASS = RLXCAV * HCAV * QCAV at cavity stations
C         zero everywhere else
          DO IS=1, 2
            DO IBL=1, NBL(IS)
              MASS(IBL,IS) = 0.0
            ENDDO
            IF(NCAVS(IS).LE.0) GO TO 310
            DO IBL = ICAV1(IS), ICAV2(IS)
              IF(LCAVP(IBL,IS)) THEN
                MASS(IBL,IS) = RLXCAV * HCAV(IBL,IS) * QCAV
              ENDIF
            ENDDO
C
C---------- No MASS tail past cavity closure.
C           The FM taper (FCLTAPER=0.50) already ensures MASS and
C           dMASS/ds go smoothly to zero at ICAV2.  Adding a tail
C           past ICAV2 was found to introduce Cp oscillations:
C           the extra displacement sources in the tail create an
C           artificial velocity reduction via DIJ that oscillates
C           with the panel discretization.  Without a tail, the
C           closure transition is monotone (a single-panel jump
C           at ICAV2 followed by smooth recovery).
  308       CONTINUE
  310       CONTINUE
          ENDDO
C
C-------- compute new UEDG from UINV + DIJ * MASS
          CALL UESET
C
C-------- override UEDG at cavity stations to QCAV
C         The displacement changes Ue outside the cavity;
C         inside the cavity, Ue is prescribed = QCAV.
          DO IS=1, 2
            IF(NCAVS(IS).LE.0) GO TO 320
            DO IBL = ICAV1(IS), ICAV2(IS)
              IF(LCAVP(IBL,IS)) THEN
                UEDG(IBL,IS) = SIGN(QCAV, UEDG(IBL,IS))
              ENDIF
            ENDDO
  320       CONTINUE
          ENDDO
C
C-------- smooth the velocity jump at cavity closure
C         The DIJ influence from all cavity MASS sources creates
C         a large velocity deficit at post-cavity stations.
C         Inside the cavity UEDG = QCAV (forced).  At the first
C         post-cavity station the full DUI is exposed, causing
C         a Cp jump toward zero.
C
C         Fix: blend UEDG from QCAV at the cavity edge to the
C         raw UESET value over NBLEND stations past closure.
C         This smooths the velocity transition without adding
C         MASS sources.
C
          DO IS=1, 2
            IF(NCAVS(IS).LE.0) GO TO 325
C
            IC2 = ICAV2(IS)
            UECAV = SIGN(QCAV, UEDG(IC2,IS))
C
C---------- blend over NBLEND stations past closure
            NBLEND = 6
            DO IBL = IC2+1, MIN(IC2+NBLEND, NBL(IS))
C
C------------ blending weight: cos^2 decay from 1 to 0
              FRAC = FLOAT(IBL - IC2) / FLOAT(NBLEND + 1)
              WGT  = COS(0.5*3.14159265 * FRAC)**2
C
C------------ blended UEDG
              UEDG(IBL,IS) = WGT * UECAV
     &                     + (1.0 - WGT) * UEDG(IBL,IS)
            ENDDO
C
  325       CONTINUE
          ENDDO
C
C-------- propagate to panel velocities and circulation
          CALL QVFUE
          CALL GAMQV
C
  400   CONTINUE
C
C------ recompute Cp from the feedback-modified panel velocities
C       Use GAM (= QVIS after GAMQV) which includes cavity displacement
        CALL CPCALC(N,GAM,QINF,MINF,CPI)
C
C------ recompute CL, CM from modified circulation
        CALL CLCALC(N,X,Y,GAM,GAM_A,ALFA,MINF,QINF,
     &              XCMREF,YCMREF,
     &              CL,CM,CDP, CL_ALF,CL_MSQ)
C
C------ re-detect cavity from updated Cp
        CALL CAVREGION
C
        IF(.NOT.LCAVZ) THEN
C-------- cavity disappeared after feedback -- report and exit
          WRITE(*,*) '  Cavity vanished during feedback iteration'
          GO TO 900
        ENDIF
C
C------ report iteration
        WRITE(*,1100) ICAVIT, NCAVS(1), NCAVS(2), CL
 1100   FORMAT('  Iter', I3, ':  Ncav=', I4, '/', I4,
     &         '  CL=', F8.4)
C
C------ check convergence: cavity extent unchanged
        LCONV = .TRUE.
        DO IS=1, 2
          IF(ICAV1(IS).NE.ICAV1S(IS)) LCONV = .FALSE.
          IF(ICAV2(IS).NE.ICAV2S(IS)) LCONV = .FALSE.
        ENDDO
C
        IF(LCONV) THEN
          WRITE(*,*) '  Cavity extent converged.'
          NCAVITER = ICAVIT
          RCAVEXT  = 0.0
          LCAVCONV = .TRUE.
          GO TO 600
        ENDIF
C
  500 CONTINUE
C
C---- did not converge
      WRITE(*,*) '  Cavity extent not converged after', NCAVMX,
     &           ' iterations'
      NCAVITER = NCAVMX
      LCAVCONV = .FALSE.
C
C==== Post-process converged (or best) cavity solution ====
C
  600 CONTINUE
C
C---- final cavity thickness
      CALL CAVTHICK
C
C---- apply closure model
      DO IS=1, 2
        IF(NCAVS(IS).GT.0) THEN
          IF(ICAVMOD.EQ.1) THEN
            CALL CAVCLOSE_FM(IS)
          ELSE
            CALL CAVCLOSE_RJ(IS)
          ENDIF
        ENDIF
      ENDDO
C
C---- compute cavity drag
      CALL CAVDRAG
C
C---- display cavity information
      CALL CAVSHOW
C
C---- save plot-copy variables for CPCAV drawing
  900 CONTINUE
      LCAVZP = LCAVZ
      DO IS=1, 2
        NCAVSP(IS) = NCAVS(IS)
        ICAV1P(IS) = ICAV1(IS)
        ICAV2P(IS) = ICAV2(IS)
      ENDDO
C
C---- clean up working arrays (extent info preserved in plot-copy)
      DO IS=1, 2
        NCAVS(IS) = 0
        ICAV1(IS) = 0
        ICAV2(IS) = 0
        DO IBL=1, IBLTE(IS)
          LCAVP(IBL,IS) = .FALSE.
        ENDDO
      ENDDO
      NCAVP = 0
      LCAVZ = .FALSE.
C
C---- zero out MASS so it doesn't contaminate a later viscous solve
      DO IS=1, 2
        DO IBL=1, NBL(IS)
          MASS(IBL,IS) = 0.0
        ENDDO
      ENDDO
C
C---- reset LIPAN so VISCAL will redo IBLPAN+IBLSYS if user switches to VISC
      LIPAN = .FALSE.
C
      RETURN
      END ! CAVINV_FB



      SUBROUTINE CPCAV
C-----------------------------------------------------------
C     Draws cavity overlay on the Cp vs x plot.
C
C     Draws:
C       1. Dashed horizontal line at Cp = -sigma across plot
C       2. Thick colored line segment on Cp = -sigma over
C          the actual cavitated region (per side)
C       3. Short vertical tick marks at cavity detachment
C          and closure x/c locations
C       4. Label "s = X.XX" near the line
C
C     All scaling variables (PFAC, FACA, XOFA, CH) are
C     accessed from the XFOIL.INC /CR13/ common block.
C
C     Must be called after Cp curves are drawn and before
C     PLFLUSH, while the re-origin from CPX is still active.
C-----------------------------------------------------------
      INCLUDE 'XFOIL.INC'
      INCLUDE 'XCAV.INC'
C
C---- bail out if cavitation not active or no cavity present
      IF(.NOT.LCAV)  RETURN
      IF(SIGMA.LE.0.0) RETURN
C
C---- Cp level on the cavity
      CPSIG = -SIGMA
C
C---- y-coordinate of the Cp = -sigma line in plot space
C     (y_plot = -PFAC * Cp, so y_plot = -PFAC*(-sigma) = PFAC*sigma)
      YCAV = -PFAC * CPSIG
C
C---- save current color
      CALL GETCOLOR(ICOL0)
C
C==== 1. Dashed line at Cp = -sigma across full plot ====
C     Use the same DASH routine as the sonic Cp line
      IF(CPSIG .GE. CPMIN .AND. CPSIG .LE. CPMAX) THEN
        CALL DASH(0.0, 1.0, YCAV)
      ENDIF
C
C==== 2. Thick colored line segment over cavitated region ====
      IF(.NOT.LCAVZP) GO TO 90
C
      DO 80 IS=1, 2
        IF(NCAVSP(IS).LE.0) GO TO 80
C
C------ set color: suction side = green(ICOLS(1)), pressure = red(ICOLS(2))
        CALL NEWCOLOR(ICOLS(IS))
        CALL NEWPEN(3)
C
C------ x-coordinates of cavity endpoints in plot space
        X1PLT = (XCAV1(IS)*CHORD + XOFA) * FACA
        X2PLT = (XCAV2(IS)*CHORD + XOFA) * FACA
C
C------ draw thick line from detachment to closure at Cp = -sigma
        CALL PLOT(X1PLT, YCAV, 3)
        CALL PLOT(X2PLT, YCAV, 2)
C
C==== 3. Vertical tick marks at detachment and closure ====
C       Use magenta to match the cavity thickness profile on the airfoil
        CALL NEWCOLORNAME('magenta')
        CALL NEWPEN(2)
        TICK = 0.01
C
C------ detachment tick
        CALL PLOT(X1PLT, YCAV-TICK, 3)
        CALL PLOT(X1PLT, YCAV+TICK, 2)
C
C------ closure tick
        CALL PLOT(X2PLT, YCAV-TICK, 3)
        CALL PLOT(X2PLT, YCAV+TICK, 2)
C
   80 CONTINUE
C
C==== 4. Label "s = X.XX" near the Cp = -sigma line ====
      CALL NEWCOLOR(ICOL0)
      CALL NEWPEN(2)
C
C---- position label at right side of plot
      CCH = 0.7*CH
      XLAB = 0.80
      YLAB = YCAV + 1.5*CCH
C
C---- draw "s" (sigma) label using math font, then "= X.XX"
      CALL PLMATH(XLAB        , YLAB, 1.0*CCH, 's', 0.0, 1)
      CALL PLCHAR(XLAB+1.0*CCH, YLAB, CCH, ' = ', 0.0, 3)
      CALL PLNUMB(XLAB+4.0*CCH, YLAB, CCH, SIGMA, 0.0, 3)
C
C
C==== 5. Draw cavity thickness profile on airfoil body ====
C     Offset each cavitated panel node along its outward normal
C     by HCAV(IBL,IS), then draw as a thick colored polyline.
C     Connect endpoints back to the airfoil surface to close
C     the cavity shape visually.
C     (Same technique as CPDISP for displacement surface)
C     Use plot-copy variables (LCAVZP, NCAVSP, ICAV1P, ICAV2P)
C     since the operational cavity state is cleared after the
C     cavity iteration to keep Phase 1 clean for the next run.
C
      IF(.NOT.LCAVZP) GO TO 90
C
      DO 85 IS=1, 2
        IF(NCAVSP(IS).LE.0) GO TO 85
C
C------ use magenta for the cavity profile to distinguish from
C       the thin displacement surface drawn by CPDISP
        CALL NEWCOLORNAME('magenta')
        CALL NEWPEN(3)
C
C------ start at the airfoil surface at detachment point (pen-up)
        I = IPAN(ICAV1P(IS),IS)
        XPLT = (X(I)+XOFA)*FACA
        YPLT = (Y(I)+YOFA)*FACA
        CALL PLOT(XPLT, YPLT, 3)
C
C------ draw offset curve: detachment to closure (pen-down)
        DO IBL = ICAV1P(IS), ICAV2P(IS)
          I = IPAN(IBL,IS)
          HCAVI = HCAV(IBL,IS)
          IF(HCAVI .LT. 0.0) HCAVI = 0.0
          XPLT = (X(I) + NX(I)*HCAVI + XOFA)*FACA
          YPLT = (Y(I) + NY(I)*HCAVI + YOFA)*FACA
          CALL PLOT(XPLT, YPLT, 2)
        ENDDO
C
C------ close back to airfoil surface at closure point (pen-down)
        I = IPAN(ICAV2P(IS),IS)
        XPLT = (X(I)+XOFA)*FACA
        YPLT = (Y(I)+YOFA)*FACA
        CALL PLOT(XPLT, YPLT, 2)
C
C------ draw airfoil surface back to detachment to close the shape
C       (thin pen for the baseline contour portion)
        CALL NEWPEN(1)
        IF(ICAV2P(IS) .GT. ICAV1P(IS)+1) THEN
          DO IBL = ICAV2P(IS)-1, ICAV1P(IS), -1
            I = IPAN(IBL,IS)
            XPLT = (X(I)+XOFA)*FACA
            YPLT = (Y(I)+YOFA)*FACA
            CALL PLOT(XPLT, YPLT, 2)
          ENDDO
        ENDIF
C
   85 CONTINUE
C
C---- restore default pen
   90 CONTINUE
      CALL NEWCOLOR(ICOL0)
      CALL NEWPEN(2)
C
      RETURN
      END ! CPCAV
