C
C====  Cavitation modeling INCLUDE file for XFOIL  =====
C
C     Sheet cavitation model with two closure models:
C       ICAVMOD = 1:  Franc-Michel short cavity
C       ICAVMOD = 2:  Open cavity with re-entrant jet
C
C---- Cavitation state variables
C
C   SIGMA     cavitation number  (P_inf - P_v) / (0.5 * rho * V_inf^2)
C   QCAV      cavity surface velocity  = QINF * sqrt(1 + SIGMA)
C   CLCAV     cavity length / chord  (total, suction + pressure)
C   CDCAV     cavity pressure drag coefficient
C   SIGMIN    inception sigma  = -Cp_min  (from current solution)
C
C   LCAV      .TRUE. if cavitation modeling is active
C   LCAVZ     .TRUE. if a cavity currently exists in the solution
C   LCAVP(..) .TRUE. if panel I on side IS is cavitated (BL index, side)
C
C   ICAVMOD   closure model:  1 = Franc-Michel,  2 = re-entrant jet
C   ICAV1(.)  BL station index of cavity detachment on each side
C   ICAV2(.)  BL station index of cavity closure on each side
C   NCAVS(.)  number of cavitated BL stations on each side
C   NCAVP     number of cavitated panels (total, for AIC rebuild)
C
C   XCAV1(.)  x/c of cavity detachment on each side
C   XCAV2(.)  x/c of cavity closure on each side
C   SCAVL(.)  cavity arc length on each side
C   HCAV(..)  cavity thickness at each BL station on each side
C   HCAVP(.)  cavity thickness at each panel node
C
C   NCAVITER  number of outer cavity-extent iterations performed
C   RCAVEXT   RMS change in cavity extent from previous outer iteration
C   LCAVCONV  .TRUE. if cavity extent has converged
C
C   PVAP      vapor pressure (Pa), default 2337 for water at 20C
C   PINF      freestream static pressure (Pa), default 101325
C   RHOL      liquid density (kg/m^3), default 998.2 for water at 20C
C
C   HCLOSE(.) cavity thickness at closure on each side
C   HCMAX(.)  maximum cavity thickness on each side
C   CDCAV_P   pressure drag component of CDcav
C   CDCAV_J   jet momentum drag component (RJ model only)
C   FCLTAPER  fraction of cavity length used for FM closure taper
C   ICCLOSE(.) BL station where FM taper begins on each side
C
C   MCAV(..)  cavity mass source = HCAV * QCAV at each BL station
C             This feeds back into the inviscid solution via DIJ coupling.
C             Added to MASS before QVFUE/GAMQV to modify outer velocities.
C
C---- LCAV and SIGMA live in XFOIL.INC (/CL01/ and /CR09/ respectively)
C     They must NOT be re-declared here.
      LOGICAL LCAVZ, LCAVCONV, LCAVZP
      LOGICAL LCAVP(IVX,ISX)
C
C---- MCAV must be explicitly typed REAL because its name starts
C     with 'M' (in the implicit I-N integer range).  Without this,
C     subroutines lacking IMPLICIT REAL(M) would see MCAV as INTEGER,
C     causing a COMMON block size mismatch with -fdefault-real-8.
      REAL MCAV
C
      COMMON /CCAVR/ QCAV, CLCAV, CDCAV, SIGMIN,
     &               XCAV1(ISX), XCAV2(ISX), SCAVL(ISX),
     &               HCAV(IVX,ISX), HCAVP(IQX),
     &               RCAVEXT, PVAP, PINF, RHOL,
     &               HCLOSE(ISX), HCMAX(ISX),
     &               CDCAV_P, CDCAV_J, FCLTAPER,
     &               MCAV(IVX,ISX)
      COMMON /CCAVI/ ICAVMOD, ICAV1(ISX), ICAV2(ISX),
     &               NCAVS(ISX), NCAVP, NCAVITER,
     &               ICCLOSE(ISX),
     &               NCAVSP(ISX), ICAV1P(ISX), ICAV2P(ISX)
      COMMON /CCAVL/ LCAVZ, LCAVCONV, LCAVP, LCAVZP
C
