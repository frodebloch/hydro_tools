"""
Validation and demonstration of the propeller model and optimiser.

Reproduces calculations from the R script to verify correctness,
then demonstrates the power and fuel consumption optimisers.
"""

import math
import numpy as np

from propeller_model import load_c4_55_data, load_c_series_data, CSeriesPropeller
from optimiser import (
    find_optimal_operating_point,
    find_min_fuel_operating_point,
    generate_optimal_combinator,
    fit_combinator_curve,
    SimpleDieselEngine,
    make_man_l27_38,
)


DATA_PATH_C440 = "/home/blofro/src/prop_model/c4_40.dat"
DATA_PATH_C455 = "/home/blofro/src/prop_model/C4_55.dat"
DATA_PATH_C470 = "/home/blofro/src/prop_model/c4_70.dat"


def validate_propeller_model():
    """Validate against known R code behaviour."""
    print("=" * 70)
    print("PROPELLER MODEL VALIDATION")
    print("=" * 70)

    data = load_c4_55_data(DATA_PATH_C455)
    print(f"\nData loaded: {len(data.design_pitches)} design pitches, "
          f"{len(data.pitches)} pitch values, "
          f"{data.n_coefficients} Fourier coefficients")
    print(f"  Design pitches: {data.design_pitches}")
    print(f"  Pitch values:   {data.pitches}")

    # Create propeller matching CombinatorTheWorld: D=4.6m, design_pitch=1.2
    prop = CSeriesPropeller(data, design_pitch=1.2, diameter=4.6, rho=1025.0)
    print(f"\nPropeller: D={prop.diameter}m, design P/D={prop.design_pitch}")

    # --- Test: Bollard pull (Va=0) at known pitch and rpm ---
    # From CombinatorTheWorld: pitch_ratio around 0.7, rpm=100.6
    n = 100.6 / 60.0  # rev/s
    Va = 0.0
    pitch = 0.8  # tabulated value

    T = prop.thrust(pitch, n, Va)
    P = prop.power(pitch, n, Va)
    Q = prop.torque(pitch, n, Va)
    print(f"\n--- Bollard pull test: pitch={pitch}, rpm={n*60:.1f}, Va={Va} ---")
    print(f"  Thrust: {T/1000:.1f} kN")
    print(f"  Torque: {Q/1000:.1f} kNm")
    print(f"  Power:  {P/1000:.1f} kW")

    # --- Test: Free-running at Va > 0 ---
    Va = 5.0  # m/s (~10 knots)
    n = 2.0  # rev/s = 120 rpm
    pitch = 1.0
    T = prop.thrust(pitch, n, Va)
    P = prop.power(pitch, n, Va)
    Q = prop.torque(pitch, n, Va)
    J = Va / (n * prop.diameter)
    eta0 = prop.eta0(pitch, n, Va)
    print(f"\n--- Free running: pitch={pitch}, rpm={n*60:.0f}, Va={Va} m/s ---")
    print(f"  J  = {J:.3f}")
    print(f"  T  = {T/1000:.1f} kN")
    print(f"  Q  = {Q/1000:.1f} kNm")
    print(f"  P  = {P/1000:.1f} kW")
    print(f"  eta0 = {eta0:.4f}")

    # --- KT/KQ curves for a range of J ---
    print(f"\n--- KT/KQ table for design_pitch=1.2, pitch=1.0 ---")
    print(f"  {'J':>6s}  {'KT':>10s}  {'10KQ':>10s}  {'eta0':>8s}")
    for J in np.arange(0.1, 1.3, 0.1):
        n_test = Va / (J * prop.diameter)
        T_test = prop.thrust(1.0, n_test, Va)
        Q_test = prop.torque(1.0, n_test, Va)
        KT = prop.KT(T_test, n_test)
        KQ = prop.KQ(Q_test, n_test)
        eta = prop.eta0(1.0, n_test, Va)
        print(f"  {J:6.2f}  {KT:10.5f}  {10*KQ:10.5f}  {eta:8.4f}")


def demo_power_optimiser():
    """Demonstrate minimum power optimiser."""
    print("\n" + "=" * 70)
    print("MINIMUM POWER OPTIMISER")
    print("=" * 70)

    data = load_c4_55_data(DATA_PATH_C455)
    prop = CSeriesPropeller(data, design_pitch=1.2, diameter=4.6, rho=1025.0)

    Va = 5.0  # m/s
    T_required = 200e3  # 200 kN

    print(f"\nRequired: T={T_required/1000:.0f} kN at Va={Va:.1f} m/s")
    print(f"Propeller: D={prop.diameter}m, design P/D={prop.design_pitch}")

    result = find_optimal_operating_point(
        prop, Va, T_required,
        max_rpm=200.0,
        max_torque=500e3,
    )

    if result.found:
        print(f"\n  Optimal pitch:  {result.pitch:.3f}")
        print(f"  Shaft speed:    {result.rpm:.1f} rpm ({result.n:.3f} rev/s)")
        print(f"  Thrust:         {result.thrust/1000:.1f} kN")
        print(f"  Torque:         {result.torque/1000:.1f} kNm")
        print(f"  Shaft power:    {result.power_kw:.1f} kW")
        print(f"  eta_0:          {result.eta0:.4f}")
    else:
        print("  No feasible operating point found!")

    # Show all pitch options — tabulated values first, then PCHIP fine grid
    from optimiser import _find_n_for_thrust

    def _show_pitch_sweep(pitches, label):
        print(f"\n  --- {label} ---")
        print(f"  {'P/D':>6s}  {'rpm':>8s}  {'Power kW':>10s}  {'Torque kNm':>12s}  {'eta0':>8s}")
        for pitch in pitches:
            if pitch <= 0:
                continue
            n = _find_n_for_thrust(prop, pitch, Va, T_required, n_min=0.1, n_max=200/60)
            if n is None:
                continue
            rpm = n * 60
            Q = prop.torque(pitch, n, Va)
            P = prop.power(pitch, n, Va)
            eta = prop.eta0(pitch, n, Va)
            print(f"  {pitch:6.2f}  {rpm:8.1f}  {P/1000:10.1f}  {Q/1000:12.1f}  {eta:8.4f}")

    _show_pitch_sweep(prop.pitch_table, "Tabulated pitch values only")

    fine_pitches = np.arange(0.1, 1.61, 0.10)
    _show_pitch_sweep(fine_pitches, "PCHIP-interpolated fine grid (step=0.10)")


def demo_fuel_optimiser():
    """Demonstrate minimum fuel consumption optimiser."""
    print("\n" + "=" * 70)
    print("MINIMUM FUEL CONSUMPTION OPTIMISER")
    print("=" * 70)

    data = load_c4_55_data(DATA_PATH_C455)
    prop = CSeriesPropeller(data, design_pitch=1.2, diameter=4.6, rho=1025.0)

    # Engine: 3000 kW MCR, max 150 rpm (direct drive to propeller)
    engine = SimpleDieselEngine(
        max_power_kw=3000.0,
        max_engine_rpm=150.0,
        min_engine_rpm=60.0,
        sfoc_min=185.0,      # g/kWh
        load_optimal=0.80,
        sfoc_curvature=0.5,
    )

    Va = 5.0  # m/s
    T_required = 200e3  # 200 kN
    aux_load = 200.0  # 200 kW auxiliary electrical load

    print(f"\nRequired: T={T_required/1000:.0f} kN at Va={Va:.1f} m/s")
    print(f"Engine:   MCR={engine.max_power_kw:.0f} kW, "
          f"rpm range={engine.min_engine_rpm:.0f}-{engine.max_engine_rpm:.0f}")
    print(f"Auxiliary load: {aux_load:.0f} kW")

    # Case 1: No auxiliary load
    print("\n--- Case 1: No auxiliary load ---")
    result = find_min_fuel_operating_point(
        prop, Va, T_required, engine,
        auxiliary_power_kw=0.0,
    )
    if result.found:
        print(f"  Optimal pitch:   {result.pitch:.3f}")
        print(f"  Shaft rpm:       {result.rpm:.1f}")
        print(f"  Shaft power:     {result.power_kw:.1f} kW")
        print(f"  Engine power:    {result.engine_power_kw:.1f} kW")
        print(f"  Fuel rate:       {result.fuel_rate:.1f} g/h")
        print(f"  SFOC:            {result.fuel_rate/result.engine_power_kw:.1f} g/kWh")
        print(f"  eta_0:           {result.eta0:.4f}")

    # Case 2: With auxiliary load
    print(f"\n--- Case 2: With {aux_load:.0f} kW auxiliary load ---")
    result = find_min_fuel_operating_point(
        prop, Va, T_required, engine,
        auxiliary_power_kw=aux_load,
    )
    if result.found:
        print(f"  Optimal pitch:   {result.pitch:.3f}")
        print(f"  Shaft rpm:       {result.rpm:.1f}")
        print(f"  Shaft power:     {result.power_kw:.1f} kW")
        print(f"  Engine power:    {result.engine_power_kw:.1f} kW (incl {aux_load:.0f} kW aux)")
        print(f"  Fuel rate:       {result.fuel_rate:.1f} g/h")
        print(f"  SFOC:            {result.fuel_rate/result.engine_power_kw:.1f} g/kWh")
        print(f"  eta_0:           {result.eta0:.4f}")

    # Case 3: Sweep over speeds
    print(f"\n--- Optimal operating points across speed range ---")
    print(f"  {'Va m/s':>7s}  {'T kN':>6s}  {'P/D':>5s}  {'rpm':>6s}  "
          f"{'P_shaft kW':>10s}  {'P_eng kW':>10s}  {'Fuel g/h':>9s}  {'eta0':>6s}")
    for Va_test in np.arange(1.0, 8.0, 1.0):
        # Scale thrust roughly with Va^2 (drag-like)
        T_test = 200e3 * (Va_test / 5.0) ** 2
        result = find_min_fuel_operating_point(
            prop, Va_test, T_test, engine,
            auxiliary_power_kw=aux_load,
        )
        if result.found:
            print(f"  {Va_test:7.1f}  {T_test/1000:6.0f}  {result.pitch:5.2f}  "
                  f"{result.rpm:6.1f}  {result.power_kw:10.1f}  "
                  f"{result.engine_power_kw:10.1f}  {result.fuel_rate:9.1f}  "
                  f"{result.eta0:6.4f}")
        else:
            print(f"  {Va_test:7.1f}  {T_test/1000:6.0f}  -- no feasible point --")


def demo_bar_interpolation():
    """Demonstrate blade area ratio interpolation between C4-55 and C4-70."""
    print("\n" + "=" * 70)
    print("BLADE AREA RATIO INTERPOLATION")
    print("=" * 70)

    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)

    print(f"\nC4-55: {len(data_55.pitches)} pitch values, range [{data_55.pitches[0]}, {data_55.pitches[-1]}]")
    print(f"C4-70: {len(data_70.pitches)} pitch values, range [{data_70.pitches[0]}, {data_70.pitches[-1]}]")

    D = 4.6
    dp = 1.2
    Va = 5.0
    n = 2.0  # rev/s = 120 rpm

    # Create single-BAR models for comparison
    prop_55 = CSeriesPropeller(data_55, design_pitch=dp, diameter=D, area_ratio=0.55)
    prop_70 = CSeriesPropeller(data_70, design_pitch=dp, diameter=D, area_ratio=0.70)

    # Create multi-BAR models at 0.55, 0.625, and 0.70
    bar_data = {0.55: data_55, 0.70: data_70}
    prop_mid = CSeriesPropeller(bar_data, design_pitch=dp, diameter=D, area_ratio=0.625)

    print(f"\nPropeller: D={D}m, design P/D={dp}")
    print(f"Operating: Va={Va} m/s, rpm={n*60:.0f}")

    # Compare KT/KQ curves at different BARs
    print(f"\n--- KT comparison: BAR=0.55 vs 0.625 (interpolated) vs 0.70, pitch=1.0 ---")
    print(f"  {'J':>6s}  {'KT 0.55':>10s}  {'KT 0.625':>10s}  {'KT 0.70':>10s}  {'delta %':>9s}")
    for J in np.arange(0.1, 1.2, 0.1):
        n_test = Va / (J * D)
        T_55 = prop_55.thrust(1.0, n_test, Va)
        T_70 = prop_70.thrust(1.0, n_test, Va)
        T_mid = prop_mid.thrust(1.0, n_test, Va)
        KT_55 = prop_55.KT(T_55, n_test)
        KT_70 = prop_70.KT(T_70, n_test)
        KT_mid = prop_mid.KT(T_mid, n_test)
        delta = 100.0 * (KT_70 - KT_55) / abs(KT_55) if abs(KT_55) > 1e-10 else 0.0
        print(f"  {J:6.2f}  {KT_55:10.5f}  {KT_mid:10.5f}  {KT_70:10.5f}  {delta:8.1f}%")

    # Optimiser comparison across BARs
    T_required = 200e3
    print(f"\n--- Power optimiser: T={T_required/1000:.0f} kN at Va={Va} m/s ---")
    print(f"  {'BAR':>6s}  {'P/D':>6s}  {'rpm':>8s}  {'Power kW':>10s}  {'eta0':>8s}")
    for bar_val in [0.55, 0.60, 0.625, 0.65, 0.70]:
        prop = CSeriesPropeller(bar_data, design_pitch=dp, diameter=D, area_ratio=bar_val)
        result = find_optimal_operating_point(
            prop, Va, T_required,
            max_rpm=200.0,
            max_torque=500e3,
        )
        if result.found:
            print(f"  {bar_val:6.3f}  {result.pitch:6.3f}  {result.rpm:8.1f}  "
                  f"{result.power_kw:10.1f}  {result.eta0:8.4f}")
        else:
            print(f"  {bar_val:6.3f}  -- no feasible point --")


def demo_muzzle_diagram():
    """Demonstrate fuel optimiser with MAN L27/38 muzzle diagram engine model.

    Uses actual hull resistance and efficiency data from the service prediction.
    Compares our propeller model output against the prediction values.
    """
    print("\n" + "=" * 70)
    print("MUZZLE DIAGRAM ENGINE MODEL: MAN L27/38")
    print("=" * 70)

    engine = make_man_l27_38()
    gear_ratio = 800.0 / 117.6  # 6.803

    print(f"\nEngine:     {engine.name}")
    print(f"MCR:        {engine.max_power_kw:.0f} kW at {engine.max_engine_rpm:.0f} rpm")
    print(f"Min rpm:    {engine.min_engine_rpm:.0f}")
    print(f"Gear ratio: {gear_ratio:.3f}")
    print(f"Max prop rpm: {engine.max_engine_rpm / gear_ratio:.1f}")

    # Validate SFOC at a few known points from the diagram
    print(f"\n--- SFOC spot checks (from diagram contours) ---")
    print(f"  {'Eng RPM':>8s}  {'Power kW':>10s}  {'SFOC g/kWh':>12s}  {'Expected':>10s}")
    checks = [
        (620, 1400, "181-182"),
        (640, 1600, "181-182"),
        (580, 1285, "181-182"),
        (510, 730,  "185-187"),
        (800, 2920, "187-190"),
        (700, 2079, "182-183"),
    ]
    for rpm, pwr, expected in checks:
        s = engine.sfoc(pwr, rpm)
        print(f"  {rpm:8.0f}  {pwr:10.0f}  {s:12.1f}  {expected:>10s}")

    # ----------------------------------------------------------------
    # Service prediction data (from hull resistance calculation)
    # Allowance = 15% on PD-trial, Head wind 0 m/s, BF 0
    # ----------------------------------------------------------------
    # Columns: V [kts], FN, RT [kN], T [kN], PE [kW], PD [kW], ETAD, N [RPM]
    service_pred = np.array([
        [ 8.00, .1200,  67,  77.6,  275,  341, .807,  64.2],
        [ 8.50, .1275,  77,  89.8,  337,  416, .809,  68.6],
        [ 9.00, .1350,  88, 102.9,  407,  502, .810,  72.9],
        [ 9.50, .1425,  99, 116.5,  485,  598, .811,  77.2],
        [10.00, .1500, 111, 130.3,  569,  702, .811,  81.5],
        [10.50, .1575, 122, 144.2,  660,  814, .811,  85.6],
        [11.00, .1650, 134, 158.6,  759,  936, .810,  89.8],
        [11.50, .1725, 146, 173.5,  865, 1068, .810,  93.9],
        [12.00, .1800, 159, 188.8,  979, 1211, .809,  98.0],
        [12.50, .1875, 171, 204.5, 1100, 1362, .807, 102.0],
        [13.00, .1950, 184, 220.6, 1229, 1523, .807, 106.0],
        [13.50, .2025, 197, 237.3, 1367, 1696, .806, 110.0],
        [14.00, .2099, 211, 254.7, 1516, 1884, .805, 114.0],
        [14.50, .2174, 225, 273.1, 1677, 2092, .802, 118.1],
        [15.00, .2249, 240, 292.7, 1852, 2325, .797, 122.4],
        [15.50, .2324, 257, 315.2, 2050, 2591, .791, 126.8],
        [16.00, .2399, 276, 341.2, 2276, 2897, .786, 131.5],
    ])

    # Hull efficiency elements (trial conditions)
    # Columns: V [kts], FN, ADVC (J), THDF (t), WFT (w), ETAH, ETAO, ETAR, ETAD, CTH
    hull_eff = np.array([
        [ 8.00, .1200, .628, .140, .239, 1.130, .725, 1.000, .820, .753],
        [ 8.50, .1275, .624, .142, .240, 1.129, .723, 1.008, .822, .775],
        [ 9.00, .1350, .620, .145, .242, 1.127, .721, 1.015, .824, .795],
        [ 9.50, .1425, .618, .148, .243, 1.124, .719, 1.021, .825, .810],
        [10.00, .1500, .616, .151, .243, 1.121, .718, 1.025, .825, .818],
        [10.50, .1575, .616, .153, .242, 1.118, .718, 1.028, .825, .820],
        [11.00, .1650, .615, .155, .242, 1.115, .717, 1.030, .824, .822],
        [11.50, .1725, .615, .157, .242, 1.112, .717, 1.033, .824, .823],
        [12.00, .1800, .615, .160, .242, 1.108, .718, 1.035, .823, .822],
        [12.50, .1875, .616, .164, .242, 1.103, .718, 1.037, .821, .820],
        [13.00, .1950, .616, .167, .242, 1.099, .718, 1.040, .821, .819],
        [13.50, .2025, .616, .170, .243, 1.095, .718, 1.043, .820, .817],
        [14.00, .2099, .617, .173, .242, 1.091, .718, 1.045, .819, .815],
        [14.50, .2174, .617, .177, .242, 1.086, .719, 1.045, .815, .813],
        [15.00, .2249, .617, .180, .240, 1.079, .719, 1.045, .810, .811],
        [15.50, .2324, .617, .184, .239, 1.071, .718, 1.046, .805, .815],
        [16.00, .2399, .615, .190, .238, 1.063, .717, 1.048, .799, .827],
    ])

    # Extract columns for easy access
    sp_V     = service_pred[:, 0]  # ship speed [kts]
    sp_RT    = service_pred[:, 2]  # total resistance [kN]
    sp_T     = service_pred[:, 3]  # thrust [kN]
    sp_PE    = service_pred[:, 4]  # effective power [kW]
    sp_PD    = service_pred[:, 5]  # delivered power [kW]
    sp_ETAD  = service_pred[:, 6]  # propulsive efficiency
    sp_N     = service_pred[:, 7]  # propeller RPM

    he_ADVC  = hull_eff[:, 2]  # advance coefficient J
    he_THDF  = hull_eff[:, 3]  # thrust deduction factor t
    he_WFT   = hull_eff[:, 4]  # wake fraction w
    he_ETAO  = hull_eff[:, 6]  # open-water efficiency
    he_ETAR  = hull_eff[:, 7]  # relative rotative efficiency

    # ----------------------------------------------------------------
    # Set up propeller model
    # ----------------------------------------------------------------
    D = 4.66  # Back-calculated from service prediction J values (pending confirmation)
    BAR = 0.432  # Blade area ratio from the actual propeller
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    prop = CSeriesPropeller(bar_data, design_pitch=0.771, diameter=D, area_ratio=BAR, rho=1025.0)

    print(f"\nPropeller:  D={prop.diameter}m, design P/D={prop.design_pitch}, BAR={BAR}")
    print(f"Note: design P/D=0.771 extrapolated below lowest tabulated value (0.8) via PCHIP.")
    print(f"Note: D=4.66m back-calculated from service prediction J values (pending).")
    print(f"BAR interpolation: C4-40/C4-55/C4-70, BAR=0.432 interpolated between 0.40 and 0.55.")

    # ----------------------------------------------------------------
    # Part 1: Validate propeller model against the service prediction
    #
    # For each speed, use the prediction's thrust T, wake fraction w,
    # and prop RPM N to compute our model's power and eta0, and compare.
    # ----------------------------------------------------------------
    print(f"\n--- Propeller model validation against service prediction ---")
    print(f"  {'V kn':>5s}  {'T kN':>5s}  {'w':>5s}  {'t':>5s}  "
          f"{'N rpm':>6s}  {'PD pred':>8s}  {'PD model':>9s}  {'err%':>6s}  "
          f"{'eta0 pred':>10s}  {'eta0 model':>11s}  {'J pred':>7s}  {'J model':>8s}")

    for i in range(len(sp_V)):
        Vs_kn = sp_V[i]
        Vs = Vs_kn * 0.5144  # m/s
        w = he_WFT[i]
        t = he_THDF[i]
        Va = Vs * (1.0 - w)
        T = sp_T[i] * 1000.0  # N
        N_pred = sp_N[i]       # prop RPM
        n_pred = N_pred / 60.0  # rev/s

        # Our model's power and eta0 at the prediction's operating point
        # First, find the pitch that gives this thrust at this RPM
        # (the prediction implicitly defines an operating pitch)
        Q_model = prop.torque(prop.design_pitch, n_pred, Va)
        P_model = Q_model * 2.0 * math.pi * n_pred
        T_model = prop.thrust(prop.design_pitch, n_pred, Va)
        eta0_model = prop.eta0(prop.design_pitch, n_pred, Va)
        J_model = Va / (n_pred * prop.diameter)

        PD_pred = sp_PD[i]
        eta0_pred = he_ETAO[i]
        J_pred = he_ADVC[i]
        err = 100.0 * (P_model / 1000.0 - PD_pred) / PD_pred

        print(f"  {Vs_kn:5.1f}  {sp_T[i]:5.1f}  {w:5.3f}  {t:5.3f}  "
              f"{N_pred:6.1f}  {PD_pred:8.0f}  {P_model/1000:9.1f}  {err:+6.1f}  "
              f"{eta0_pred:10.3f}  {eta0_model:11.4f}  {J_pred:7.3f}  {J_model:8.4f}")

    # ----------------------------------------------------------------
    # Part 2: Fuel-optimal operating points using actual hull data
    #
    # Use interpolated RT, w, t from the service prediction tables.
    # Compare our optimiser's choice against the prediction's N and PD.
    # ----------------------------------------------------------------
    print(f"\n--- Fuel-optimal operating points (actual hull data) ---")
    print(f"  {'V kn':>5s}  {'T kN':>6s}  {'P/D':>5s}  "
          f"{'N opt':>6s}  {'N pred':>7s}  {'PD opt':>7s}  {'PD pred':>8s}  "
          f"{'eta0 opt':>9s}  {'eta0 pred':>10s}  "
          f"{'Eng rpm':>8s}  {'SFOC':>6s}  {'Fuel kg/h':>10s}")

    for i in range(len(sp_V)):
        Vs_kn = sp_V[i]
        Vs = Vs_kn * 0.5144
        w = he_WFT[i]
        t = he_THDF[i]
        Va = Vs * (1.0 - w)
        T_required = sp_RT[i] * 1000.0 / (1.0 - t)  # T = R / (1 - t)

        result = find_min_fuel_operating_point(
            prop, Va, T_required, engine,
            gear_ratio=gear_ratio,
            shaft_efficiency=0.97,
            auxiliary_power_kw=0.0,
        )

        N_pred = sp_N[i]
        PD_pred = sp_PD[i]
        eta0_pred = he_ETAO[i]

        if result.found:
            print(f"  {Vs_kn:5.1f}  {T_required/1000:6.1f}  {result.pitch:5.2f}  "
                  f"{result.rpm:6.1f}  {N_pred:7.1f}  {result.power_kw:7.1f}  {PD_pred:8.0f}  "
                  f"{result.eta0:9.4f}  {eta0_pred:10.3f}  "
                  f"{result.engine_rpm:8.1f}  {result.sfoc:6.1f}  {result.fuel_rate/1000:10.2f}")
        else:
            print(f"  {Vs_kn:5.1f}  {T_required/1000:6.1f}  -- no feasible point --")


def demo_optimal_combinator():
    """Generate an optimal combinator curve using the service prediction hull data.

    Uses the MAN L27/38 muzzle diagram engine model with 15% sea margin.
    Compares the optimal combinator against the service prediction's operating schedule.
    """
    print("\n" + "=" * 70)
    print("OPTIMAL COMBINATOR CURVE (15% sea margin)")
    print("=" * 70)

    engine = make_man_l27_38()
    gear_ratio = 800.0 / 117.6  # 6.803

    # ----------------------------------------------------------------
    # Service prediction data (15% sea margin already included in PD)
    # ----------------------------------------------------------------
    # Columns: V [kts], FN, RT [kN], T [kN], PE [kW], PD [kW], ETAD, N [RPM]
    service_pred = np.array([
        [ 8.00, .1200,  67,  77.6,  275,  341, .807,  64.2],
        [ 8.50, .1275,  77,  89.8,  337,  416, .809,  68.6],
        [ 9.00, .1350,  88, 102.9,  407,  502, .810,  72.9],
        [ 9.50, .1425,  99, 116.5,  485,  598, .811,  77.2],
        [10.00, .1500, 111, 130.3,  569,  702, .811,  81.5],
        [10.50, .1575, 122, 144.2,  660,  814, .811,  85.6],
        [11.00, .1650, 134, 158.6,  759,  936, .810,  89.8],
        [11.50, .1725, 146, 173.5,  865, 1068, .810,  93.9],
        [12.00, .1800, 159, 188.8,  979, 1211, .809,  98.0],
        [12.50, .1875, 171, 204.5, 1100, 1362, .807, 102.0],
        [13.00, .1950, 184, 220.6, 1229, 1523, .807, 106.0],
        [13.50, .2025, 197, 237.3, 1367, 1696, .806, 110.0],
        [14.00, .2099, 211, 254.7, 1516, 1884, .805, 114.0],
        [14.50, .2174, 225, 273.1, 1677, 2092, .802, 118.1],
        [15.00, .2249, 240, 292.7, 1852, 2325, .797, 122.4],
        [15.50, .2324, 257, 315.2, 2050, 2591, .791, 126.8],
        [16.00, .2399, 276, 341.2, 2276, 2897, .786, 131.5],
    ])

    # Hull efficiency elements
    # Columns: V [kts], FN, ADVC (J), THDF (t), WFT (w), ETAH, ETAO, ETAR, ETAD, CTH
    hull_eff = np.array([
        [ 8.00, .1200, .628, .140, .239, 1.130, .725, 1.000, .820, .753],
        [ 8.50, .1275, .624, .142, .240, 1.129, .723, 1.008, .822, .775],
        [ 9.00, .1350, .620, .145, .242, 1.127, .721, 1.015, .824, .795],
        [ 9.50, .1425, .618, .148, .243, 1.124, .719, 1.021, .825, .810],
        [10.00, .1500, .616, .151, .243, 1.121, .718, 1.025, .825, .818],
        [10.50, .1575, .616, .153, .242, 1.118, .718, 1.028, .825, .820],
        [11.00, .1650, .615, .155, .242, 1.115, .717, 1.030, .824, .822],
        [11.50, .1725, .615, .157, .242, 1.112, .717, 1.033, .824, .823],
        [12.00, .1800, .615, .160, .242, 1.108, .718, 1.035, .823, .822],
        [12.50, .1875, .616, .164, .242, 1.103, .718, 1.037, .821, .820],
        [13.00, .1950, .616, .167, .242, 1.099, .718, 1.040, .821, .819],
        [13.50, .2025, .616, .170, .243, 1.095, .718, 1.043, .820, .817],
        [14.00, .2099, .617, .173, .242, 1.091, .718, 1.045, .819, .815],
        [14.50, .2174, .617, .177, .242, 1.086, .719, 1.045, .815, .813],
        [15.00, .2249, .617, .180, .240, 1.079, .719, 1.045, .810, .811],
        [15.50, .2324, .617, .184, .239, 1.071, .718, 1.046, .805, .815],
        [16.00, .2399, .615, .190, .238, 1.063, .717, 1.048, .799, .827],
    ])

    sp_V   = service_pred[:, 0]
    sp_RT  = service_pred[:, 2]  # RT [kN] — this is the trial resistance
    sp_T   = service_pred[:, 3]  # T [kN] — includes sea margin
    sp_PD  = service_pred[:, 5]  # PD [kW] — includes sea margin
    sp_N   = service_pred[:, 7]  # prop RPM

    he_THDF = hull_eff[:, 3]  # t
    he_WFT  = hull_eff[:, 4]  # w

    # ----------------------------------------------------------------
    # Set up propeller model
    # ----------------------------------------------------------------
    D = 4.66
    BAR = 0.432
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    prop = CSeriesPropeller(bar_data, design_pitch=0.771, diameter=D, area_ratio=BAR, rho=1025.0)

    print(f"\nEngine:     {engine.name}")
    print(f"Gear ratio: {gear_ratio:.3f}")
    print(f"Propeller:  D={D}m, design P/D=0.771, BAR={BAR}")
    print(f"Sea margin: 15% (already included in the prediction T and PD)")

    # ----------------------------------------------------------------
    # The service prediction T and PD already include the 15% sea margin.
    # The RT column is the trial (calm-water) resistance.
    # T = RT / (1-t) is the trial thrust; the sea margin is then
    # applied to PD.  We can back-calculate the sea-margin thrust as:
    #   T_service = T_prediction (which already has it baked in)
    # So we use T directly, not RT/(1-t).
    # ----------------------------------------------------------------

    # Generate optimal combinator using the prediction's thrust directly
    # (which already includes the 15% sea margin).
    print(f"\n--- Optimal combinator (fuel-minimum) ---")
    print(f"  {'V kn':>5s}  {'T kN':>6s}  {'P/D':>5s}  "
          f"{'N opt':>6s}  {'N pred':>7s}  {'PD opt':>7s}  {'PD pred':>8s}  "
          f"{'Eng rpm':>8s}  {'SFOC':>6s}  {'Fuel kg/h':>10s}  {'eta0':>6s}")

    # Build arrays for the combinator generator
    # We pass T directly as "resistance" and set t=0, sea_margin=0
    # since the prediction's T already includes everything.
    combo_points = []
    for i in range(len(sp_V)):
        Vs_kn = sp_V[i]
        Vs = Vs_kn * 0.5144
        w = he_WFT[i]
        Va = Vs * (1.0 - w)
        T_required = sp_T[i] * 1000.0  # N — prediction thrust (includes margin)

        op = find_min_fuel_operating_point(
            prop, Va, T_required, engine,
            gear_ratio=gear_ratio,
            shaft_efficiency=0.97,
            auxiliary_power_kw=0.0,
            pitch_step=0.005,
        )

        if op.found:
            from optimiser import CombinatorPoint
            combo_points.append(CombinatorPoint(
                Vs_kn=Vs_kn, Va=Va, T_required=T_required,
                pitch=op.pitch, n=op.n, rpm=op.rpm,
                engine_rpm=op.engine_rpm, power_kw=op.power_kw,
                engine_power_kw=op.engine_power_kw, eta0=op.eta0,
                fuel_rate=op.fuel_rate, sfoc=op.sfoc,
            ))
            print(f"  {Vs_kn:5.1f}  {sp_T[i]:6.1f}  {op.pitch:5.2f}  "
                  f"{op.rpm:6.1f}  {sp_N[i]:7.1f}  {op.power_kw:7.1f}  {sp_PD[i]:8.0f}  "
                  f"{op.engine_rpm:8.1f}  {op.sfoc:6.1f}  {op.fuel_rate/1000:10.2f}  "
                  f"{op.eta0:6.4f}")
        else:
            print(f"  {Vs_kn:5.1f}  {sp_T[i]:6.1f}  -- no feasible point --")

    # ----------------------------------------------------------------
    # Fit a smooth combinator curve and show it
    # ----------------------------------------------------------------
    if len([p for p in combo_points if p.found]) >= 2:
        combinator = fit_combinator_curve(combo_points)

        print(f"\n--- Fitted combinator curve: P/D = f(shaft RPM) ---")
        feasible = [p for p in combo_points if p.found]
        rpm_min = min(p.rpm for p in feasible)
        rpm_max = max(p.rpm for p in feasible)
        print(f"  Valid range: {rpm_min:.1f} - {rpm_max:.1f} shaft RPM")
        print(f"  {'RPM':>6s}  {'P/D':>6s}")
        for rpm in np.arange(math.floor(rpm_min), math.ceil(rpm_max) + 1, 2.0):
            if rpm < rpm_min or rpm > rpm_max:
                continue
            pd = combinator(rpm)
            print(f"  {rpm:6.1f}  {pd:6.3f}")

    # ----------------------------------------------------------------
    # Compare: what if we use the prediction's RPM schedule but
    # optimise pitch only?
    # ----------------------------------------------------------------
    print(f"\n--- Pitch-only optimisation at prediction's RPM schedule ---")
    print(f"  {'V kn':>5s}  {'N pred':>7s}  {'P/D opt':>7s}  "
          f"{'PD opt':>7s}  {'PD pred':>8s}  {'err%':>6s}  "
          f"{'eta0 opt':>9s}  {'Eng rpm':>8s}  {'SFOC':>6s}  {'Fuel kg/h':>10s}")

    from optimiser import _find_n_for_thrust

    for i in range(len(sp_V)):
        Vs_kn = sp_V[i]
        Vs = Vs_kn * 0.5144
        w = he_WFT[i]
        Va = Vs * (1.0 - w)
        T_required = sp_T[i] * 1000.0  # N
        N_pred = sp_N[i]
        n_pred = N_pred / 60.0

        # Sweep pitch, but fix RPM = prediction's RPM.
        # Find the pitch that delivers T_required at n_pred.
        best_fuel = float("inf")
        best_pitch = None
        best_result = None

        pitches = prop.pitch_table
        p_min = max(0.0, float(pitches[0]))
        p_max = float(pitches[-1])
        for pitch in np.arange(p_min, p_max + 0.001, 0.005):
            T_at_pitch = prop.thrust(pitch, n_pred, Va)
            # We want T_at_pitch >= T_required (close enough)
            if T_at_pitch < T_required * 0.99:
                continue

            Q = prop.torque(pitch, n_pred, Va)
            P = Q * 2.0 * math.pi * n_pred
            P_kw = P / 1000.0
            eng_rpm = N_pred * gear_ratio
            P_eng_kw = P_kw / 0.97
            if P_eng_kw > engine.max_power(eng_rpm) or P_eng_kw <= 0:
                continue
            fuel = engine.fuel_rate(P_eng_kw, eng_rpm)
            if fuel < best_fuel:
                best_fuel = fuel
                best_pitch = pitch
                best_result = (pitch, P_kw, P_eng_kw, eng_rpm, fuel, prop.eta0(pitch, n_pred, Va))

        if best_result is not None:
            pitch, P_kw, P_eng_kw, eng_rpm, fuel, eta0 = best_result
            sfoc = fuel / P_eng_kw if P_eng_kw > 0 else 0
            err = 100.0 * (P_kw - sp_PD[i]) / sp_PD[i]
            print(f"  {Vs_kn:5.1f}  {N_pred:7.1f}  {pitch:7.2f}  "
                  f"{P_kw:7.1f}  {sp_PD[i]:8.0f}  {err:+6.1f}  "
                  f"{eta0:9.4f}  {eng_rpm:8.1f}  {sfoc:6.1f}  {fuel/1000:10.2f}")
        else:
            print(f"  {Vs_kn:5.1f}  {N_pred:7.1f}  -- no feasible pitch --")


def demo_single_speed_margin_sweep():
    """Multi-speed analysis across a wide thrust margin range.

    For each speed (8, 10, 12 kn), compares a factory combinator against
    an actively optimised combinator across thrust margins from -50% to
    +100%.

    The factory combinator is a piecewise-linear schedule mapping lever
    position to (pitch, RPM), with ~6 breakpoints.  It was designed for
    the full 8-15.5 kn speed range at 15% sea margin.  In operation the
    speed controller adjusts the lever until thrust equilibrium is reached
    at the actual advance velocity -- the combinator does NOT know the
    speed through water.

    When the shaft generator is engaged (PTO > 0), a separate SG
    combinator schedule is used, designed for 250 kW PTO within the
    SG RPM band (engine 640-800 rpm).

    The vessel has Flettner rotors providing variable wind-assist thrust,
    meaning the propeller thrust demand can be significantly reduced
    (negative margins) or increased (positive margins, e.g. head wind)
    compared to the factory design condition.
    """
    print("\n" + "=" * 80)
    print("MULTI-SPEED MARGIN SWEEP: 8, 10, 12 kn, -50% to +100% thrust margin")
    print("=" * 80)

    from optimiser import _find_n_for_thrust

    engine = make_man_l27_38()
    gear_ratio = 800.0 / 117.6  # 6.803

    # ----------------------------------------------------------------
    # Set up propeller model (same as demo_optimal_combinator)
    # ----------------------------------------------------------------
    D = 4.66
    BAR = 0.432
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    prop = CSeriesPropeller(bar_data, design_pitch=0.771, diameter=D, area_ratio=BAR, rho=1025.0)

    # ----------------------------------------------------------------
    # Hull data from service prediction (with 15% sea margin)
    # ----------------------------------------------------------------
    # Speed [kn], wake fraction w, thrust T [kN]
    hull_speeds = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
                            12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5])
    hull_w      = np.array([.239, .240, .242, .243, .243, .242, .242, .242,
                            .242, .242, .242, .243, .242, .242, .240, .239])
    hull_T_kN   = np.array([67.1, 89.8, 102.9, 116.5, 130.3, 144.2, 158.6,
                            173.5, 188.8, 204.5, 220.6, 237.3, 254.7, 273.1,
                            292.7, 315.2])

    # ----------------------------------------------------------------
    # Factory combinator: piecewise-linear schedule
    # ----------------------------------------------------------------
    min_prop_rpm = engine.min_rpm() / gear_ratio  # 70.56 shaft RPM
    max_prop_rpm = engine.max_rpm() / gear_ratio  # 117.6 shaft RPM

    # Shaft generator RPM constraint
    sg_engine_rpm_min = 48.0 / 60.0 * engine.max_rpm()  # 640 rpm
    sg_engine_rpm_max = engine.max_rpm()                  # 800 rpm
    sg_prop_rpm_min = sg_engine_rpm_min / gear_ratio      # ~94.1 rpm
    sg_prop_rpm_max = sg_engine_rpm_max / gear_ratio      # ~117.6 rpm

    combo_lever = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    combo_rpm   = np.array([min_prop_rpm, min_prop_rpm, min_prop_rpm,
                            min_prop_rpm, 95.0, max_prop_rpm])
    combo_pitch = np.array([0.02, 0.50, 0.75, 1.00, 1.10, 1.15])

    def factory_rpm(lv):
        return float(np.interp(lv, combo_lever, combo_rpm))

    def factory_pitch(lv):
        return float(np.interp(lv, combo_lever, combo_pitch))

    # SG factory combinator (designed for 250 kW PTO)
    sg_combo_lever = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    sg_combo_rpm   = np.array([sg_prop_rpm_min, sg_prop_rpm_min,
                               sg_prop_rpm_min, sg_prop_rpm_min,
                               sg_prop_rpm_min, 97.0])
    sg_combo_pitch = np.array([0.02, 0.20, 0.45, 0.65, 0.80, 0.86])

    def sg_factory_rpm(lv):
        return float(np.interp(lv, sg_combo_lever, sg_combo_rpm))

    def sg_factory_pitch(lv):
        return float(np.interp(lv, sg_combo_lever, sg_combo_pitch))

    # ----------------------------------------------------------------
    # Common parameters
    # ----------------------------------------------------------------
    shaft_eff = 0.97
    n_min_prop = engine.min_rpm() / gear_ratio / 60.0
    n_max_prop = engine.max_rpm() / gear_ratio / 60.0

    print(f"\nEngine:     {engine.name}")
    print(f"Gear ratio: {gear_ratio:.3f}")
    print(f"Propeller:  D={D}m, design P/D=0.771, BAR={BAR}")

    print(f"\nFactory combinator breakpoints:")
    print(f"  {'Lever%':>6s}  {'RPM':>6s}  {'P/D':>5s}")
    for i in range(len(combo_lever)):
        print(f"  {combo_lever[i]:6.0f}  {combo_rpm[i]:6.1f}  {combo_pitch[i]:5.2f}")

    print(f"\nSG factory combinator breakpoints (designed for 250 kW PTO):")
    print(f"  {'Lever%':>6s}  {'RPM':>6s}  {'P/D':>5s}")
    for i in range(len(sg_combo_lever)):
        print(f"  {sg_combo_lever[i]:6.0f}  {sg_combo_rpm[i]:6.1f}  {sg_combo_pitch[i]:5.2f}")

    print(f"\nShaft generator RPM constraint (when PTO > 0):")
    print(f"  Engine: {sg_engine_rpm_min:.0f} - {sg_engine_rpm_max:.0f} rpm  "
          f"(48-60 Hz)")
    print(f"  Shaft:  {sg_prop_rpm_min:.1f} - {sg_prop_rpm_max:.1f} rpm")

    # ----------------------------------------------------------------
    # Sweep parameters
    # ----------------------------------------------------------------
    margins = np.arange(-50, 105, 5)  # -50, -45, ..., +95, +100
    pto_levels = [0, 200, 400, 600, 800]  # kW shaft generator load
    analysis_speeds = [8.0, 10.0, 12.0]  # kn
    key_margins = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 50, 75, 100]

    # Grand summary: summary_cache[(Vs_kn, margin_pct, pto_kw)] = (fr, op)
    summary_cache = {}

    for Vs_kn in analysis_speeds:
        # Interpolate hull data for this speed
        w = float(np.interp(Vs_kn, hull_speeds, hull_w))
        T_nominal_kN = float(np.interp(Vs_kn, hull_speeds, hull_T_kN))
        Vs = Vs_kn * 0.5144  # m/s
        Va = Vs * (1.0 - w)

        print(f"\n\n{'#'*105}")
        print(f"# SPEED: {Vs_kn:.0f} kn,  Va = {Va:.3f} m/s,  w = {w:.3f},  "
              f"T_nominal = {T_nominal_kN:.1f} kN (incl. 15% sea margin)")
        print(f"{'#'*105}")

        # ----------------------------------------------------------
        # Helper functions (close over Va for this speed)
        # ----------------------------------------------------------
        def find_factory_lever(T_req_N, pto_kw=0.0, tol_N=100, _Va=Va):
            if pto_kw > 0:
                get_pitch = sg_factory_pitch
                get_rpm = sg_factory_rpm
            else:
                get_pitch = factory_pitch
                get_rpm = factory_rpm

            lo, hi = 0.0, 100.0

            p_hi, r_hi = get_pitch(hi), get_rpm(hi)
            n_hi = r_hi / 60.0
            T_max = prop.thrust(p_hi, n_hi, _Va) if n_hi > 0.01 and p_hi > 0.01 else 0
            if T_req_N > T_max + tol_N:
                return None

            p_lo, r_lo = get_pitch(lo), get_rpm(lo)
            n_lo = r_lo / 60.0
            T_min = prop.thrust(p_lo, n_lo, _Va) if n_lo > 0.01 and p_lo > 0.01 else 0
            if T_req_N < T_min - tol_N:
                return None

            for _ in range(80):
                mid = (lo + hi) / 2.0
                p = get_pitch(mid)
                r = get_rpm(mid)
                n = r / 60.0
                if n < 0.01 or p < 0.01:
                    lo = mid
                    continue
                T = prop.thrust(p, n, _Va)
                if abs(T - T_req_N) < tol_N:
                    return mid
                if T < T_req_N:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2.0

        def eval_factory(T_kN, pto_kw=0.0, _Va=Va):
            T_req_N = T_kN * 1000.0
            lv = find_factory_lever(T_req_N, pto_kw=pto_kw, _Va=_Va)
            if lv is None:
                return None

            if pto_kw > 0:
                pitch = sg_factory_pitch(lv)
                rpm = sg_factory_rpm(lv)
            else:
                pitch = factory_pitch(lv)
                rpm = factory_rpm(lv)
            n = rpm / 60.0
            eng_rpm = rpm * gear_ratio

            T_check = prop.thrust(pitch, n, _Va)
            if abs(T_check - T_req_N) > 500:
                return None

            Q = prop.torque(pitch, n, _Va)
            P_shaft = Q * 2.0 * math.pi * n
            P_shaft_kw = P_shaft / 1000.0
            P_eng_kw = P_shaft_kw / shaft_eff + pto_kw
            eta0 = prop.eta0(pitch, n, _Va)

            if (eng_rpm < engine.min_rpm() or eng_rpm > engine.max_rpm()
                    or P_eng_kw <= 0 or P_eng_kw > engine.max_power(eng_rpm)):
                return None

            fuel = engine.fuel_rate(P_eng_kw, eng_rpm)
            return (fuel, pitch, rpm, P_shaft_kw, eng_rpm, eta0)

        def find_optimised(T_kN, pto_kw=0.0, _Va=Va):
            T_req_N = T_kN * 1000.0

            if pto_kw > 0:
                eff_n_min = sg_prop_rpm_min / 60.0
                eff_n_max = sg_prop_rpm_max / 60.0
                eff_eng_rpm_min = sg_engine_rpm_min
                eff_eng_rpm_max = sg_engine_rpm_max
            else:
                eff_n_min = n_min_prop
                eff_n_max = n_max_prop
                eff_eng_rpm_min = engine.min_rpm()
                eff_eng_rpm_max = engine.max_rpm()

            def _eval_pitch(pitch):
                n = _find_n_for_thrust(prop, pitch, _Va, T_req_N,
                                       n_min=max(eff_n_min, 0.1),
                                       n_max=eff_n_max)
                if n is None:
                    return None
                rpm = n * 60.0
                eng_rpm = rpm * gear_ratio
                if eng_rpm > eff_eng_rpm_max or eng_rpm < eff_eng_rpm_min:
                    return None

                Q = prop.torque(pitch, n, _Va)
                P = Q * 2.0 * math.pi * n
                P_kw = P / 1000.0
                P_eng = P_kw / shaft_eff + pto_kw
                if P_eng > engine.max_power(eng_rpm) or P_eng <= 0:
                    return None

                fuel = engine.fuel_rate(P_eng, eng_rpm)
                eta0 = prop.eta0(pitch, n, _Va)
                return (fuel, pitch, rpm, P_kw, eng_rpm, eta0)

            # Coarse pass (step=0.05)
            coarse_grid = np.arange(0.00, 1.50, 0.05)
            best = None
            best_coarse_pitch = None
            for pitch in coarse_grid:
                result = _eval_pitch(pitch)
                if result is not None and (best is None or result[0] < best[0]):
                    best = result
                    best_coarse_pitch = pitch

            # Fine pass: refine around coarse best (±0.05, step=0.005)
            if best_coarse_pitch is not None:
                fine_lo = max(0.00, best_coarse_pitch - 0.05)
                fine_hi = min(1.50, best_coarse_pitch + 0.05)
                fine_grid = np.arange(fine_lo, fine_hi + 0.001, 0.005)
                for pitch in fine_grid:
                    result = _eval_pitch(pitch)
                    if result is not None and (best is None or result[0] < best[0]):
                        best = result

            return best

        # ----------------------------------------------------------
        # Sweep margins for each PTO level
        # ----------------------------------------------------------
        for pto_kw in pto_levels:
            print(f"\n{'='*105}")
            sg_note = f" [SG: {sg_engine_rpm_min:.0f}-{sg_engine_rpm_max:.0f} eng rpm]" if pto_kw > 0 else ""
            print(f"{Vs_kn:.0f} kn — PTO shaft generator load: {pto_kw} kW{sg_note}")
            print(f"{'='*105}")

            print(f"\n{'Margin':>7s}  {'T kN':>6s}  "
                  f"{'f P/D':>5s} {'f RPM':>6s} {'f PD':>6s} {'f Fuel':>7s} {'f eta0':>6s}  "
                  f"{'o P/D':>5s} {'o RPM':>6s} {'o PD':>6s} {'o Fuel':>7s} {'o eta0':>6s}  "
                  f"{'Save%':>6s}")
            print("-" * 105)

            for margin_pct in margins:
                T_kN = T_nominal_kN * (1.0 + margin_pct / 100.0)

                fr = eval_factory(T_kN, pto_kw=pto_kw)
                op = find_optimised(T_kN, pto_kw=pto_kw)

                # Cache for summary
                summary_cache[(Vs_kn, int(margin_pct), pto_kw)] = (fr, op)

                if fr is not None:
                    f_fuel, f_pitch, f_rpm, f_pd, f_eng_rpm, f_eta0 = fr
                    f_str = (f"{f_pitch:5.2f} {f_rpm:6.1f} {f_pd:6.0f} "
                             f"{f_fuel/1000:7.1f} {f_eta0:6.4f}")
                else:
                    f_fuel = None
                    f_str = f"{'--':>5s} {'--':>6s} {'--':>6s} {'--':>7s} {'--':>6s}"

                if op is not None:
                    o_fuel, o_pitch, o_rpm, o_pd, o_eng_rpm, o_eta0 = op
                    o_str = (f"{o_pitch:5.2f} {o_rpm:6.1f} {o_pd:6.0f} "
                             f"{o_fuel/1000:7.1f} {o_eta0:6.4f}")
                else:
                    o_fuel = None
                    o_str = f"{'--':>5s} {'--':>6s} {'--':>6s} {'--':>7s} {'--':>6s}"

                if f_fuel is not None and o_fuel is not None and f_fuel > 0:
                    saving = 100.0 * (f_fuel - o_fuel) / f_fuel
                    save_str = f"{saving:+6.1f}%"
                else:
                    save_str = f"{'--':>6s}"

                print(f"{margin_pct:+6.0f}%  {T_kN:6.1f}  {f_str}  {o_str}  {save_str}")

    # ----------------------------------------------------------------
    # Grand summary: saving at key margins for each speed and PTO level
    # ----------------------------------------------------------------
    for Vs_kn in analysis_speeds:
        T_nominal_kN = float(np.interp(Vs_kn, hull_speeds, hull_T_kN))

        print(f"\n{'='*105}")
        print(f"SUMMARY {Vs_kn:.0f} kn: fuel saving (%) at key margins for each PTO level")
        print(f"  (PTO > 0: SG combinator + RPM constraint "
              f"{sg_engine_rpm_min:.0f}-{sg_engine_rpm_max:.0f} engine rpm)")
        print(f"{'='*105}")

        header = f"{'Margin':>7s}  {'T kN':>6s}"
        for pto_kw in pto_levels:
            header += f"  {pto_kw:>5d}kW"
        print(header)
        print("-" * (16 + 8 * len(pto_levels)))

        for margin_pct in key_margins:
            T_kN = T_nominal_kN * (1.0 + margin_pct / 100.0)
            row = f"{margin_pct:+6.0f}%  {T_kN:6.1f}"
            for pto_kw in pto_levels:
                fr, op = summary_cache.get((Vs_kn, margin_pct, pto_kw), (None, None))
                if fr is not None and op is not None and fr[0] > 0:
                    saving = 100.0 * (fr[0] - op[0]) / fr[0]
                    row += f"  {saving:+6.1f}%"
                else:
                    row += f"  {'--':>7s}"
            print(row)


if __name__ == "__main__":
    validate_propeller_model()
    demo_power_optimiser()
    demo_fuel_optimiser()
    demo_bar_interpolation()
    demo_muzzle_diagram()
    demo_optimal_combinator()
    demo_single_speed_margin_sweep()
