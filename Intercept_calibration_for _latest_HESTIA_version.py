# -*- coding: utf-8 -*-
"""
intercept_estimation_rev14.py
==============================

HESTIA rev14 — Standalone intercept calibration script.

PURPOSE
-------
Na de introductie van rev14 wijzigingen (individuele K_p-distributie, NSAID,
post-finish module) moet de intercept van het collapse risk model opnieuw
worden gekalibreerd. Dit script voert een Newton-iteratie uit op een vaste
populatie (N=50.000) onder referentie-meteorologische condities en levert
de nieuwe intercepts voor de drie eindpunten (EHS, ziekenhuisopname, EHBO).

USAGE
-----
    python intercept_estimation_rev14.py

Configuratie kan aangepast worden in het CONFIGURATION-blok hieronder.

RUNTIME
-------
N=50.000, parallel: ~10-20 min (afhankelijk van hardware)
N=10.000 (snelle test): ~3-6 min

OUTPUT
------
- Console: iteratielog en eindrapport met plakklare COLLAPSE_ENDPOINTS
- intercept_estimation_rev14_results.json
- Kalibratiepopulatie .pkl (herbruikbaar)

AUTHOR
------
HESTIA project / Koos de Boer, Apr-2026
"""

import sys
import os
import json
import time
import pickle
import glob
import importlib
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from colorama import init, Fore, Style

init(autoreset=True)

# =============================================================================
#  CONFIGURATION (pas hier aan voor uw situatie)
# =============================================================================

# --- Populatie parameters ---
N_POPULATION   = 10000      # gebruik 50000 voor productie, 10000 voor test
RANDOM_SEED    = 42          # vaste seed voor reproduceerbaarheid

# --- Referentie meteorologie (must activeer voldoende T_rect > 40.5°C) ---
# Rev14 heeft individuele K_p (sommigen stoppen niet) en post-finish afterglow.
# Daarom starten we iets warmer dan voorheen: 20→26°C, matige luchtvochtigheid.
REF_CONDITIONS = {
    'tdb_start'    : 20.0,    # °C drogebol bij start
    'tdb_end'      : 26.0,    # °C drogebol bij finish (oplopend)
    'rh'           : 55.0,    # % relatieve luchtvochtigheid
    'wind_ms'      : 2.5,     # m/s op 1.5m hoogte
    'cloud_cover'  : 20,      # % bewolking (weinig, voor zoninstraling)
    'pressure'     : 1013.0,  # hPa
    'duration_h'   : 4.0,     # simulatie-uren (dekt langzamere lopers)
    'interval_min' : 10,      # weersinterval (minuten)
    'lat'          : 42.36,   # Boston, MA (referentielocatie)
    'lon'          : -71.06,
    'start_time'   : '2017-04-17 10:00:00',
    'timezone'     : 'America/New_York',
    'met_value'    : 11.0,    # referentie MET voor event-pace (alleen voor liveability check)
    'clo_value'    : 0.2,     # korte broek + shirt
    'training_factor'        : 0.30,   # gemiddelde trainingstoestand
    'acclimatization_factor' : 0.35,   # matige acclimatisatie
}

# --- Kalibratiedoelen (incidentie per 10.000 deelnemers) ---
ENDPOINTS = {
    'ehs': {
        'label'   : 'EHS klinisch (Boston 2012-2019 gepoold)',
        'p_target': 9.0 / 10_000,          # 0.00090 Breslow 2021
        'source'  : 'Breslow RG et al. (2021) Am J Sports Med 49(10):2696-2703',
        'status'  : 'VALIDATED',           # na kalibratie wordt dit de status
    },
    'hospitalisation': {
        'label'   : 'Ziekenhuisopname (DtD 2024)',
        'p_target': 50.0 / 35_000,         # 0.001429 GHOR NHN
        'source'  : 'GHOR Noord-Holland Noord, Dam tot Damloop 2024',
        'status'  : 'PROVISIONAL',
    },
    'ehbo': {
        'label'   : 'EHBO-contact alle incidenten (DtD 2024 schatting)',
        'p_target': 150.0 / 35_000,        # 0.004286 centrale schatting
        'source'  : 'Schatting GHOR NHN / DtD 2024 (ongepubliceerd)',
        'status'  : 'PROVISIONAL',
    },
}

# --- Newton parameters ---
INTERCEPT_INIT  = -8.0
MAX_ITER        = 40
TOL_ABS         = 1e-9      # absolute fout in p_mean
TOL_REL         = 1e-7      # relatieve verandering in intercept

# --- Uitvoerbestanden ---
RESULTS_JSON = 'intercept_estimation_rev14_results.json'
POP_PKL      = f'intercept_estimation_population_N{N_POPULATION}_seed{RANDOM_SEED}_rev14.pkl'

# =============================================================================
#  DYNAMISCHE ENGINE IMPORTER (kies nieuwste versie)
# =============================================================================

def _import_engine():
    """
    Importeert de nieuwste HESTIA_Data_Engine_CVR_v*.py uit dezelfde map.
    Geeft foutmelding als geen engine gevonden.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(script_dir, "HESTIA_Data_Engine_CVR_v*.py")
    files = glob.glob(pattern)
    if not files:
        raise ImportError(f"Geen HESTIA engine gevonden in {script_dir}")

    # Kies bestand met hoogste versienummer (bijv. v9 > v8)
    def version_from_filename(f):
        basename = os.path.basename(f)
        # zoek '_v' gevolgd door cijfers
        import re
        m = re.search(r'_v(\d+)\.py$', basename)
        return int(m.group(1)) if m else 0
    latest_file = max(files, key=version_from_filename)
    module_name = os.path.splitext(os.path.basename(latest_file))[0]

    # Voeg script_dir toe aan sys.path voor de zekerheid
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    engine = importlib.import_module(module_name)
    print(f"{Fore.GREEN}Engine geladen: {module_name} (rev{version_from_filename(latest_file)}){Style.RESET_ALL}")
    return engine

# =============================================================================
#  REFERENTIE WEER OPBOUWEN (synthetisch)
# =============================================================================

def build_reference_interp(cfg: dict) -> list:
    """Bouw een synthetische weersreeks (zelfde formaat als interpolate_weather)."""
    import pytz
    from pythermalcomfort.utilities import wet_bulb_tmp

    tz = pytz.timezone(cfg['timezone'])
    start_dt = pd.Timestamp(cfg['start_time']).tz_localize(tz)
    end_dt = start_dt + pd.Timedelta(hours=cfg['duration_h'])
    times = pd.date_range(start=start_dt, end=end_dt, freq=f"{cfg['interval_min']}min")
    n_steps = len(times)
    temps = np.linspace(cfg['tdb_start'], cfg['tdb_end'], n_steps)

    interp_data = []
    for i, t in enumerate(times):
        tdb = float(temps[i])
        rh = float(cfg['rh'])
        interp_data.append({
            'time': t,
            'temp': tdb,
            'rh': rh,
            'wind': float(cfg['wind_ms']),
            'clouds': float(cfg['cloud_cover']),
            'pressure': float(cfg['pressure']),
            'twb': float(wet_bulb_tmp(tdb=tdb, rh=rh)),
        })
    print(f"  Referentie-interp: {n_steps} stappen, {cfg['tdb_start']}→{cfg['tdb_end']}°C, RH={cfg['rh']}%, wind={cfg['wind_ms']} m/s")
    return interp_data

# =============================================================================
#  POPULATIE LADEN/GENEREREN
# =============================================================================

def get_population(engine):
    """Laad bestaande kalibratiepopulatie of genereer nieuwe."""
    if os.path.isfile(POP_PKL):
        ans = input(f"\nKalibratiepopulatie '{POP_PKL}' gevonden. Laden? (y/n, aanbevolen=y): ").strip().lower()
        if ans in ('y', 'yes', ''):
            try:
                population = engine.load_population(POP_PKL)
                if len(population) >= N_POPULATION:
                    print(f"{Fore.GREEN}Populatie geladen: {len(population)} profielen.{Style.RESET_ALL}")
                    return population[:N_POPULATION]
                else:
                    print(f"{Fore.YELLOW}Populatie heeft {len(population)} profielen; {N_POPULATION} nodig. Opnieuw genereren.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Laden mislukt ({e}). Opnieuw genereren.{Style.RESET_ALL}")

    print(f"\nGenereer kalibratiepopulatie N={N_POPULATION}, seed={RANDOM_SEED}...")
    t0 = time.time()
    population = engine.generate_base_population(
        n_simulations          = N_POPULATION,
        training_factor        = REF_CONDITIONS['training_factor'],
        acclimatization_factor = REF_CONDITIONS['acclimatization_factor'],
        random_seed            = RANDOM_SEED,
    )
    print(f"  Generatie: {time.time()-t0:.1f}s, {len(population)} geldige profielen.")
    engine.save_population(population, POP_PKL)
    return population

# =============================================================================
#  DETECTIE OF PARALLEL WERKT (Windows/Spyder fix)
# =============================================================================

def _detect_parallel_safe() -> bool:
    if os.name != 'nt':
        return True
    in_ipython = (
        hasattr(sys, 'ps1') or
        any(k in sys.modules for k in ('IPython', 'ipykernel', 'spyder', 'spyder_kernels'))
    )
    if in_ipython:
        print("  [INFO] Spyder/IPython op Windows -> use_parallel=False (voorkomt crash).")
        print("  Tip: voor snelheid start via terminal: python intercept_estimation_rev14.py")
        return False
    return True

# =============================================================================
#  SIMULATIE RUNNER
# =============================================================================

def run_calibration_simulation(engine, population: list, interp_data: list) -> dict:
    parallel = _detect_parallel_safe()
    n = len(population)
    print(f"\nSimuleer {n} deelnemers op referentiecondities (parallel={parallel})...")
    if not parallel:
        mins_est = n * 0.72 / 60
        print(f"  Geschatte looptijd: {mins_est:.0f}–{mins_est*1.5:.0f} min.")

    t0 = time.time()
    all_results, stats, results_df = engine.run_monte_carlo_adult(
        interp_data            = interp_data,
        lat                    = REF_CONDITIONS['lat'],
        lon                    = REF_CONDITIONS['lon'],
        met_value              = REF_CONDITIONS['met_value'],
        clo_value              = REF_CONDITIONS['clo_value'],
        n_simulations          = n,
        age_configuration      = 'standard',
        training_factor        = REF_CONDITIONS['training_factor'],
        acclimatization_factor = REF_CONDITIONS['acclimatization_factor'],
        use_parallel           = parallel,
        base_population        = population,
        random_seed            = None,
    )
    elapsed = time.time() - t0
    print(f"  Simulatie gereed: {elapsed/60:.1f} min ({elapsed:.0f}s)")

    if all_results is None:
        raise RuntimeError("Simulatie mislukt: all_results is None")

    # Extraheer uitkomstvectoren (alleen tijdens race, niet post-finish)
    # T_rect max tijdens race
    t_rect_max = np.array([
        max((r['t_rect'] for r in sim if not np.isnan(r['t_rect'])), default=np.nan)
        for sim in all_results
    ])

    # CO_reserve min tijdens race (gebruik nanmin, fallback 2.0)
    co_res_min = np.array([
        np.nanmin([r.get('co_reserve', np.nan) for r in sim])
        for sim in all_results
    ])
    co_res_min = np.where(np.isfinite(co_res_min), co_res_min, 2.0)

    # Dehydratie aan het einde van de race (laatste tijdstap)
    dehy_end = np.array([
        sim[-1]['water'] / 1000.0 / max(1.0, population[i].weight) * 100.0
        for i, sim in enumerate(all_results)
    ])
    dehy_end = np.where(np.isfinite(dehy_end), dehy_end, 0.0)

    # Diagnostiek
    print(f"\n  T_rect distributie (race-max):")
    for pct in (5, 25, 50, 75, 90, 95, 99):
        print(f"    P{pct:2d}: {np.percentile(t_rect_max, pct):.3f}°C")
    print(f"    % > 39.5°C : {np.mean(t_rect_max > 39.5)*100:.2f}%")
    print(f"    % > 40.5°C : {np.mean(t_rect_max > 40.5)*100:.2f}%")
    print(f"  CO_reserve min P50: {np.percentile(co_res_min, 50):.3f}")
    print(f"  Dehydratie eind P50: {np.percentile(dehy_end, 50):.2f}%")

    return {
        't_rect_max': t_rect_max,
        'co_res_min': co_res_min,
        'dehy_end'  : dehy_end,
        'stats'     : stats,
        'elapsed_s' : elapsed,
    }

# =============================================================================
#  COLLAPSE PROBABILITY FUNCTIES (gewichten uit engine)
# =============================================================================

def _get_weights_from_engine(engine):
    """Haal de gewichten W_T1 etc. uit de engine module."""
    return {
        'W_T1': getattr(engine, 'W_T1', 1.0),
        'W_T2': getattr(engine, 'W_T2', 4.0),
        'W_C':  getattr(engine, 'W_C',  0.8),
        'W_D':  getattr(engine, 'W_D',  0.5),
    }

def compute_z_vector(sim_out: dict, intercept: float, weights: dict) -> np.ndarray:
    t   = sim_out['t_rect_max']
    co  = sim_out['co_res_min']
    deh = sim_out['dehy_end']
    # Nan-bescherming (zou al gefixed moeten zijn)
    t   = np.where(np.isfinite(t),   t,   38.9)
    co  = np.where(np.isfinite(co),  co,  2.0)
    deh = np.where(np.isfinite(deh), deh, 0.0)

    z = (intercept
         + weights['W_T1'] * np.clip(t - 39.5, 0, None)
         + weights['W_T2'] * np.clip(t - 40.5, 0, None)
         + weights['W_C']  * np.clip(2.0 - co, 0, None)
         + weights['W_D']  * np.clip(deh - 3.0, 0, None))
    return z

def mean_p_and_deriv(sim_out: dict, intercept: float, weights: dict):
    z = compute_z_vector(sim_out, intercept, weights)
    z = np.clip(z, -500, 500)
    p_vec = 1.0 / (1.0 + np.exp(-z))
    p_mean = float(p_vec.mean())
    dp_di = float((p_vec * (1.0 - p_vec)).mean())
    return p_mean, dp_di

def bisection_intercept(sim_out: dict, p_target: float, weights: dict,
                        lo: float = -30.0, hi: float = 5.0,
                        tol: float = 1e-7, max_iter: int = 80) -> dict:
    print(f"\n  [Bisectie] Zoeken in [{lo}, {hi}], target={p_target:.8f}")
    p_lo, _ = mean_p_and_deriv(sim_out, lo, weights)
    p_hi, _ = mean_p_and_deriv(sim_out, hi, weights)
    if p_lo > p_target:
        print(f"  [Bisectie] FOUT: p(lo)={p_lo:.6f} > target. Verklein lo.")
        return {'converged': False, 'intercept': np.nan}
    if p_hi < p_target:
        print(f"  [Bisectie] FOUT: p(hi)={p_hi:.6f} < target. Vergroot hi.")
        return {'converged': False, 'intercept': np.nan}

    history = []
    for i in range(1, max_iter+1):
        mid = (lo + hi) / 2.0
        p_mid, _ = mean_p_and_deriv(sim_out, mid, weights)
        error = p_mid - p_target
        history.append({'iter': i, 'intercept': mid, 'p_mean': p_mid, 'error': error})
        if i <= 5 or i % 10 == 0:
            print(f"  [Bisectie] iter {i:3d}: intercept={mid:.6f}, p_mean={p_mid:.8f}, error={error:+.3e}")
        if abs(hi - lo) < tol:
            return {'intercept': mid, 'p_final': p_mid, 'p_target': p_target,
                    'error_final': error, 'n_iter': i, 'converged': True,
                    'history': history, 'method': 'bisection'}
        if p_mid < p_target:
            lo = mid
        else:
            hi = mid
    mid = (lo+hi)/2
    p_mid, _ = mean_p_and_deriv(sim_out, mid, weights)
    return {'intercept': mid, 'p_final': p_mid, 'p_target': p_target,
            'error_final': p_mid - p_target, 'n_iter': max_iter,
            'converged': False, 'history': history, 'method': 'bisection'}

def calibrate_intercept(sim_out: dict, p_target: float, weights: dict,
                        label: str = '') -> dict:
    intercept = INTERCEPT_INIT
    history = []
    print(f"\n{'='*60}")
    print(f"Newton-iteratie: {label}")
    print(f"  Target P_obs = {p_target:.8f} ({p_target*10000:.4f}/10.000)")
    print(f"  Start intercept = {INTERCEPT_INIT}")
    print(f"{'='*60}")
    print(f"  {'Iter':>4}  {'Intercept':>12}  {'p_mean':>12}  {'Error':>12}  {'|Δintercept|':>14}")
    print(f"  {'-'*60}")

    converged = False
    delta_intercept = np.inf
    for i in range(1, MAX_ITER+1):
        p_mean, dp_di = mean_p_and_deriv(sim_out, intercept, weights)
        error = p_mean - p_target
        history.append({'iter': i, 'intercept': intercept, 'p_mean': p_mean,
                        'error': error, 'delta_intercept': delta_intercept})
        err_col = Fore.GREEN if abs(error) < TOL_ABS*10 else Fore.CYAN if abs(error) < p_target*0.01 else Fore.YELLOW
        print(f"  {i:4d}  {intercept:12.6f}  {p_mean:12.8f}  {err_col}{error:+.3e}{Style.RESET_ALL}  {delta_intercept:14.3e}")

        if abs(error) < TOL_ABS and abs(delta_intercept) < TOL_REL:
            converged = True
            break
        if abs(dp_di) < 1e-15 or not np.isfinite(p_mean) or not np.isfinite(dp_di):
            print(f"{Fore.YELLOW}  Afgeleide nul of NaN → schakel over naar bisectie.{Style.RESET_ALL}")
            break
        delta_intercept = error / dp_di
        intercept -= delta_intercept
        if intercept < -20 or intercept > 5:
            print(f"{Fore.YELLOW}  Intercept buiten bereik ({intercept:.3f}) → bisectie.{Style.RESET_ALL}")
            break
    else:
        print(f"{Fore.YELLOW}  Max iteraties bereikt, schakel naar bisectie.{Style.RESET_ALL}")

    if not converged:
        bis = bisection_intercept(sim_out, p_target, weights)
        if bis['converged']:
            return {**bis, 'label': label}
        else:
            print(f"{Fore.RED}  Bisectie ook niet geconvergeerd.{Style.RESET_ALL}")
            return {'intercept': np.nan, 'p_final': np.nan, 'p_target': p_target,
                    'error_final': np.nan, 'n_iter': i, 'converged': False,
                    'history': history, 'label': label}

    p_final, _ = mean_p_and_deriv(sim_out, intercept, weights)
    error_final = p_final - p_target
    status = f"{Fore.GREEN}GECONVERGEERD{Style.RESET_ALL}" if converged else f"{Fore.RED}NIET GECONVERGEERD{Style.RESET_ALL}"
    print(f"\n  Status: {status}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  p_final: {p_final:.8f} (target {p_target:.8f})")
    return {'intercept': intercept, 'p_final': p_final, 'p_target': p_target,
            'error_final': error_final, 'n_iter': i, 'converged': converged,
            'history': history, 'label': label}

# =============================================================================
#  SENSITIVITEIT VOOR EP3 (EHBO) 
# =============================================================================

def sensitivity_analysis_ehbo(sim_out: dict, weights: dict) -> dict:
    print(f"\n{'='*60}")
    print("Gevoeligheidsanalyse EP3 (EHBO-contact, DtD 2024)")
    print(f"{'='*60}")
    n_range = [100, 125, 150, 175, 200, 225, 250]
    n_total = 35_000
    results = {}
    print(f"  {'N_EHBO':>8}  {'P_obs/10k':>12}  {'Intercept':>12}  {'p_final':>12}")
    print(f"  {'-'*50}")
    for n_ehbo in n_range:
        p_t = n_ehbo / n_total
        res = calibrate_intercept(sim_out, p_t, weights, label=f"EP3_N={n_ehbo}")
        results[n_ehbo] = res['intercept']
        status = "OK" if res['converged'] else "WARN"
        print(f"  {n_ehbo:8d}  {p_t*10000:12.4f}  {res['intercept']:12.6f}  {res['p_final']:12.8f}  {status}")
    return results

# =============================================================================
#  BOOTSTRAP CI VOOR EP1
# =============================================================================

def bootstrap_ci(sim_out: dict, p_target: float, intercept_cal: float,
                 weights: dict, n_boot: int = 500) -> tuple:
    N = len(sim_out['t_rect_max'])
    boots = []
    print(f"\n  Bootstrap CI ({n_boot} resamples, N={N})...")
    t0 = time.time()
    for b in range(n_boot):
        idx = np.random.randint(0, N, size=N)
        boot = {
            't_rect_max': sim_out['t_rect_max'][idx],
            'co_res_min': sim_out['co_res_min'][idx],
            'dehy_end'  : sim_out['dehy_end'][idx],
        }
        intercept = intercept_cal
        for _ in range(15):
            p_mean, dp_di = mean_p_and_deriv(boot, intercept, weights)
            error = p_mean - p_target
            if abs(error) < 1e-7 or abs(dp_di) < 1e-15:
                break
            intercept -= error / dp_di
        boots.append(intercept)
        if (b+1) % 100 == 0:
            print(f"    {b+1}/{n_boot}  elapsed: {time.time()-t0:.0f}s")
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]  (±{(ci_hi-ci_lo)/2:.6f})")
    return ci_lo, ci_hi

# =============================================================================
#  RAPPORTAGE
# =============================================================================

def print_final_report(calibration_results: dict, sensitivity: dict,
                       sim_meta: dict, weights: dict):
    print(f"\n\n{'#'*72}")
    print("# HESTIA rev14 — KALIBRATIERAPPORT")
    print(f"# Gegenereerd: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"# N_populatie : {sim_meta['n_population']}")
    print(f"# Seed        : {sim_meta['random_seed']}")
    print(f"# Condities   : T={REF_CONDITIONS['tdb_start']}→{REF_CONDITIONS['tdb_end']}°C, "
          f"RH={REF_CONDITIONS['rh']}%, wind={REF_CONDITIONS['wind_ms']} m/s")
    print(f"# Gewichten   : W_T1={weights['W_T1']}, W_T2={weights['W_T2']}, W_C={weights['W_C']}, W_D={weights['W_D']}")
    print(f"# Simulatietijd: {sim_meta['elapsed_s']/60:.1f} min")
    print(f"{'#'*72}")

    print(f"\n{'='*72}")
    print("SAMENVATTING INTERCEPTS")
    print(f"{'='*72}")
    fmt = "  {:<22}  {:>12}  {:>10}  {:>8}  {:>6}"
    print(fmt.format('Eindpunt', 'Intercept', 'p_obs/10k', 'Conv.iter', 'Status'))
    print("  " + "-"*68)
    for ep_key, res in calibration_results.items():
        ep = ENDPOINTS[ep_key]
        conv = f"ja ({res['n_iter']})" if res['converged'] else f"NEE ({res['n_iter']})"
        p10k = res['p_target'] * 10_000
        print(fmt.format(ep_key, f"{res['intercept']:.6f}", f"{p10k:.4f}", conv, ep['status']))

    if sensitivity:
        print(f"\n  EP3 gevoeligheidsrange (EHBO, N/35.000):")
        for n_ehbo, ic in sorted(sensitivity.items()):
            p10k = n_ehbo / 35_000 * 10_000
            print(f"    N={n_ehbo:4d} ({p10k:.1f}/10k) → intercept: {ic:.6f}")

    # Output voor COLLAPSE_ENDPOINTS
    print(f"\n{'='*72}")
    print("PLAK-KLAAR: vervang COLLAPSE_ENDPOINTS in HESTIA_Data_Engine_CVR_v9.py")
    print(f"{'='*72}")
    print("\nCOLLAPSE_ENDPOINTS = {")
    for ep_key in ('ehs', 'hospitalisation', 'ehbo'):
        res = calibration_results[ep_key]
        ep = ENDPOINTS[ep_key]
        ic = res['intercept']
        p_obs_str = f"{ep['p_target']*10_000:.4f} / 10_000"
        if ep_key == 'ehs':
            n_part = 10_000
            n_obs = ep['p_target'] * n_part
        else:
            n_part = 35_000
            n_obs = ep['p_target'] * n_part
        print(f"    '{ep_key}': {{")
        print(f"        'label'        : '{ep['label']}',")
        print(f"        'p_obs'        : {n_obs:.1f} / {n_part},")
        print(f"        'n_obs'        : {n_obs:.1f},")
        print(f"        'n_participants': {n_part},")
        print(f"        'intercept_kal': {ic:.6f},   # rev14 gecalibreerd {datetime.now().strftime('%Y-%m-%d')}")
        print(f"        'source'       : '{ep['source']}',")
        print(f"        'status'       : '{ep['status']}',")
        if ep_key == 'ehbo' and sensitivity:
            print(f"        'sensitivity'  : {{")
            for n_ehbo, ic_s in sorted(sensitivity.items()):
                print(f"            {n_ehbo}: {ic_s:.6f},")
            print(f"        }},")
        print(f"    }},")
    print(f"}}")
    print(f"\n{'='*72}")

def save_results_json(calibration_results: dict, sensitivity: dict,
                      sim_meta: dict, sim_out: dict, weights: dict):
    payload = {
        'metadata': {
            'hestia_version'      : 'rev14',
            'script'              : 'intercept_estimation_rev14.py',
            'generated_utc'       : datetime.now(timezone.utc).isoformat(),
            'n_population'        : sim_meta['n_population'],
            'random_seed'         : sim_meta['random_seed'],
            'simulation_elapsed_s': sim_meta['elapsed_s'],
            'ref_conditions'      : REF_CONDITIONS,
            'model_weights'       : weights,
        },
        'population_statistics': {
            't_rect_max_mean' : float(sim_out['t_rect_max'].mean()),
            't_rect_max_p50'  : float(np.percentile(sim_out['t_rect_max'], 50)),
            't_rect_max_p90'  : float(np.percentile(sim_out['t_rect_max'], 90)),
            't_rect_max_p95'  : float(np.percentile(sim_out['t_rect_max'], 95)),
            'pct_above_39_5'  : float(np.mean(sim_out['t_rect_max'] > 39.5) * 100),
            'pct_above_40_5'  : float(np.mean(sim_out['t_rect_max'] > 40.5) * 100),
            'co_res_min_p50'  : float(np.percentile(sim_out['co_res_min'], 50)),
            'dehy_end_p50'    : float(np.percentile(sim_out['dehy_end'], 50)),
        },
        'calibration_results': {
            ep: {
                'intercept'   : res['intercept'],
                'p_target'    : res['p_target'],
                'p_final'     : res['p_final'],
                'error_final' : res['error_final'],
                'n_iter'      : res['n_iter'],
                'converged'   : res['converged'],
                'label'       : ENDPOINTS[ep]['label'],
                'source'      : ENDPOINTS[ep]['source'],
                'status'      : ENDPOINTS[ep]['status'],
            }
            for ep, res in calibration_results.items()
        },
        'sensitivity_ep3': {str(n): ic for n, ic in sensitivity.items()},
    }
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"{Fore.GREEN}Resultaten opgeslagen: {RESULTS_JSON}{Style.RESET_ALL}")

# =============================================================================
#  MAIN
# =============================================================================

def main():
    print(f"{Style.BRIGHT}HESTIA intercept_estimation_rev14.py (rev14 kalibratie){Style.RESET_ALL}")
    print(f"N={N_POPULATION:,}, seed={RANDOM_SEED}\n")

    # 1. Laad engine
    engine = _import_engine()
    weights = _get_weights_from_engine(engine)
    print(f"Gebruikte gewichten: W_T1={weights['W_T1']}, W_T2={weights['W_T2']}, W_C={weights['W_C']}, W_D={weights['W_D']}")

    # 2. Bouw referentieweer
    print("\nBouw referentie-meteorologie...")
    interp_data = build_reference_interp(REF_CONDITIONS)

    # 3. Populatie
    population = get_population(engine)
    n_pop = len(population)
    sim_meta = {'n_population': n_pop, 'random_seed': RANDOM_SEED}

    # 4. Simulatie
    sim_out = run_calibration_simulation(engine, population, interp_data)
    sim_meta['elapsed_s'] = sim_out.pop('elapsed_s')

    # 5. Kalibreer alle eindpunten
    calibration_results = {}
    for ep_key, ep_cfg in ENDPOINTS.items():
        res = calibrate_intercept(sim_out, ep_cfg['p_target'], weights, label=ep_cfg['label'])
        calibration_results[ep_key] = res

    # 6. Sensitiviteit EP3
    sensitivity = sensitivity_analysis_ehbo(sim_out, weights)

    # 7. Optionele bootstrap voor EP1
    boot = input("\nBootstrap 95% CI voor EP1? (duurt ~2-5 min, y/n, default=n): ").strip().lower()
    if boot in ('y', 'yes'):
        ci_lo, ci_hi = bootstrap_ci(sim_out, ENDPOINTS['ehs']['p_target'],
                                    calibration_results['ehs']['intercept'],
                                    weights, n_boot=500)
        calibration_results['ehs']['ci_95_lower'] = ci_lo
        calibration_results['ehs']['ci_95_upper'] = ci_hi

    # 8. Rapportage en opslag
    print_final_report(calibration_results, sensitivity, sim_meta, weights)
    save_results_json(calibration_results, sensitivity, sim_meta, sim_out, weights)

    print(f"\n{Fore.GREEN}Kalibratie voltooid. Plak de COLLAPSE_ENDPOINTS in uw engine.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()