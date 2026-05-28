"""
HESTIA_CVR_Console.py
=====================
Console visualisation of CVR module output for HESTIA.

Matches the visual style of HESTIA_Data_Engine.py:
  - colorama colour coding
  - Fixed-width ASCII panels
  - Horizontal bar charts as inline graphs
  - Colour thresholds consistent with calculate_adult_risk_classification()

Public API
----------
    from HESTIA_CVR_Console import (
        print_cvr_population_summary,   # population-level CVR statistics
        print_cvr_time_series,          # single-runner time series (debug/demo)
        print_cvr_comparison,           # side-by-side edition comparison
        calculate_cvr_risk_score,       # integer risk score 0-4
    )

    # Dutch aliases retained for backward compatibility:
    print_cvr_populatie_samenvatting  -> print_cvr_population_summary
    print_cvr_tijdreeks               -> print_cvr_time_series
    print_cvr_vergelijking            -> print_cvr_comparison

Version history
---------------
v1.0  2026-03  Initial release: HR, CVS index, CO reserve, AVA, dehydration.
v2.0  2026-03  AVA (arteriovenous anastomosis) indicator removed (rev05).
               At exercise intensity in warm conditions AVA blood flow is
               elevated -- the indicator never triggered and had no
               diagnostic value in the DtD scenario domain.
v3.0  2026-03  Collapse risk section added (rev06):
               two-phase logistic model output displayed with per-1000
               expected count and calibration transparency line.
v3.1  2026-03  All text, docstrings, and inline comments translated to
               English. Dutch function names kept as aliases.
               Nested helper functions _coll_kleur, _per1000_kleur promoted
               to module level with docstrings.
               Dead _waarde() function removed from print_cvr_comparison.
               Per-row delta thresholds: collapse rows use 0.05 instead of
               0.5 to show meaningful sub-percent differences.
               _pct_balk replaced by direct _balk call with fixed 5% scale
               for collapse display (avoids CVS colour mapping on sub-1%).
               Demo bouw_stats() extended with collapse risk keys.
               ava_nul parameter removed from demo maak_reeks().

Author  : HESTIA project / Veiligheidsregio Noord-Holland Noord (GHOR NHN)
"""


import numpy as np
from colorama import Fore, Back, Style, init

init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# OPMAAK-CONSTANTEN  (consistent met HESTIA_Data_Engine.py)
# ─────────────────────────────────────────────────────────────────────────────

BREEDTE = 72          # totale consolebreedte
BAR_MAX = 36          # maximale breedte van horizontale balk

# Colour thresholds -- consistent with calculate_adult_risk_classification()
def _hr_kleur(hr, hr_max):
    """Return colour for heart rate as fraction of HR_max."""
    pct = hr / hr_max if hr_max > 0 else 0
    if pct < 0.70:  return Fore.GREEN
    if pct < 0.85:  return Fore.YELLOW
    if pct < 0.95:  return Fore.MAGENTA
    return Fore.RED

def _cvs_kleur(cvs_index):
    """Return colour for cardiovascular stress index (fraction 0-1)."""
    if cvs_index < 0.70:  return Fore.GREEN
    if cvs_index < 0.85:  return Fore.YELLOW
    if cvs_index < 0.95:  return Fore.MAGENTA
    return Fore.RED

def _reserve_kleur(reserve):
    """Return colour for cardiac output reserve (L/min)."""
    if reserve > 5.0:   return Fore.GREEN
    if reserve > 3.0:   return Fore.YELLOW
    if reserve > 1.0:   return Fore.MAGENTA
    return Fore.RED

def _dehy_kleur(pct):
    """Return colour for dehydration as percentage of body mass."""
    if pct < 2.0:   return Fore.GREEN
    if pct < 3.0:   return Fore.YELLOW
    if pct < 4.0:   return Fore.MAGENTA
    return Fore.RED


def _coll_kleur(pct):
    """
    Return colour for collapse risk percentage.

    Thresholds reflect absolute probability scale (not CVS-index scale):
    <  5% -- low (GREEN)
    < 15% -- elevated (YELLOW)
    < 35% -- high (MAGENTA)
    >= 35% -- extreme (RED)

    Added v3.0; promoted to module level in v3.1.
    """
    if pct < 5:   return Fore.GREEN
    if pct < 15:  return Fore.YELLOW
    if pct < 35:  return Fore.MAGENTA
    return Fore.RED


def _per1000_kleur(n):
    """
    Return colour for expected collapses per 1000 participants.

    Calibrated against DtD 2024 observed rate (1.43 / 1000):
    < 0.5  -- low (GREEN)
    < 1.5  -- moderate, near observed DtD 2024 rate (YELLOW)
    < 3.0  -- high (MAGENTA)
    >= 3.0 -- extreme (RED)

    Added v3.0; promoted to module level in v3.1.
    """
    if n < 0.5:   return Fore.GREEN
    if n < 1.5:   return Fore.YELLOW
    if n < 3.0:   return Fore.MAGENTA
    return Fore.RED

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _lijn(karakter='─', breedte=BREEDTE):
    return karakter * breedte

def _koptekst(tekst, karakter='═'):
    pad = max(0, BREEDTE - len(tekst) - 4)
    links = pad // 2
    rechts = pad - links
    return (Fore.CYAN + karakter * links + '  ' +
            Style.BRIGHT + tekst + Style.NORMAL +
            '  ' + karakter * rechts + Style.RESET_ALL)

def _balk(waarde, maximum, breedte=BAR_MAX, kleur=Fore.GREEN):
    """Return a horizontal ASCII bar scaled to value/maximum."""
    if maximum <= 0:
        gevuld = 0
    else:
        gevuld = int(min(1.0, waarde / maximum) * breedte)
    leeg = breedte - gevuld
    return kleur + '█' * gevuld + Fore.WHITE + Style.DIM + '░' * leeg + Style.RESET_ALL

def _pct_balk(pct_0_100, breedte=BAR_MAX):
    """Return a horizontal bar for a percentage in [0, 100]."""
    kleur = (_cvs_kleur(pct_0_100 / 100))
    return _balk(pct_0_100, 100, breedte, kleur)


# ─────────────────────────────────────────────────────────────────────────────
# 1. POPULATION SUMMARY  (after run_monte_carlo_adult)
# ─────────────────────────────────────────────────────────────────────────────

def print_cvr_population_summary(
    stats: dict,
    start_group_label: str = "",
    n_participants: int = 0,
    edition: str = "",
):
    """
    Print a structured population-level CVR summary after a Monte Carlo run.

    Renders five sections: heart rate, cardiovascular stress index,
    cardiac output reserve, dehydration, and collapse risk (only shown
    when p_collapse_gemiddeld is present in the stats dict).

    Expected keys in stats (all produced by run_monte_carlo_adult):
      hr_piek_p50/p95          peak HR percentiles (beats/min)
      hr_max_populatie         mean age-predicted HR_max
      cvs_piek_p50/p95         CVS index percentiles (fraction 0-1)
      pct_cvs_boven_90         % participants with CVS > 0.90
      pct_decompensatie        % participants with CO_reserve < 2 L/min
      co_reserve_min_p50/p05   CO reserve percentiles (L/min)
      dehy_pct_p50/p95         end-dehydration percentiles (%)
      p_collapse_gemiddeld     mean individual collapse probability (%)
      p_collapse_p95           P95 collapse probability (%)
      pct_hoog_collapsrisico   % participants with P(collapse) > 50%
      verwacht_collapsen_per_1000  expected collapses per 1000 participants
      collapse_intercept_kal   calibrated logistic intercept
      collapse_p_obs_pct       observed calibration incidence (%)

    Parameters
    ----------
    stats : dict
        Statistics dict returned by run_monte_carlo_adult().
    start_group_label : str
        Start group label shown in the panel header.
    n_participants : int
        Number of simulated participants (shown in header).
    edition : str
        Event edition label (e.g. "DtD 2024").

    Changes vs previous version
    ---------------------------
    v2.0: AVA section removed (pct_ava_gesloten / ava_open no longer exists).
    v3.0: Collapse risk section added.
    v3.1: Renamed print_cvr_population_summary. Parameters renamed.
          _coll_kleur and _per1000_kleur moved to module level.
          Collapse bar uses _balk with fixed 5% scale instead of _pct_balk.
    """
    print()
    print(_koptekst(f"CVR ANALYSIS  {edition}  {start_group_label}"))
    if n_participants:
        print(f"  {Fore.WHITE}Population: {Style.BRIGHT}{n_participants}{Style.RESET_ALL} simulated participants")
    print(_lijn())

    # ── Hartslag ──────────────────────────────────────────────────────────────
    print(f"\n  {Style.BRIGHT}HEART RATE (estimated){Style.RESET_ALL}  "
          f"{Fore.WHITE + Style.DIM}HR = CO_jos3 / SV  [Lloyd 2022 Eq. 2]{Style.RESET_ALL}")

    hr_p50  = stats.get('hr_piek_p50',  0)
    hr_p95  = stats.get('hr_piek_p95',  0)
    hr_max  = stats.get('hr_max_populatie', 185)

    kleur50 = _hr_kleur(hr_p50, hr_max)
    kleur95 = _hr_kleur(hr_p95, hr_max)

    print(f"  {'Median peak HR':<28} {kleur50}{hr_p50:>5.0f} bpm{Style.RESET_ALL}  "
          f"{_balk(hr_p50, hr_max, 20, kleur50)}  "
          f"{kleur50}{hr_p50/hr_max*100:>4.0f}% HR_max{Style.RESET_ALL}")
    print(f"  {'P95 peak HR':<28} {kleur95}{hr_p95:>5.0f} bpm{Style.RESET_ALL}  "
          f"{_balk(hr_p95, hr_max, 20, kleur95)}  "
          f"{kleur95}{hr_p95/hr_max*100:>4.0f}% HR_max{Style.RESET_ALL}")

    # ── CVS-index ─────────────────────────────────────────────────────────────
    print(f"\n  {Style.BRIGHT}CARDIOVASCULAR STRESS  (CVS index = CO / CO_max){Style.RESET_ALL}")

    cvs_p50 = stats.get('cvs_piek_p50', 0) * 100
    cvs_p95 = stats.get('cvs_piek_p95', 0) * 100
    pct_90  = stats.get('pct_cvs_boven_90', 0)

    print(f"  {'Median peak CVS index':<28} {_cvs_kleur(cvs_p50/100)}{cvs_p50:>5.1f}%{Style.RESET_ALL}  "
          f"{_pct_balk(cvs_p50)}")
    print(f"  {'P95 peak CVS index':<28} {_cvs_kleur(cvs_p95/100)}{cvs_p95:>5.1f}%{Style.RESET_ALL}  "
          f"{_pct_balk(cvs_p95)}")
    print(f"  {'% with CVS index > 90%':<28} "
          f"{_cvs_kleur(pct_90/100)}{pct_90:>5.1f}%{Style.RESET_ALL}  "
          f"{_pct_balk(pct_90)}")

    # ── CO-reserve ───────────────────────────────────────────────────────────
    print(f"\n  {Style.BRIGHT}CARDIAC OUTPUT RESERVE{Style.RESET_ALL}  "
          f"{Fore.WHITE + Style.DIM}< 2.0 L/min = decompensation risk{Style.RESET_ALL}")

    res_p50 = stats.get('co_reserve_min_p50', 0)
    res_p05 = stats.get('co_reserve_min_p05', 0)
    pct_dec = stats.get('pct_decompensatie', 0)

    print(f"  {'Median min CO reserve':<28} "
          f"{_reserve_kleur(res_p50)}{res_p50:>5.1f} L/min{Style.RESET_ALL}  "
          f"{_balk(res_p50, 10, 20, _reserve_kleur(res_p50))}")
    print(f"  {'P05 min CO reserve':<28} "
          f"{_reserve_kleur(res_p05)}{res_p05:>5.1f} L/min{Style.RESET_ALL}  "
          f"{_balk(res_p05, 10, 20, _reserve_kleur(res_p05))}")
    print(f"  {'% with decompensation risk':<28} "
          f"{_cvs_kleur(pct_dec/100)}{pct_dec:>5.1f}%{Style.RESET_ALL}  "
          f"{_pct_balk(pct_dec)}")

    # ── Dehydratie ────────────────────────────────────────────────────────────
    print(f"\n  {Style.BRIGHT}DEHYDRATION{Style.RESET_ALL}")

    dhy_p50 = stats.get('dehy_pct_p50', 0)
    dhy_p95 = stats.get('dehy_pct_p95', 0)

    print(f"  {'Median end dehydration':<28} "
          f"{_dehy_kleur(dhy_p50)}{dhy_p50:>5.1f}%{Style.RESET_ALL}  "
          f"{_balk(dhy_p50, 6, 20, _dehy_kleur(dhy_p50))}")
    print(f"  {'P95 end dehydration':<28} "
          f"{_dehy_kleur(dhy_p95)}{dhy_p95:>5.1f}%{Style.RESET_ALL}  "
          f"{_balk(dhy_p95, 6, 20, _dehy_kleur(dhy_p95))}")

    # ── Collapsrisico ─────────────────────────────────────────────────────────
    p_gem   = stats.get('p_collapse_gemiddeld',        float('nan'))
    p95_coll= stats.get('p_collapse_p95',              float('nan'))
    pct_hoog= stats.get('pct_hoog_collapsrisico',      float('nan'))
    per1000 = stats.get('verwacht_collapsen_per_1000', float('nan'))
    intercept_kal = stats.get('collapse_intercept_kal', float('nan'))
    p_obs_pct     = stats.get('collapse_p_obs_pct',     float('nan'))

    if not np.isnan(p_gem):
        print(f"\n  {Style.BRIGHT}COLLAPSRISICO{Style.RESET_ALL}  "
              f"{Fore.WHITE + Style.DIM}tweefasig logistisch model  "
              f"(T_rect ≥39.5/40.5°C + CO_reserve + dehydratie){Style.RESET_ALL}")

        print(f"  {'Mean collapse risk':<28} "
              f"{_coll_kleur(p_gem)}{p_gem:>5.2f}%{Style.RESET_ALL}  "
              f"{_balk(min(p_gem, 5.0), 5.0, BAR_MAX, _coll_kleur(p_gem))}")
        print(f"  {'P95 collapse risk':<28} "
              f"{_coll_kleur(p95_coll)}{p95_coll:>5.2f}%{Style.RESET_ALL}  "
              f"{_balk(min(p95_coll, 5.0), 5.0, BAR_MAX, _coll_kleur(p95_coll))}")
        print(f"  {'% participants risk > 50%':<28} "
              f"{_coll_kleur(pct_hoog)}{pct_hoog:>5.1f}%{Style.RESET_ALL}  "
              f"{_pct_balk(pct_hoog)}")
        print(f"  {'Expected per 1000 participants':<28} "
              f"{_per1000_kleur(per1000)}{per1000:>5.2f}{Style.RESET_ALL}  "
              f"{Fore.WHITE + Style.DIM}based on mean collapse probability{Style.RESET_ALL}")
        if not np.isnan(intercept_kal):
            print(f"  {Fore.WHITE + Style.DIM}"
                  f"Calibration: intercept={intercept_kal:+.3f}  "
                  f"target={p_obs_pct:.4f}% (DtD 2024 admissions: 50/35,000)"
                  f"{Style.RESET_ALL}")

    print()
    print(_lijn())


# ─────────────────────────────────────────────────────────────────────────────
# 2. SINGLE-RUNNER TIME SERIES  (optional -- debug / illustration)
# ─────────────────────────────────────────────────────────────────────────────

def print_cvr_time_series(cvr_time_series, runner_label: str = "", hr_max: float = 185):
    """
    Print a compact time series of CVRState objects for a single runner.

    Each row = one time step, showing HR, HR%, CVS index, CO demand,
    CO reserve, dehydration, core temperature, and central blood temperature,
    with colour-coded risk levels. Sub-sampled to max 20 rows.

    Parameters
    ----------
    cvr_time_series : CVRTimeSeries
        Time series returned by koppel_cvr_aan_jos3().
    runner_label : str
        Descriptive label shown in the panel header.
    hr_max : float
        Age-predicted maximal HR for this runner (beats/min).

    Changes vs previous version
    ---------------------------
    v2.0: AVA column removed from table header and rows.
    v3.1: Renamed print_cvr_time_series. Parameters renamed.
    """
    states = cvr_time_series.states
    if not states:
        print(Fore.YELLOW + "  No CVR data available." + Style.RESET_ALL)
        return

    print()
    print(_koptekst(f"CVR TIME SERIES  {runner_label}"))
    print(f"  {'Min':>4}  {'HR':>5}  {'HR%':>4}  {'CVS':>5}  "
          f"{'CO_req':>6}  {'CO_res':>6}  {'Dehy':>5}  T_core  T_cb")
    print(_lijn('─'))

    step = max(1, len(states) // 20)   # maximum 20 rows in console

    for s in states[::step]:
        hr_pct = s.HR / hr_max * 100 if hr_max > 0 else 0
        khr    = _hr_kleur(s.HR, hr_max)
        kcvs   = _cvs_kleur(s.CVS_index)
        kres   = _reserve_kleur(s.CO_reserve)
        kdhy   = _dehy_kleur(s.dehydratie_pct)
        dec_s  = f" {Fore.RED}DECOMP{Style.RESET_ALL}" if s.decompensatie else ""

        print(
            f"  {s.t_min:>4.0f}  "
            f"{khr}{s.HR:>5.0f}{Style.RESET_ALL}  "
            f"{khr}{hr_pct:>3.0f}%{Style.RESET_ALL}  "
            f"{kcvs}{s.CVS_index*100:>4.0f}%{Style.RESET_ALL}  "
            f"{s.CO_gevraagd:>5.1f}L  "
            f"{kres}{s.CO_reserve:>5.1f}L{Style.RESET_ALL}  "
            f"{kdhy}{s.dehydratie_pct:>4.1f}%{Style.RESET_ALL}  "
            f"{s.t_core:>5.2f}°  "
            f"{s.t_cb:>5.2f}°"
            f"{dec_s}"
        )

    print(_lijn('─'))
    print(f"  Peak HR: {_hr_kleur(cvr_time_series.max_HR(), hr_max)}"
          f"{cvr_time_series.max_HR():.0f} bpm{Style.RESET_ALL}  |  "
          f"Max CVS: {_cvs_kleur(cvr_time_series.max_CVS_index())}"
          f"{cvr_time_series.max_CVS_index()*100:.0f}%{Style.RESET_ALL}  |  "
          f"Min reserve: {_reserve_kleur(cvr_time_series.min_CO_reserve())}"
          f"{cvr_time_series.min_CO_reserve():.1f} L/min{Style.RESET_ALL}")

    decomp_t = cvr_time_series.decompensatie_tijdstip()
    if decomp_t is not None:
        print(f"  {Fore.RED}▲ Decompensation risk from t={decomp_t:.0f} min{Style.RESET_ALL}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPARISON OF TWO EDITIONS  (core output for DtD 2024 vs 2025)
# ─────────────────────────────────────────────────────────────────────────────

def print_cvr_comparison(
    stats_a: dict, label_a: str,
    stats_b: dict, label_b: str,
    n_a: int = 0, n_b: int = 0,
):
    """
    Print a side-by-side comparison of CVR statistics for two editions,
    with a delta column (Δ = A − B) and per-row colour coding.

    Delta thresholds per metric category:
      Standard physiological metrics : 0.5 (absolute)
      Collapse risk percentages      : 0.05 (sub-percent scale)
      Expected collapses per 1000    : 0.05

    Parameters
    ----------
    stats_a / stats_b : dict
        Statistics dicts from run_monte_carlo_adult() for each edition.
    label_a / label_b : str
        Edition labels (e.g. 'DtD 2024' / 'DtD 2025').
    n_a / n_b : int
        Number of simulated participants shown in column headers.

    Changes vs previous version
    ---------------------------
    v2.0: AVA row (pct_ava_gesloten) removed.
    v3.0: Collapse risk rows added.
    v3.1: Renamed print_cvr_comparison. Dead _waarde() helper removed.
          Per-row delta thresholds introduced (collapse rows use 0.05).
    """

    def _delta_kleur(delta, hoger_is_slechter=True, threshold=0.5):
        if abs(delta) < threshold:
            return Fore.WHITE
        if hoger_is_slechter:
            return Fore.RED if delta > 0 else Fore.GREEN
        else:
            return Fore.GREEN if delta > 0 else Fore.RED

    print()
    print(_koptekst(f"CVR VERGELIJKING  {label_a}  vs  {label_b}"))
    lbl_a = f"{label_a}" + (f" (n={n_a})" if n_a else "")
    lbl_b = f"{label_b}" + (f" (n={n_b})" if n_b else "")
    print(f"  {'Maatstaf':<35}  {lbl_a:>14}  {lbl_b:>14}  {'Δ':>8}")
    print(_lijn('─'))

    rijen = [
        # (label, key_a, key_b, factor, fmt, higher_is_worse, delta_threshold)
        ('Median peak HR (bpm)',          'hr_piek_p50',              'hr_piek_p50',              1,   '.0f', True,  0.5),
        ('P95 peak HR (bpm)',             'hr_piek_p95',              'hr_piek_p95',              1,   '.0f', True,  0.5),
        ('Median peak CVS index (%)',     'cvs_piek_p50',             'cvs_piek_p50',             100, '.1f', True,  0.5),
        ('P95 peak CVS index (%)',        'cvs_piek_p95',             'cvs_piek_p95',             100, '.1f', True,  0.5),
        ('% CVS index > 90%',             'pct_cvs_boven_90',         'pct_cvs_boven_90',         1,   '.1f', True,  0.5),
        ('% with decompensation risk',    'pct_decompensatie',        'pct_decompensatie',        1,   '.1f', True,  0.5),
        ('Median min CO reserve (L)',     'co_reserve_min_p50',       'co_reserve_min_p50',       1,   '.1f', False, 0.5),
        ('P05 min CO reserve (L)',        'co_reserve_min_p05',       'co_reserve_min_p05',       1,   '.1f', False, 0.5),
        ('Mean collapse risk (%)',        'p_collapse_gemiddeld',     'p_collapse_gemiddeld',     1,   '.2f', True,  0.05),
        ('P95 collapse risk (%)',         'p_collapse_p95',           'p_collapse_p95',           1,   '.2f', True,  0.05),
        ('Expected per 1000',             'verwacht_collapsen_per_1000', 'verwacht_collapsen_per_1000', 1, '.2f', True, 0.05),
        ('% high collapse risk (>50%)',   'pct_hoog_collapsrisico',   'pct_hoog_collapsrisico',   1,   '.1f', True,  0.5),
        ('Median end dehydration (%)',    'dehy_pct_p50',             'dehy_pct_p50',             1,   '.1f', True,  0.5),
        ('P95 end dehydration (%)',       'dehy_pct_p95',             'dehy_pct_p95',             1,   '.1f', True,  0.5),
    ]

    for label, key_a, key_b, factor, fmt, hoger_is_slechter, d_thresh in rijen:
        va = stats_a.get(key_a, float('nan')) * factor
        vb = stats_b.get(key_b, float('nan')) * factor

        if np.isnan(va) or np.isnan(vb):
            delta_str = '   —'
            dkleur    = Fore.WHITE
        else:
            delta = va - vb
            dkleur = _delta_kleur(delta, hoger_is_slechter, threshold=d_thresh)
            delta_str = f"{delta:+.2f}"

        va_str = f"{va:{fmt}}" if not np.isnan(va) else '—'
        vb_str = f"{vb:{fmt}}" if not np.isnan(vb) else '—'

        # Kleur waarden op basis van welke hoger/slechter is
        if not np.isnan(va) and not np.isnan(vb):
            if hoger_is_slechter:
                kl_a = Fore.RED   if va > vb else Fore.GREEN
                kl_b = Fore.GREEN if va > vb else Fore.RED
            else:
                kl_a = Fore.GREEN if va > vb else Fore.RED
                kl_b = Fore.RED   if va > vb else Fore.GREEN
            if abs(va - vb) < d_thresh:
                kl_a = kl_b = Fore.WHITE
        else:
            kl_a = kl_b = Fore.WHITE

        print(f"  {label:<35}  "
              f"{kl_a}{va_str:>14}{Style.RESET_ALL}  "
              f"{kl_b}{vb_str:>14}{Style.RESET_ALL}  "
              f"{dkleur}{delta_str:>8}{Style.RESET_ALL}")

    print(_lijn())
    print(f"  {Fore.WHITE + Style.DIM}Green = better  |  Red = worse  |  "
          f"delta = {label_a} - {label_b}{Style.RESET_ALL}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CVR RISK SCORE  (extension of the existing HESTIA risk classification)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_cvr_risk_score(cvs_index: float, co_reserve: float,
                              hr_pct_max: float) -> int:
    """
    Return a composite CVR risk level 0-4.

    Consistent with calculate_adult_risk_classification() in the Data Engine.
    Final score = maximum across CVS, CO reserve, and HR sub-scores.

    Score  CVS index   CO reserve   %HR_max   Label
    -----  ----------  -----------  --------  -------
      0    < 70%       > 5.0 L/min  < 70%     No risk
      1    < 80%       > 3.0 L/min  < 80%     Low risk
      2    < 90%       > 2.0 L/min  < 90%     Moderate risk
      3    < 95%       > 1.0 L/min  < 95%     High risk
      4    >= 95%      <= 1.0 L/min >= 95%    Extreme risk

    Parameters
    ----------
    cvs_index : float
        Cardiovascular stress index = CO_demanded / CO_max (fraction 0-1).
    co_reserve : float
        Minimum cardiac output reserve (L/min).
    hr_pct_max : float
        Heart rate as percentage of HR_max (0-100).

    Returns
    -------
    int
        Risk score 0-4.
    """
    scores = {
        'cvs':     (0 if cvs_index < 0.70 else 1 if cvs_index < 0.80
                    else 2 if cvs_index < 0.90 else 3 if cvs_index < 0.95 else 4),
        'reserve': (0 if co_reserve > 5.0 else 1 if co_reserve > 3.0
                    else 2 if co_reserve > 2.0 else 3 if co_reserve > 1.0 else 4),
        'hr_pct':  (0 if hr_pct_max < 70 else 1 if hr_pct_max < 80
                    else 2 if hr_pct_max < 90 else 3 if hr_pct_max < 95 else 4),
    }
    return max(scores.values())


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  (draai als standalone script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude')
    sys.path.insert(0, '/mnt/user-data/outputs')

    from HESTIA_CVR_Module_v2 import (
        RunnerProfile, JOS3Outputs, CVRModel,
        koppel_cvr_aan_jos3, CVRTimeSeries
    )

    print()
    print(Fore.CYAN + Style.BRIGHT +
          "  HESTIA CVR Console — Demonstratie DtD 2024 vs 2025" +
          Style.RESET_ALL)
    print()

    # ── Gesimuleerde JOS-3 tijdreeksen (zoals in vorige module) ───────────────
    def maak_reeks(co_start, co_eind, tc_start, tc_eind,
                   tcb_start, tcb_eind, zweet_kgh,
                   bfsk_start, bfsk_eind, n=100):
        reeks = []
        for i in range(n):
            f    = (i / n) ** 0.5
            co   = co_start   + f * (co_eind   - co_start)
            tc   = tc_start   + f * (tc_eind   - tc_start)
            tcb  = tcb_start  + f * (tcb_eind  - tcb_start)
            bfsk = bfsk_start + f * (bfsk_eind - bfsk_start)
            gv   = zweet_kgh * (i / 60)
            reeks.append(JOS3Outputs(
                t_min=i, cardiac_output=co, t_core_mean=tc, t_cb=tcb,
                weight_loss_g_s=gv, bf_skin_total=bfsk,
                bf_ava_hand=1.2, bf_ava_foot=1.2,  # open state (exercise in heat)
            ))
        return reeks

    lopers = [
        RunnerProfile(mass=70, height=175, age=35, sex='male',   vo2max=50),
        RunnerProfile(mass=65, height=168, age=52, sex='male',   vo2max=38),
        RunnerProfile(mass=58, height=163, age=38, sex='female', vo2max=42),
    ]

    reeks_2024 = maak_reeks(800,1480, 37.0,40.4, 36.8,39.8, 1.5, 80,480)
    reeks_2025 = maak_reeks(780,1220, 37.0,39.5, 36.8,39.1, 1.1, 80,340)

    # ── Bouw gesimuleerde populatiestatistieken voor demonstratie ─────────────
    def bouw_stats(reeksen_per_loper):
        hr_pieken, cvs_pieken, reserves, dehys = [], [], [], []
        hr_maxen = []
        for loper, reeks in reeksen_per_loper:
            ts = koppel_cvr_aan_jos3(loper, reeks)
            hr_pieken.append(ts.max_HR())
            cvs_pieken.append(ts.max_CVS_index())
            reserves.append(ts.min_CO_reserve())
            dehys.append(ts.eindstaat().dehydratie_pct)
            hr_maxen.append(CVRModel(loper).HR_max)

        hr_a = np.array(hr_pieken)
        cv_a = np.array(cvs_pieken)
        re_a = np.array(reserves)
        dh_a = np.array(dehys)
        hm_a = np.array(hr_maxen)

        # Two-phase collapse risk (mirrors run_monte_carlo_adult logic)
        t_core_arr = np.array([koppel_cvr_aan_jos3(l, r).eindstaat().t_core
                               for l, r in reeksen_per_loper])
        W_T1, W_T2, W_C, W_D = 1.0, 4.0, 0.8, 0.5
        P_OBS_D = 50 / 35_000
        z0 = (W_T1 * np.clip(t_core_arr - 39.5, 0, None)
              + W_T2 * np.clip(t_core_arr - 40.5, 0, None)
              + W_C  * np.clip(2.0 - re_a, 0, None)
              + W_D  * np.clip(dh_a - 3.0, 0, None))
        intc = float(np.log(P_OBS_D/(1-P_OBS_D))) - float(np.nanmean(z0))
        for _ in range(5):
            pp = 1/(1+np.exp(-(intc+z0)))
            g  = float(np.nanmean(pp*(1-pp)))
            if g < 1e-12: break
            intc += (P_OBS_D - float(np.nanmean(pp))) / g
        p_coll = 1/(1+np.exp(-(intc+z0)))
        return {
            'hr_piek_p50':               np.percentile(hr_a, 50),
            'hr_piek_p95':               np.percentile(hr_a, 95),
            'cvs_piek_p50':              np.percentile(cv_a, 50),
            'cvs_piek_p95':              np.percentile(cv_a, 95),
            'pct_cvs_boven_90':          np.mean(cv_a > 0.90) * 100,
            'pct_decompensatie':         np.mean(re_a < 2.0)  * 100,
            'co_reserve_min_p50':        np.percentile(re_a, 50),
            'co_reserve_min_p05':        np.percentile(re_a,  5),
            'dehy_pct_p50':              np.percentile(dh_a, 50),
            'dehy_pct_p95':              np.percentile(dh_a, 95),
            'hr_max_populatie':          np.mean(hm_a),
            'p_collapse_gemiddeld':      float(np.nanmean(p_coll)) * 100,
            'p_collapse_p95':            float(np.nanpercentile(p_coll, 95)) * 100,
            'pct_hoog_collapsrisico':    float(np.nanmean(p_coll > 0.50)) * 100,
            'verwacht_collapsen_per_1000': float(np.nanmean(p_coll)) * 1000,
            'collapse_intercept_kal':    float(intc),
            'collapse_p_obs_pct':        P_OBS_D * 100,
            'p_collapse_per_sim':        p_coll,
        }

    pairs_2024 = [(l, reeks_2024) for l in lopers]
    pairs_2025 = [(l, reeks_2025) for l in lopers]
    stats_2024 = bouw_stats(pairs_2024)
    stats_2025 = bouw_stats(pairs_2025)

    # ── 1. Populatiesamenvatting per editie ───────────────────────────────────
    print_cvr_population_summary(stats_2024, "Business Run 5", 3, "DtD 2024")
    print_cvr_population_summary(stats_2025, "Business Run 5", 3, "DtD 2025")

    # ── 2. Tijdreeks van één loper ────────────────────────────────────────────
    ts_2024 = koppel_cvr_aan_jos3(lopers[1], reeks_2024)  # male 52y
    print_cvr_time_series(ts_2024, "Male 52y VO2max=38 -- DtD 2024",
                          hr_max=CVRModel(lopers[1]).HR_max)

    # ── 3. Vergelijking 2024 vs 2025 ─────────────────────────────────────────
    print_cvr_comparison(
        stats_2024, "DtD 2024", stats_2025, "DtD 2025",
        n_a=3, n_b=3
    )

# =============================================================================
# BACKWARD-COMPATIBLE DUTCH ALIASES  (v3.1)
# Existing code using Dutch function names continues to work unchanged.
# New code should use the English names.
# =============================================================================
print_cvr_populatie_samenvatting = print_cvr_population_summary
print_cvr_tijdreeks              = print_cvr_time_series
print_cvr_vergelijking           = print_cvr_comparison
