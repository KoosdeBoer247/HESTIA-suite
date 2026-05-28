"""
HESTIA_CVR_Module_v2.py
=======================
Cardiovascular Response module voor HESTIA — versie 2.

Herziene architectuur op basis van volledige JOS-3 outputvariabelen:
  - cardiac_output [L/h] uit JOS-3 als directe CO-vraag (geen schatting meer)
  - bf_skin, bf_ava_hand/foot, t_cb als aanvullende JOS-3 inputs
  - HR berekend via Lloyd et al. (2022) Eq. 2: CO = SV × HR

Bronnen:
  Lloyd A, Fiala D, Heyde C, Havenith G (2022).
    "A mathematical model for predicting cardiovascular responses at rest
    and during exercise in demanding environmental conditions."
    J Appl Physiol 133(2):247–261. doi:10.1152/japplphysiol.00619.2021
    PMC9342140 — CC-BY 4.0

  Takahashi Y et al. (2021). Thermoregulation Model JOS-3 with New Open
    Source Code. Energy & Buildings. doi:10.1016/j.enbuild.2020.110575

  Tanaka H et al. (2001). Age-predicted maximal heart rate revisited.
    J Am Coll Cardiol 37:153–156.  [HR_max = 208 - 0.7×leeftijd]

  Rowell LB (1986). Human Circulation: Regulation During Physical Stress.
    Oxford University Press. [CO_max bij inspanning in warmte]

  Gonzalez-Alonso J et al. (2008). Haemodynamics and the human
    cardiovascular response to heat and exercise.
    J Physiol 586:45–49.  [SV-daling bij hittestress]

Auteur  : HESTIA project / Veiligheidsregio NHN
Versie  : 2.0 — maart 2026

Regel 264 uit gecomment. ivm correctie voor females


"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerProfile:
    """
    Antropometrische en fysiologische kenmerken van één loper.
    vo2max in mL/kg/min — toe te voegen aan Monte Carlo-sampling in HESTIA.

    Aanbevolen Monte Carlo-verdeling voor vo2max bij DtD-populatie:
        Normaal(μ=45, σ=10) mL/kg/min, begrensd op [20, 80]
        Bron: Scharhag-Rosenberger et al. (2010) voor recreatieve hardlopers
    """
    mass:    float    # kg
    height:  float    # cm
    age:     float    # jaren
    sex:     str      # 'male' | 'female'
    vo2max:  float    # mL/kg/min


@dataclass
class JOS3Outputs:
    """
    Relevante JOS-3 outputs op één tijdstip.
    Eenheden exact zoals gepubliceerd in pythermalcomfort-documentatie.
    """
    t_min:               float          # tijdstip in minuten (berekend door HESTIA)
    cardiac_output:      float          # L/h  — som alle bloedstromen
    t_core_mean:         float          # °C   — gemiddelde kerntemperatuur (t_core.mean())
    t_cb:                float          # °C   — centrale bloedtemperatuur
    weight_loss_g_s:     float          # g/s  — cumulatief gewichtsverlies zweten+ademhaling
    bf_skin_total:       float          # L/h  — som bf_skin over alle 17 segmenten
    bf_ava_hand:         float          # L/h  — AVA bloedstroom hand
    bf_ava_foot:         float          # L/h  — AVA bloedstroom voet


@dataclass
class CVRState:
    """Cardiovasculaire toestand op één tijdstip — output van CVR-module."""
    t_min:              float = 0.0
    # Hartslagschatting (primaire nieuwe uitkomst)
    HR:                 float = 0.0    # slagen/min — geschat
    HR_max:             float = 0.0    # slagen/min — leeftijdsafhankelijk maximum
    HR_reserve_pct:     float = 0.0    # % van HR_reserve benut (Karvonen)
    # Cardiac output
    CO_gevraagd:        float = 0.0    # L/min — uit JOS-3 (cardiac_output / 60)
    CO_max:             float = 0.0    # L/min — maximum bij huidige condities
    CO_reserve:         float = 0.0    # L/min — beschikbare reserve
    SV:                 float = 0.0    # mL/slag — slagvolume
    # Cardiovasculaire belasting
    CVS_index:          float = 0.0    # CO_gevraagd / CO_max  (0–1)
    decompensatie:      bool  = False  # CO_reserve < drempel
    # AVA-status (thermoregulatoir signaal)
    ava_open:           bool  = True   # False = AVA gesloten → grens bereikt
    # Dehydratie
    dehydratie_pct:     float = 0.0    # % lichaamsgewichtsverlies
    # Doorgegeven JOS-3 waarden
    t_core:             float = 0.0
    t_cb:               float = 0.0
    bf_skin_total_lh:   float = 0.0   # L/h voor logging


@dataclass
class CVRTimeSeries:
    """Tijdreeks van CVRState-objecten voor één loper."""
    states: List[CVRState] = field(default_factory=list)

    def append(self, s: CVRState):
        self.states.append(s)

    def max_CVS_index(self) -> float:
        return max(s.CVS_index for s in self.states) if self.states else 0.0

    def max_HR(self) -> float:
        return max(s.HR for s in self.states) if self.states else 0.0

    def min_CO_reserve(self) -> float:
        return min(s.CO_reserve for s in self.states) if self.states else 0.0

    def decompensatie_tijdstip(self) -> Optional[float]:
        for s in self.states:
            if s.decompensatie:
                return s.t_min
        return None

    def ava_sluiting_tijdstip(self) -> Optional[float]:
        """Eerste tijdstip waarop AVA sluit — vroegst detecteerbaar signaal."""
        for s in self.states:
            if not s.ava_open:
                return s.t_min
        return None

    def eindstaat(self) -> Optional[CVRState]:
        return self.states[-1] if self.states else None


# ─────────────────────────────────────────────────────────────────────────────
# CVR MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CVRModel:
    """
    CVR Model v2 — gekoppeld aan JOS-3 via cardiac_output als directe input.

    Kernwijziging t.o.v. v1:
        CO_gevraagd = jos3.cardiac_output / 60   (L/h → L/min)
    Dit vervangt de MET-gebaseerde schatting uit v1 volledig.

    HR-schatting via Lloyd et al. (2022) Eq. 2:
        CO = SV × HR  →  HR = CO / SV
    SV gecorrigeerd voor hittestress via Gonzalez-Alonso (2008).

    JOS-3 cardiac index (ci, default 2.59 L/min/m²) schiet tekort
    als begrenzing omdat het een vaste schaalfactor is, geen fysiologisch
    maximum. CO_max in deze module is dat maximum wél — individueel
    bepaald via VO2max, leeftijd, hittestress en dehydratie.
    """

    # Drempelwaarden
    DECOMPENSATIE_RESERVE = 2.0    # L/min
    AVA_SLUIT_DREMPEL     = 0.10   # L/h — AVA vrijwel gesloten

    def __init__(self, runner: RunnerProfile):
        self.runner = runner
        self._vo2max_abs = (runner.vo2max * runner.mass) / 1000.0  # L/min
        self._bereken_basisparameters()

    # ── Basisparameters (Lloyd 2022, Tanaka 2001) ─────────────────────────────

    def _bereken_basisparameters(self):
        """
        Eenmalige berekening van leeftijds- en fitnessafhankelijke grenzen.

        SV_max  (Eq. 6 Lloyd 2022): SVmax = 40.59 + 24.81 × VO2max_abs
        SV_rust (Eq. 7 Lloyd 2022): 85.1 mL/slag populatiegemiddelde
        HR_max  (Eq. 8, Tanaka 2001): 208 − 0.7 × leeftijd
        HR_rust (Eq. 9 Lloyd 2022):   90.93 − 0.64 × VO2max_rel
        CO_max  (Eq. 10):             SV_max × HR_max / 1000
        CO_rust (Eq. 11):             SV_rust × HR_rust / 1000
        """
        v = self._vo2max_abs

        self.SV_max  = 40.59 + 24.81 * v         # mL/slag
        self.SV_rust = 85.1                        # mL/slag
        self.HR_max  = 208 - 0.7 * self.runner.age
        self.HR_rust = max(40.0, 90.93 - 0.64 * self.runner.vo2max)
        self.CO_max  = (self.SV_max  / 1000) * self.HR_max
        self.CO_rust = (self.SV_rust / 1000) * self.HR_rust

    # ── Omgevingscorrecties ────────────────────────────────────────────────────

    def _co_max_gecorrigeerd(self, t_core: float, dehydratie_pct: float) -> float:
        """
        CO_max gereduceerd door hittestress en dehydratie.

        Hitte (Lloyd 2022, Section 'Heat strain'):
            ~2% reductie per °C boven 38.0°C kerntemperatuur.
            Fysiologisch mechanisme: verminderde ventriculaire vulling
            door hoge huidbloedstroom (Frank-Starling).

        Dehydratie (Lloyd 2022, Section 'Dehydration'):
            ~1% reductie per % lichaamsgewichtsverlies boven 2%.
            Mechanisme: verminderd plasmavolume → lager slagvolume.
        """
        # Hitte
        dt_hitte = max(0.0, t_core - 38.0)
        factor_hitte = max(0.70, 1.0 - 0.02 * dt_hitte)

        # Dehydratie
        dt_dehy = max(0.0, dehydratie_pct - 2.0)
        factor_dehy = max(0.85, 1.0 - 0.01 * dt_dehy)

        return self.CO_max * factor_hitte * factor_dehy

    def _sv_gecorrigeerd(self, t_core: float, co_gevraagd: float) -> float:
        """
        Slagvolume gecorrigeerd voor hittestress.

        Gonzalez-Alonso et al. (2008) J Physiol 586:45-49:
        Bij hoge kerntemperatuur daalt SV door:
          (1) verhoogde hartfrequentie (kortere ventriculaire vultijd)
          (2) perifere vasodilatatie verlaagt ventriculaire vuldruk

        Empirische correctie: SV daalt ~3 mL/slag per °C boven 38°C.
        Range: 85% tot 100% van SV_max op basis van fitnessniveau.

        CO_gevraagd gebruikt als extra correctie: bij hoge CO-vraag
        compenseert HR deels het lagere SV.
        """
        # Basis SV geïnterpoleerd op fitnessniveau
        sv_basis = self.SV_rust + (self.SV_max - self.SV_rust) * min(
            1.0, co_gevraagd / max(0.1, self.CO_max)
        )

        # Hittecorrectie Gonzalez-Alonso
        dt = max(0.0, t_core - 38.0)
        sv_gecorr = sv_basis - 3.0 * dt

        return max(50.0, sv_gecorr)  # minimum fysiologisch SV: ~50 mL

    # ── Eén tijdstap ───────────────────────────────────────────────────────────

    def bereken_stap(
        self,
        jos3: JOS3Outputs,
        decompensatie_drempel: float = DECOMPENSATIE_RESERVE,
    ) -> CVRState:
        """
        Bereken cardiovasculaire toestand op basis van JOS-3 output.

        Eenheidconversies:
            cardiac_output [L/h]  → CO_gevraagd [L/min] : delen door 60
            weight_loss_g_s [g/s] → kg cumulatief       : × t_sec / 1000
              (weight_loss_g_s is al cumulatief in JOS-3 output)
        """

        # --- CO vraag rechtstreeks uit JOS-3 ---
        
        
        #co_gevraagd = jos3.cardiac_output / 60.0   # L/h → L/min
        
        # NA — Fick-principe correctie
        # Literatuur: Bassett & Howley 2000 (Med Sci Sports Exerc 32:70-84)
        #             Rowell 1986 (Human Circulation, Oxford UP)
        # a_v_diff = arteriovenoeus O2-verschil; stijgt lineair met VO2max
        # Factor 0.97 gekalibreerd op Leiden 2026 (N=5000):
            #   M: 3.7% decomp, ratio F/M = 0.58x, CO_reserve 39% CO_max
        FICK_FACTOR = 0.97
        co_jos3     = jos3.cardiac_output / 60          # L/h → L/min
        vo2_abs     = co_jos3 * 0.155                            # terugrekenen VO2_abs
        a_v_diff    = (0.100 + 0.0015 * self.runner.vo2max) * FICK_FACTOR
        co_gevraagd = vo2_abs / a_v_diff        
        
        

        # --- Dehydratie ---
        # weight_loss_by_evap_and_res is cumulatief in g/s × tijdstap
        # In HESTIA_Data_Engine wordt dit opgebouwd als lopende som;
        # hier verwachten we het cumulatieve verlies in kg
        dehydratie_pct = (jos3.weight_loss_g_s / self.runner.mass) * 100

        # --- CO_max gecorrigeerd ---
        co_max = self._co_max_gecorrigeerd(jos3.t_core_mean, dehydratie_pct)

        # --- Slagvolume ---
        sv = self._sv_gecorrigeerd(jos3.t_core_mean, co_gevraagd)

        # --- Hartslag (KERN van de module) ---
        # Uit Lloyd (2022) Eq. 2: HR = CO / SV
        # CO_gevraagd = werkelijke thermofysiologische vraag van JOS-3
        # SV = gecorrigeerd voor hitte en fitnessniveau
        hr = (co_gevraagd / (sv / 1000.0))

        # Begrenzen op fysiologisch maximum en minimum
        hr = np.clip(hr, self.HR_rust, self.HR_max)

        # --- Reserve en belasting ---
        co_reserve  = max(-4.0, co_max - co_gevraagd)
        cvs_index   = np.clip(co_gevraagd / co_max, 0.0, 1.0) if co_max > 0 else 1.0

        # --- Karvonen HR-reserve percentage ---
        # (HR - HR_rust) / (HR_max - HR_rust) × 100
        hr_reserve_pct = np.clip(
            (hr - self.HR_rust) / max(1.0, self.HR_max - self.HR_rust) * 100,
            0.0, 100.0
        )

        # --- AVA-status ---
        # bf_ava_hand + bf_ava_foot in L/h; beide < drempel = gesloten
        ava_totaal = jos3.bf_ava_hand + jos3.bf_ava_foot
        ava_open   = ava_totaal > self.AVA_SLUIT_DREMPEL

        # --- Decompensatie ---
        decompensatie = co_reserve < decompensatie_drempel

        return CVRState(
            t_min            = jos3.t_min,
            HR               = round(hr, 1),
            HR_max           = round(self.HR_max, 1),
            HR_reserve_pct   = round(hr_reserve_pct, 1),
            CO_gevraagd      = round(co_gevraagd, 2),
            CO_max           = round(co_max, 2),
            CO_reserve       = round(co_reserve, 2),
            SV               = round(sv, 1),
            CVS_index        = round(cvs_index, 3),
            decompensatie    = decompensatie,
            ava_open         = ava_open,
            dehydratie_pct   = round(dehydratie_pct, 2),
            t_core           = round(jos3.t_core_mean, 2),
            t_cb             = round(jos3.t_cb, 2),
            bf_skin_total_lh = round(jos3.bf_skin_total, 1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# HESTIA KOPPELINGSINTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def koppel_cvr_aan_jos3(
    runner:           RunnerProfile,
    jos3_output_lijst: List[JOS3Outputs],
    decompensatie_drempel: float = 2.0,
) -> CVRTimeSeries:
    """
    Verwerk een lijst JOS3Outputs tot een CVRTimeSeries.

    Gebruik in HESTIA_Data_Engine.py:

        from HESTIA_CVR_Module_v2 import (
            RunnerProfile, JOS3Outputs, koppel_cvr_aan_jos3
        )

        # Na de bestaande JOS-3 simulatielus:
        jos3_outputs = []
        gewichtsverlies_kg = 0.0
        for stap_idx, sim_stap in enumerate(simulatie_tijdstappen):
            jos3_model.simulate(times=1, dtime=60)
            r = jos3_model.dict_results()

            # Cumulatief gewichtsverlies opbouwen
            gewichtsverlies_kg += r['weight_loss_by_evap_and_res'][-1] * 60 / 1000

            jos3_outputs.append(JOS3Outputs(
                t_min            = stap_idx,
                cardiac_output   = r['cardiac_output'][-1],       # L/h
                t_core_mean      = r['t_core'][-1].mean(),         # °C
                t_cb             = r['t_cb'][-1],                  # °C
                weight_loss_g_s  = gewichtsverlies_kg,             # kg cumulatief
                bf_skin_total    = r['bf_skin'][-1].sum(),          # L/h
                bf_ava_hand      = r['bf_ava_hand'][-1],           # L/h
                bf_ava_foot      = r['bf_ava_foot'][-1],           # L/h
            ))

        cvr_profiel = RunnerProfile(
            mass=runner_weight, height=runner_height * 100,
            age=runner_age, sex=gender, vo2max=vo2max_sample
        )
        cvr_ts = koppel_cvr_aan_jos3(cvr_profiel, jos3_outputs)

        # Uitkomstmaten voor HESTIA-aggregatie:
        hr_max_run          = cvr_ts.max_HR()
        cvs_index_max       = cvr_ts.max_CVS_index()
        decompensatie_t     = cvr_ts.decompensatie_tijdstip()
        ava_sluiting_t      = cvr_ts.ava_sluiting_tijdstip()
    """
    model = CVRModel(runner)
    tijdreeks = CVRTimeSeries()
    for jos3_stap in jos3_output_lijst:
        state = model.bereken_stap(jos3_stap, decompensatie_drempel)
        tijdreeks.append(state)
    return tijdreeks


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: DtD 2024 vs 2025 met gesimuleerde JOS-3 tijdreeksen
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 72)
    print("HESTIA CVR Module v2 — Demonstratie DtD 2024 vs 2025")
    print("Koppeling via JOS-3 cardiac_output (L/h) als directe CO-input")
    print("=" * 72)

    # ── Drie representatieve lopers ───────────────────────────────────────────
    lopers = [
        RunnerProfile(mass=70, height=175, age=35, sex='male',   vo2max=50),
        RunnerProfile(mass=65, height=168, age=52, sex='male',   vo2max=38),
        RunnerProfile(mass=58, height=163, age=38, sex='female', vo2max=42),
    ]
    loper_labels = [
        'Man  35j  VO2=50',
        'Man  52j  VO2=38',
        'Vrouw 38j VO2=42',
    ]

    # ── Gesimuleerde JOS-3 tijdreeksen (100 min, stap 1 min) ─────────────────
    # cardiac_output: bij MET 10 stijgt CO van ~600 naar ~1200-1500 L/h
    # Afgeleid van Gonzalez-Alonso (2008): CO ~20-25 L/min = 1200-1500 L/h
    # 2024: hogere MRT → hogere BFsk → hogere CO-vraag én hogere T_core

    def maak_jos3_tijdreeks(
        co_start, co_eind,            # L/h
        t_core_start, t_core_eind,    # °C
        t_cb_start,   t_cb_eind,      # °C
        zweet_kg_h,                   # kg/h zweettempo
        bf_skin_start, bf_skin_eind,  # L/h totaal
        ava_nul_op_min=None,          # tijdstip AVA-sluiting (None = blijft open)
        n=100
    ):
        reeks = []
        for i in range(n):
            f = (i / n) ** 0.5   # lichte wortelstijging — fysiologisch realistisch
            co   = co_start   + f * (co_eind   - co_start)
            tc   = t_core_start + f * (t_core_eind - t_core_start)
            tcb  = t_cb_start   + f * (t_cb_eind   - t_cb_start)
            bfsk = bf_skin_start + f * (bf_skin_eind - bf_skin_start)
            gewicht_verlies_kg = zweet_kg_h * (i / 60)
            ava_h = 0.05 if (ava_nul_op_min and i >= ava_nul_op_min) else 1.2
            reeks.append(JOS3Outputs(
                t_min            = i,
                cardiac_output   = co,
                t_core_mean      = tc,
                t_cb             = tcb,
                weight_loss_g_s  = gewicht_verlies_kg,  # hier kg cumulatief
                bf_skin_total    = bfsk,
                bf_ava_hand      = ava_h / 2,
                bf_ava_foot      = ava_h / 2,
            ))
        return reeks

    # DtD 2024: hoge MRT (42°C), hoge CO-vraag, AVA sluit bij ~70 min
    # DtD 2025: lage MRT (34°C), lagere CO-vraag, AVA blijft open
    scenario = {
        '2024': {
            'co':        (800, 1480),
            't_core':    (37.0, 40.4),
            't_cb':      (36.8, 39.8),
            'zweet_kgh': 1.5,
            'bf_skin':   (80, 480),
            'ava_sluit': 70,
        },
        '2025': {
            'co':        (780, 1220),
            't_core':    (37.0, 39.5),
            't_cb':      (36.8, 39.1),
            'zweet_kgh': 1.1,
            'bf_skin':   (80, 340),
            'ava_sluit': None,
        },
    }

    # ── Uitvoer ───────────────────────────────────────────────────────────────
    for editie, p in scenario.items():
        print(f"\n{'─' * 72}")
        print(f"  DtD {editie}  |  MRT {'~42°C' if editie=='2024' else '~34°C'}  |  "
              f"T_lucht {'~24°C' if editie=='2024' else '~16°C'}")
        print(f"{'─' * 72}")

        hdr = (f"{'Loper':<18} | {'HR_max':>6} | {'HR_piek':>7} | {'HR%res':>6} | "
               f"{'CVS_max':>7} | {'CO_res_min':>10} | {'Decomp':>8} | "
               f"{'AVA_sl':>7} | {'Dehy%':>6}")
        print(hdr)
        print('-' * len(hdr))

        for loper, label in zip(lopers, loper_labels):
            reeks = maak_jos3_tijdreeks(
                co_start=p['co'][0], co_eind=p['co'][1],
                t_core_start=p['t_core'][0], t_core_eind=p['t_core'][1],
                t_cb_start=p['t_cb'][0], t_cb_eind=p['t_cb'][1],
                zweet_kg_h=p['zweet_kgh'],
                bf_skin_start=p['bf_skin'][0], bf_skin_eind=p['bf_skin'][1],
                ava_nul_op_min=p['ava_sluit'],
            )

            cvr = koppel_cvr_aan_jos3(loper, reeks)
            m = CVRModel(loper)

            decomp_t = cvr.decompensatie_tijdstip()
            ava_t    = cvr.ava_sluiting_tijdstip()
            eind     = cvr.eindstaat()

            decomp_str = f"{decomp_t:.0f}min" if decomp_t is not None else "—"
            ava_str    = f"{ava_t:.0f}min"    if ava_t    is not None else "—"

            print(
                f"{label:<18} | {m.HR_max:>6.0f} | {cvr.max_HR():>7.1f} | "
                f"{eind.HR_reserve_pct:>5.1f}% | {cvr.max_CVS_index():>6.1%} | "
                f"{cvr.min_CO_reserve():>9.1f}L | {decomp_str:>8} | "
                f"{ava_str:>7} | {eind.dehydratie_pct:>5.1f}%"
            )

    print(f"\n{'─' * 72}")
    print("KOLOMMEN:")
    print("  HR_max    : leeftijdsafh. maximale hartslag (Tanaka 2001)")
    print("  HR_piek   : geschatte piekHR tijdens run (Lloyd 2022 Eq.2: HR=CO/SV)")
    print("  HR%res    : % Karvonen HR-reserve benut op eindtijdstip")
    print("  CVS_max   : max cardiovasculaire belasting (CO_gevraagd/CO_max)")
    print("  CO_res_min: minimale CO-reserve (L/min) over gehele run")
    print("  Decomp    : tijdstip eerste CO_reserve < 2.0 L/min")
    print("  AVA_sl    : tijdstip AVA-sluiting (vroegst detecteerbaar signaal)")
    print("  Dehy%     : einddehydratie (% lichaamsgewichtsverlies)")
    print()
    print("BRONNEN HARTSLAGSCHATTING:")
    print("  SV_max  = 40.59 + 24.81 × VO2max_abs  (Lloyd 2022, Eq.6)")
    print("  HR_max  = 208 − 0.7 × leeftijd         (Tanaka 2001)")
    print("  HR      = CO_gevraagd / SV              (Lloyd 2022, Eq.2)")
    print("  SV corr = −3 mL/°C boven T_core 38°C   (Gonzalez-Alonso 2008)")
    print()
    print("INTEGRATIE IN HESTIA_Data_Engine.py: zie docstring koppel_cvr_aan_jos3()")
    print(f"{'=' * 72}")
