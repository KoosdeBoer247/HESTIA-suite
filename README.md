# HESTIA-suite
HESTIA — Heat Exposure Stress &amp; Thermophysiology Integrated Assessment
HESTIA is an advanced thermophysiological simulation framework designed for population-based heat stress and exertional heat illness (EHI/EHS) risk assessment during endurance events such as marathons, road races, walking events, and other mass participation activities.

The system combines detailed human thermoregulation modelling, cardiovascular response simulation, behavioural pacing adaptation, and Monte Carlo population analysis into a single integrated platform.

HESTIA was developed to support operational heat risk assessment for event organisers, medical services, and public safety authorities.

Core Features
Advanced Human Thermoregulation
Based on the JOS-3 thermoregulation model
(Takahashi et al., 2021)
Multi-segment human heat balance simulation
Dynamic core temperature evolution
Skin blood flow and sweating physiology
Environmental heat exchange modelling
Cardiovascular Response (CVR) Module
Cardiac output modelling based on:
Lloyd et al. (2022)
Rowell (1986)
Gonzalez-Alonso et al. (2008)

Heart rate estimation via:

CO=SV×HR

Dynamic stroke volume reduction under heat strain
Cardiovascular reserve estimation
Decompensation detection logic
Monte Carlo Population Simulation
Population-scale simulation (default: 5000 virtual participants)
Physiologically constrained sampling architecture
Individual variability in:
VO₂max
age
sex
body composition
sweat response
hydration behaviour
pacing behaviour
NSAID usage
heat acclimatisation
Behavioural & Perceptual Modelling
Dynamic pacing reduction under thermal stress
Interoceptive variability modelling
Individual pacing sensitivity (Kp)
Motivational factors
Voluntary dehydration behaviour
Ad-libitum drinking simulation
Environmental Modelling
Open-Meteo weather integration
Solar radiation modelling via pvlib
Urban Heat Island (UHI) correction
UTCI and WBGT support
Time-resolved meteorological forcing
Clinical Heat Illness Metrics
Exertional Heat Stroke (EHS) risk estimation

Clinical thermal dose estimation:

AUC
clinical
	​

=∫max(0,T
rect
	​

−40.5)dt

Post-finish thermal escalation simulation
Cardiovascular collapse risk modelling
Multiple operational endpoints:
EHS
hospitalisation
first-aid contact
Scientific Foundation

HESTIA integrates concepts and methodologies from:

Thermophysiology
Exercise physiology
Cardiovascular modelling
Environmental heat stress research
Human performance modelling
Mass-event medical risk assessment

Primary scientific references include:

Takahashi et al. (2021) — JOS-3 thermoregulation model
Lloyd et al. (2022) — cardiovascular response modelling
Gonzalez-Alonso et al. (2008) — heat-induced cardiovascular strain
Breslow et al. (2021) — Boston Marathon EHS incidence
Roberts (2007) — clinical thermal dose thresholds
Vanos et al. (2023) — survivability/liveability heat limits
Simulation Outputs

HESTIA can generate:

Rectal/core temperature trajectories
Cardiovascular reserve curves
Heart rate distributions
Population percentile envelopes
Vulnerable cohort analysis
Collapse probability estimates
Clinical threshold exceedance statistics
Post-finish heat illness risk
Monte Carlo uncertainty distributions
Example Applications
Marathon heat risk assessment
Medical planning for endurance events
Climate adaptation studies
Population vulnerability analysis
Operational decision support
Heatwave preparedness
Sports science research
Public health risk estimation
Current Development Status

HESTIA is an actively evolving research and operational prototype.

Current focus areas include:

calibration refinement
clinical validation
pacing behaviour modelling
uncertainty quantification
vulnerable subgroup identification
Streamlit-based operational dashboards
Technical Stack
Python
JOS-3 / pythermalcomfort
NumPy
Pandas
Matplotlib
pvlib
Open-Meteo API
Monte Carlo simulation methods
Important Disclaimer

HESTIA is currently intended for:

research
exploratory analysis
operational support
scientific development

It is not a certified medical device and should not be used as a standalone clinical diagnostic system.

All outputs should be interpreted within the context of expert physiological and medical judgement.

Author

Developed within the HESTIA project by
Koos de Boer — retired Royal Netherlands Navy maintenance engineer, specialised in high-performance technical systems, with a strong focus on thermophysiology, heat stress modelling, and population-based risk analysis.
