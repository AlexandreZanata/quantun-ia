"""Fixed agro-climate validation cases — Brazilian soybean scenarios for human-facing tests."""

from __future__ import annotations

from dataclasses import dataclass

from src.application.human_agro_scorer import AgroMunicipalityProfile, WeatherBlockStats

# Expected rank 1 = lowest P(low yield), 8 = highest (see exp_078 hypothesis.md)


@dataclass(frozen=True)
class AgroValidationCase:
    case_id: str
    title: str
    expected_tier: str
    expected_rank: int
    science_note: str
    profile: AgroMunicipalityProfile


def _wb(**kwargs: float) -> WeatherBlockStats:
    return WeatherBlockStats(**kwargs)


AGRO_VALIDATION_CASES: tuple[AgroValidationCase, ...] = (
    AgroValidationCase(
        case_id="L01",
        title="Favorable season — Lucas do Rio Verde (MT)",
        expected_tier="very_low",
        expected_rank=1,
        science_note=(
            "Center-west benchmark municipality with adequate in-season rain and moderate heat — "
            "typical good soybean window in MATOPIBA expansion belt."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Lucas do Rio Verde",
            state="MT",
            crop_year=2020,
            latitude=-13.06,
            longitude=-55.91,
            area_harvested_ha=25_000.0,
            precipitation=_wb(mean=5.5, std=1.0, min_v=3.0, max_v=9.0),
            t2m_max=_wb(mean=301.0, std=1.5, min_v=298.0, max_v=304.0),
            ndvi=_wb(mean=3.8, std=0.3, min_v=3.0, max_v=4.5),
        ),
    ),
    AgroValidationCase(
        case_id="L02",
        title="Normal pampa season — Passo Fundo (RS)",
        expected_tier="very_low",
        expected_rank=2,
        science_note=(
            "Southern Brazil row-crop municipality with balanced precipitation — "
            "lower frost/heat extremes than cerrado."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Passo Fundo",
            state="RS",
            crop_year=2019,
            latitude=-28.26,
            longitude=-52.41,
            area_harvested_ha=12_000.0,
            precipitation=_wb(mean=4.8, std=1.1, min_v=2.5, max_v=8.0),
            ndvi=_wb(mean=3.5, std=0.4, min_v=2.8, max_v=4.2),
        ),
    ),
    AgroValidationCase(
        case_id="L03",
        title="Paraná soybean belt — Londrina (PR)",
        expected_tier="low",
        expected_rank=3,
        science_note=(
            "Traditional high-productivity PR panel — mild stress but historically stable yields."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Londrina",
            state="PR",
            crop_year=2018,
            latitude=-23.30,
            longitude=-51.17,
            area_harvested_ha=18_000.0,
            precipitation=_wb(mean=4.0, std=1.0, min_v=2.0, max_v=7.0),
            ndvi=_wb(mean=3.2, std=0.5, min_v=2.5, max_v=4.0),
        ),
    ),
    AgroValidationCase(
        case_id="L04",
        title="GO frontier — Rio Verde with minor dry spells",
        expected_tier="low",
        expected_rank=4,
        science_note=(
            "Cerrado municipality with occasional mid-season dryness but still above critical thresholds."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Rio Verde",
            state="GO",
            crop_year=2017,
            latitude=-17.79,
            longitude=-50.93,
            area_harvested_ha=22_000.0,
            precipitation=_wb(mean=3.2, std=0.9, min_v=1.5, max_v=6.0),
            t2m_max=_wb(mean=302.0, std=2.0, min_v=297.0, max_v=306.0),
        ),
    ),
    AgroValidationCase(
        case_id="H01",
        title="La Niña drought — Barreiras (BA)",
        expected_tier="high",
        expected_rank=5,
        science_note=(
            "Western BA cerrado drought scenario — depressed weekly precipitation during pod fill."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Barreiras",
            state="BA",
            crop_year=2019,
            latitude=-12.15,
            longitude=-45.00,
            area_harvested_ha=8_000.0,
            precipitation=_wb(mean=1.2, std=0.4, min_v=0.5, max_v=2.5),
            t2m_max=_wb(mean=304.0, std=2.5, min_v=300.0, max_v=310.0),
            ndvi=_wb(mean=2.0, std=0.5, min_v=1.2, max_v=3.0),
        ),
    ),
    AgroValidationCase(
        case_id="H02",
        title="Heat wave — Uberaba (MG)",
        expected_tier="high",
        expected_rank=6,
        science_note=(
            "Triângulo Mineiro heat + VPD stress — reproductive-stage Tmax above 38 °C peaks."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Uberaba",
            state="MG",
            crop_year=2020,
            latitude=-19.75,
            longitude=-47.94,
            area_harvested_ha=15_000.0,
            precipitation=_wb(mean=1.8, std=0.5, min_v=0.6, max_v=3.5),
            t2m_max=_wb(mean=307.0, std=2.0, min_v=303.0, max_v=313.0),
            vapor_pressure_deficit=_wb(mean=19.0, std=4.0, min_v=13.0, max_v=28.0),
        ),
    ),
    AgroValidationCase(
        case_id="H03",
        title="Compound drought + heat — Bom Jesus (PI)",
        expected_tier="very_high",
        expected_rank=7,
        science_note=(
            "Piauí marginal expansion area with combined rainfall deficit and canopy stress (low NDVI)."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Bom Jesus",
            state="PI",
            crop_year=2019,
            latitude=-9.07,
            longitude=-44.36,
            area_harvested_ha=6_000.0,
            precipitation=_wb(mean=0.9, std=0.3, min_v=0.3, max_v=1.8),
            t2m_max=_wb(mean=307.0, std=2.5, min_v=303.0, max_v=314.0),
            ndvi=_wb(mean=1.8, std=0.4, min_v=1.0, max_v=2.8),
        ),
    ),
    AgroValidationCase(
        case_id="H04",
        title="Severe MATOPIBA stress — Balsas (MA)",
        expected_tier="very_high",
        expected_rank=8,
        science_note=(
            "Maranhão southern frontier — extreme dry season, high VPD, lowest vegetation index in panel."
        ),
        profile=AgroMunicipalityProfile(
            municipality="Balsas",
            state="MA",
            crop_year=2020,
            latitude=-7.53,
            longitude=-46.04,
            area_harvested_ha=5_000.0,
            precipitation=_wb(mean=0.7, std=0.2, min_v=0.2, max_v=1.5),
            t2m_max=_wb(mean=308.0, std=2.0, min_v=304.0, max_v=315.0),
            ndvi=_wb(mean=1.5, std=0.3, min_v=0.8, max_v=2.5),
            vapor_pressure_deficit=_wb(mean=20.0, std=5.0, min_v=14.0, max_v=30.0),
        ),
    ),
)


def low_risk_cases() -> list[AgroValidationCase]:
    return [c for c in AGRO_VALIDATION_CASES if c.case_id.startswith("L")]


def high_risk_cases() -> list[AgroValidationCase]:
    return [c for c in AGRO_VALIDATION_CASES if c.case_id.startswith("H")]


def case_by_id(case_id: str) -> AgroValidationCase:
    for case in AGRO_VALIDATION_CASES:
        if case.case_id == case_id:
            return case
    msg = f"unknown case_id: {case_id}"
    raise KeyError(msg)
