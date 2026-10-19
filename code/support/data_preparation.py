"""
Prepare the California and West Germany data for the computations.
"""

import csv

import numpy as np


CA_PREDICTORS = [
    "Log per-capita state income",
    "Share of population aged 15-24",
    "Retail cigarette price",
    "Per-capita beer consumption",
    "Cigarette sales per capita (1988)",
    "Cigarette sales per capita (1980)",
    "Cigarette sales per capita (1975)",
]

GER_PREDICTORS = [
    "Per-capita GDP (1971-80)",
    "Trade openness (1971-80)",
    "Inflation rate (1971-80)",
    "Industry share (1971-80)",
    "Schooling (1970, 1975)",
    "Investment rate (1970s)",
]

CA_VALIDATION_YEARS = list(range(1970, 1989))
GER_VALIDATION_YEARS = list(range(1981, 1991))

# Section 6 excludes the lagged-outcome predictor years from validation.
CA_PREDICTOR_YEARS = {1975, 1980, 1988}
CA_HOLDOUT_IDX = [
    i for i, year in enumerate(CA_VALIDATION_YEARS)
    if year not in CA_PREDICTOR_YEARS
]


def _load_panel(path, unit_column, variables, *, year_parser=int):
    panel = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            unit, year = row[unit_column], year_parser(row["year"])
            panel[(unit, year)] = {
                k: (float(row[k]) if row[k] != "" else np.nan)
                for k in variables
            }
    units = sorted({unit for (unit, _) in panel})
    return panel, units


def _mean_over(panel, unit, variable, years, *, skip_missing_rows=False):
    vals = [
        panel[(unit, year)][variable] for year in years
        if not skip_missing_rows or (unit, year) in panel
    ]
    vals = [value for value in vals if not np.isnan(value)]
    return float(np.mean(vals))


def _california_predictors(panel, state):
    def mean_over(variable, years):
        return _mean_over(panel, state, variable, years)

    return np.array([
        mean_over("lnincome", range(1980, 1989)),
        mean_over("age15to24", range(1980, 1989)),
        mean_over("retprice", range(1980, 1989)),
        mean_over("beer", range(1984, 1989)),
        panel[(state, 1988)]["cigsale"],
        panel[(state, 1980)]["cigsale"],
        panel[(state, 1975)]["cigsale"],
    ])


def _germany_predictors(panel, country):
    def mean_over(variable, years):
        return _mean_over(panel, country, variable, years, skip_missing_rows=True)

    training_years = range(1971, 1981)
    return np.array([
        mean_over("gdp", training_years),
        mean_over("trade", training_years),
        mean_over("infrate", training_years),
        mean_over("industry", training_years),
        mean_over("schooling", [1970, 1975]),
        mean_over("invest70", [1980]),
    ])


def _build_matrices(panel, units, *, treated, predictor_vector, outcome,
                    validation_years, outcome_years):
    donors = [unit for unit in units if unit != treated]
    X1_raw = predictor_vector(panel, treated)
    X0_raw = np.column_stack([predictor_vector(panel, unit) for unit in donors])

    allX = np.column_stack([X1_raw, X0_raw])
    sd = allX.std(axis=1, ddof=1)
    X1 = X1_raw / sd
    X0 = X0_raw / sd[:, None]

    Z1 = np.array([panel[(treated, year)][outcome] for year in validation_years])
    Z0 = np.column_stack([
        [panel[(unit, year)][outcome] for year in validation_years]
        for unit in donors
    ])
    outcomes = {
        unit: np.array([panel[(unit, year)][outcome] for year in outcome_years])
        for unit in units
    }
    return dict(X1=X1, X0=X0, Z1=Z1, Z0=Z0, donors=donors, sd=sd,
                X1_raw=X1_raw, X0_raw=X0_raw, outcomes=outcomes,
                years=list(outcome_years))


def build_california(path):
    """
    Build Prop 99 data with seven predictors and validation in 1970-1988.

    Returns scaled/raw predictors, validation outcomes, donor names, scales,
    and full outcome series for 1970-2000. 
    
    Missing predictor values are omitted from averages; 
    missing unit-year rows raise KeyError.
    """
    panel, states = _load_panel(
        path, "state", ("cigsale", "lnincome", "beer", "age15to24", "retprice"),
        year_parser=lambda year: int(float(year)),
    )
    return _build_matrices(
        panel, states, treated="California",
        predictor_vector=_california_predictors, outcome="cigsale",
        validation_years=CA_VALIDATION_YEARS, outcome_years=range(1970, 2001),
    )


def build_germany(path):
    """
    Build West Germany data with six predictors and validation in 1981-1990.

    Returns scaled/raw predictors, validation outcomes, donor names, scales,
    and full outcome series for 1960-2003. 
    
    Predictor averages omit missing values and unit-year rows.
    """
    panel, countries = _load_panel(
        path, "country", ("gdp", "infrate", "trade", "schooling",
                          "invest60", "invest70", "invest80", "industry"),
    )
    return _build_matrices(
        panel, countries, treated="West Germany",
        predictor_vector=_germany_predictors, outcome="gdp",
        validation_years=GER_VALIDATION_YEARS, outcome_years=range(1960, 2004),
    )
