"""
Calculate synthetic outcomes and treatment effects using full-precision weights.
"""

import numpy as np


def synthetic_path(data, weights):
    """
    Return the donor-weighted outcome series in the data's year order.

    ``data`` provides the ``donors`` list and ``outcomes`` mapping returned by
    the data builders. ``weights`` maps donor names to full-precision weights. 
    Omitted donors receive zero weight; a name outside the donor list raises ValueError.
    """
    donors = data["donors"]
    w = np.zeros(len(donors))
    for donor, weight in weights.items():
        w[donors.index(donor)] = weight
    return sum(w[i] * data["outcomes"][donor] for i, donor in enumerate(donors))


def effect_path(treated, synthetic):
    """
    Return treated minus synthetic outcomes at each position in the series.

    The two series must be aligned to the same years.
    """
    return np.asarray(treated) - np.asarray(synthetic)


def mean_effect(years, treated, synthetic, selected_years):
    """
    Return the mean effect over the explicitly selected years.
    """
    effects = effect_path(treated, synthetic)
    by_year = {year: float(effects[i]) for i, year in enumerate(years)}
    return float(np.mean([by_year[year] for year in selected_years]))
