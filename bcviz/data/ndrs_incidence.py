"""
This is a module which uses the download machinery in ndrs.py to collect the appropriate data for breast cancer
incidence visualisations at the MSOA and LAD level, making necessary cleaning and corrections for smooth joining later
in the program.
"""
import os
from pathlib import Path

import pandas as pd

from ..acquire import ndrs
from ..utils.logger import LoggingLevels, get_logger
from .geography import GeoHierarchy, GeoLevel

# These are all the years acceptable for MSOA-level incidence. They are overlapping 5-year pools.
ALLOWED_YEARS_MSOA = [
    "2018-2022",
    "2017-2021",
    "2016-2020",
    "2015-2019",
    "2014-2018",
    "2013-2017"
]
# TODO: LAD section (LAUA) allows for 5-year pools identical to the above, 3 year pools, or single year data. I am
# TODO: looking to add support after being able to replicate initial visualisation

# NDRS's names for its two rate types, and the short names used in our column names
RATE_TYPES = {"Non-standardised": "crude", "Age-standardised": "asr"}

# Text NDRS puts where a number would be: "[u]" = suppressed for small numbers, "NA" = no value / no flag
MISSING_MARKERS = ["[u]", "NA"]

# Start a logger
logger = get_logger("bcviz.data.ndrs_incidence", LoggingLevels.INFO)

def download_all_bc_incidence(geo_level: GeoLevel, dest: str | Path, years: str) -> Path:
    """
    Downloads all breast cancer incidence data for the specified `geo_level` to `dest`, given the specific choice
    of `years`.

    Parameters
    ----------
    geo_level : GeoLevel
        The specific GeoLevel to download all breast cancer incidence data for, must be GeoLevel.MSOA or GeoLevel.LAD.
    dest : str | Path
        The path to save the CSV with the downloaded information to.
    years : str
        The year range to download for, must be allowed.

    Returns
    -------
    Path
        The path to the downloaded CSV; same as `dest`.
    """
    assert geo_level in [GeoLevel.MSOA, GeoLevel.LAD], "NDRS only has incidence data for MSOAs & LADs"

    if geo_level == GeoLevel.MSOA:
        assert years in ALLOWED_YEARS_MSOA, "Must choose a valid 5-year range for MSOAs."
    else: # TODO: LAD ALLOWS FOR DIFFERENT RATES OF POOLING
        assert years in ALLOWED_YEARS_MSOA, "Must choose a valid 5-year range for LADs."

    geo_h = GeoHierarchy.from_ons_lookup()
    areas = geo_h.get_areas_in_gm(geo_level)
    area_names = [area.name for area in areas]

    return ndrs.download_incidence(
        geography=geo_level,
        areas=area_names,
        dest=dest,
        site="Breast",
        sex="Females",
        years=years
    )


def clean_incidence(csv_path: str | Path, geo_level: GeoLevel) -> pd.DataFrame:
    """
    Cleans the downloaded incidence data.

    NDRS uses "[u]" to indicate suppressed values for small numbers, and "NA" to indicate no value / no flag; these
    need to be removed to keep numeric columns numeric. We also will rename columns here to match the convention in
    GeoLevel.name_col and GeoLevel.code_col.

    Designed to have pivot_age_rates called after it.

    Parameters
    ----------
    csv_path : str | Path
        The path to the downloaded incidence data.
    geo_level : GeoLevel
        The level of geography the data is for.

    Returns
    -------
    pd.DataFrame
        The cleaned incidence data, with columns renamed and intended-to-be numeric columns kept as numeric.
    """
    logger.info(f"Cleaning {geo_level.name} breast cancer incidence data...")
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    raw.columns = raw.columns.str.strip()  # e.g. "Geography name " has a trailing space
    raw = raw.rename(columns={
        "Geography code": geo_level.code_col,
        "Geography name": geo_level.name_col,
        "Count": "count",
        "Rate": "rate",
        "95% lower confidence interval": "lower_ci",
        "95% upper confidence interval": "upper_ci",
        "Flag": "flag",
    })

    # Transform the values of "Type of rate" into our short names for future pivoting
    raw["rate_type"] = raw["Type of rate"].map(RATE_TYPES)
    if raw["rate_type"].isna().any():
        raise ValueError(f"Unknown rate types: {set(raw.loc[raw['rate_type'].isna(), 'Type of rate'])}")
    
    # Known markers become missing values; any other text makes to_numeric raise, rather than silently becoming NaN
    for column in ["count", "rate", "lower_ci", "upper_ci"]:
        raw[column] = pd.to_numeric(raw[column].replace(MISSING_MARKERS, None))
    raw["count"] = raw["count"].astype("Int64")  # whole numbers that can be missing
    raw["flag"] = raw["flag"].replace("NA", None)
    return raw


def pivot_age_rates(cleaned_df: pd.DataFrame, geo_level: GeoLevel) -> pd.DataFrame:
    """
    Pivots the two different types of age rates into separate columns to restore one row per area.

    NDRS incidence contains both age-standardised and non-standardised rates for breast cancer incidence, meaning every
    row in the table is duplicated. We should instead pivot the table to have these be new columns.

    This will remove the "Type of rate" column, and dedicate three columns for each rate, and it's upper and lower
    confidence intervals. The count and other information is shared between the two rate types and belongs in a single
    row.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        The cleaned incidence data.
    geo_level : GeoLevel
        The level of geography the data is for.
    
    Returns
    -------
    pd.DataFrame
        The table with the age rates split into different columns.
    """
    logger.info(f"Pivoting {geo_level.name} breast cancer incidence data on different types of rates...")
    # Rate, CI and flag differ between the two rows of an area, so they become one column per rate type
    wide = cleaned_df.pivot(index=geo_level.code_col,  # the index for the new dataframe; 1 row per code
                            columns="rate_type",       # which column to use to pivot the dataframe
                            values=["rate", "lower_ci", "upper_ci", "flag"])
    wide.columns = [f"{rate_type}_{stat}" for stat, rate_type in wide.columns]  # e.g. ("rate", "asr") -> "asr_rate"

    # Pivoting numbers and text together leaves every column as generic objects, so restore the number types,
    # and group the columns by rate type
    wide = wide.infer_objects()[[f"{t}_{s}" for t in RATE_TYPES.values() for s in
                                 ("rate", "lower_ci", "upper_ci", "flag")]]

    # Name and count are the same in both rows of an area, so take them once and remove the duplications
    per_area = cleaned_df.drop_duplicates(geo_level.code_col
                                          ).set_index(geo_level.code_col)[[geo_level.name_col, "count"]]
    return per_area.join(wide).reset_index()


def get_bc_incidence(msoa_path: str | Path = "data/gm_msoa_bc_incidence.csv",
                     lad_path: str | Path = "data/gm_lad_bc_incidence.csv",
                     years: str = "2018-2022") -> list[pd.DataFrame]:
    """
    Run all the functions in this module, resulting in a final, cleaned, pivoted breast cancer incidence table for each
    geography.

    Parameters
    ----------
    msoa_path : str | Path
        The path to save the MSOA-level incidence table to.
    lad_path : str | Path
        The path to save the LAD-level incidence table to.
    years : str
        The 5-year pool to download incidence data for.
    
    Returns
    -------
    list[pd.DataFrame]
        The final, cleaned, pivoted breast cancer incidence tables for each geography.
    """
    results = []
    for path, level in zip([msoa_path, lad_path], [GeoLevel.MSOA, GeoLevel.LAD]):
        if os.path.exists(path):
            logger.info(f"{path} already exists, skipping download and cleaning")
            continue

        logger.info(f"Downloading {level.name} incidence data for {years} to {path}")
        download_all_bc_incidence(level, path, years)
        clean_df = clean_incidence(path, level)
        final_df = pivot_age_rates(clean_df, level)
        final_df.to_csv(path, index=False)
        results.append(final_df)
    return results


if __name__ == "__main__":
    get_bc_incidence(
        msoa_path = "data/gm_msoa_bc_incidence.csv",
        lad_path = "data/gm_lad_bc_incidence.csv",
        years = "2018-2022"
    )
