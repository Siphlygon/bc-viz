"""
Join different geographical datasets together.

Currently designed to join attribute data (e.g., NDRS breast cancer incidence) to boundaries (e.g. MSOAs) but may be
expanded to join other attribute data to other attribute data.
"""
import re
from typing import overload

import geopandas as gpd
import pandas as pd

from ..utils.logger import LoggingLevels, get_logger
from .geography import GeoLevel

logger = get_logger("bcviz.data.join", LoggingLevels.INFO)

@overload
def standardise_columns(table: gpd.GeoDataFrame) -> gpd.GeoDataFrame: ...
@overload
def standardise_columns(table: pd.DataFrame) -> pd.DataFrame: ...
def standardise_columns(table):
    """
    Rename ONS-style code/name columns (e.g. MSOA21CD, ladnm) to the project's names (msoa_code, lad_name).

    The same rename `from_ons_lookup` does, pulled out so boundaries and lookups share it.

    Parameters
    ----------
    table : pd.DataFrame | gpd.GeoDataFrame
        The table to rename columns in.
    
    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        The table with renamed columns
    """
    rename = {}
    for column in table.columns:
        match = re.fullmatch(r"(OA|LSOA|MSOA|LAD)\d*(CD|NM)", column, flags=re.IGNORECASE)
        if match:
            level = GeoLevel[match.group(1).upper()]
            rename[column] = level.code_col if match.group(2).upper() == "CD" else level.name_col
    return table.rename(columns=rename)


def join_to_boundaries(boundaries: gpd.GeoDataFrame, data: pd.DataFrame, level: GeoLevel) -> gpd.GeoDataFrame:
    """
    Attach area-level data to boundaries; every polygon must find exactly one row of data, and vice versa.

    A polygon without data would look identical on a map to a suppressed area, so any mismatch raises instead.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        The boundaries to join to.
    data : pd.DataFrame
        The data to join to the boundaries.
    level : GeoLevel
        The level of geography to join.
    
    Returns
    -------
    gpd.GeoDataFrame
        The boundaries with the data attached.
    """
    # The data frames are joined based on the code column
    code = level.code_col
    # Names come from the boundaries; the data's copy would only produce clashing name_x / name_y columns
    data = data.drop(columns=[level.name_col], errors="ignore")

    logger.info(f"Joining {level.name} data to boundaries...")
    joined = boundaries.merge(data, on=code, how="outer", validate="one_to_one", indicator=True)
    no_data = joined.loc[joined["_merge"] == "left_only", code].tolist()
    no_polygon = joined.loc[joined["_merge"] == "right_only", code].tolist()
    if no_data or no_polygon:
        raise ValueError(f"{level.name} codes don't match: {len(no_data)} polygons without data (e.g. {no_data[:3]}), "
                         f"{len(no_polygon)} data rows without a polygon (e.g. {no_polygon[:3]}). "
                         "Check both use the same boundary year.")
    return joined.drop(columns="_merge")


if __name__ == "__main__":
    # Example usage
    from . import boundaries, ndrs_incidence

    geo_boundaries = boundaries.get_all_boundaries()
    geo_boundaries = [standardise_columns(boundary) for boundary in geo_boundaries]

    bc_incidences = ndrs_incidence.get_bc_incidence()
    for geo_boundary, bc_incidence, level in zip(geo_boundaries, bc_incidences, [GeoLevel.MSOA, GeoLevel.LAD]):
        joined = join_to_boundaries(geo_boundary, bc_incidence, level)
        joined.to_file(f"data/gm_{level.name.lower()}_bc_incidence_joined.gpkg")
