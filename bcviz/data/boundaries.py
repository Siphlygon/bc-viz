"""
This is a module to use the arcgis_rest infrastructure to obtain relevant boundary data for the geographies of interest
in Greater Manchester.
"""
import os
from pathlib import Path

import geopandas as gpd

from ..acquire import arcgis_rest as ag
from ..utils.logger import LoggingLevels, get_logger
from .geography import GeoHierarchy, GeoLevel

# This is the name of the FeatureService that provides the MSOA boundaries
MSOA_BOUNDARY_ID = "MSOA_2021_EW_BFE_V8_RUC"
LAD_BOUNDARY_ID = "Local_Authority_Districts_December_2021_UK_BFC_2022"

# Start logger
logger = get_logger("bcviz.data.boundaries", LoggingLevels.INFO)


def download_boundaries(geo_level: GeoLevel) -> gpd.GeoDataFrame:
    """
    This function downloads the boundaries for the given `geo_level` from the ArcGIS REST API and returns them as a
    GeoDataFrame.

    Parameters
    ----------
    geo_level : GeoLevel
        The level of geography to download boundaries for.
    
    Returns
    -------
    gpd.GeoDataFrame
        Boundaries for the given `geo_level` as a GeoDataFrame.
    """
    match geo_level:
        case GeoLevel.MSOA:
            url = ag.ons_layer_url(MSOA_BOUNDARY_ID)
        case GeoLevel.LAD:
            url = ag.ons_layer_url(LAD_BOUNDARY_ID)
        case _:
            raise ValueError(f"Invalid geo_level: {geo_level}. Must be one of {list(GeoLevel)}.")
    
    logger.info(f"Downloading {geo_level.name} boundaries from ArcGIS REST API, this will take a minute.")
    return ag.query_features(url)


def filter_to_gm(boundaries: gpd.GeoDataFrame, geo_level: GeoLevel) -> gpd.GeoDataFrame:
    """
    This function filters the boundaries (LAD or MSOA) to only include Greater Manchester, which is a large size
    reduction especially in the case of MSOA, and will therefore aid using the file for processing.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        The boundaries of geo_level to filter for GM.
    geo_level : GeoLevel
        The level of the boundaries to filter.

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe containing only the Greater Manchester boundaries.
    """
    match geo_level:
        case GeoLevel.MSOA:
            code_col = "MSOA21CD"
        case GeoLevel.LAD:
            code_col = "LAD21CD"
        case _:
            raise ValueError(f"Invalid geo_level: {geo_level}. Must be one of {list(GeoLevel)}.")

    logger.info(f"Filtering {geo_level.name} boundaries to Greater Manchester")
    geo_h = GeoHierarchy.from_ons_lookup()
    areas = geo_h.get_areas_in_gm(geo_level)
    area_codes = [area.code for area in areas]
    return boundaries[boundaries[code_col].isin(area_codes)]


def get_all_boundaries(msoa_path: str | Path = "data/gm_msoa_boundaries.gpkg",
                       lad_path: str | Path = "data/gm_lad_boundaries.gpkg",
                       gm_only: bool = True) -> list[gpd.GeoDataFrame]:
    """
    This function downloads all the boundaries for the geographies of interest and saves them to disk.

    Parameters
    ----------
    msoa_path : str | Path
        The path to save the MSOA boundaries to.
    lad_path : str | Path
        The path to save the LAD boundaries to.
    gm_only : bool, optional
        Whether to filter the boundaries to only include Greater Manchester, by default True
    
    Returns
    -------
    list[gpd.GeoDataFrame]
        The boundaries for each geography.
    """
    results = []
    for path, level in zip([msoa_path, lad_path], [GeoLevel.MSOA, GeoLevel.LAD]):
        if os.path.exists(path):
            logger.info(f"{path} already exists, skipping download")
            continue

        boundaries = download_boundaries(level)
        if gm_only: boundaries = filter_to_gm(boundaries, level)
        boundaries.to_file(path)
        results.append(boundaries)
    return results


if __name__ == "__main__":
    get_all_boundaries(
        msoa_path="data/gm_msoa_boundaries.gpkg",
        lad_path="data/gm_lad_boundaries.gpkg",
        gm_only=True
    )
