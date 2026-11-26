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


def get_msoa_boundaries() -> gpd.GeoDataFrame:
    """
    This function downloads the MSOA boundaries from the ArcGIS REST API and returns them as a GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        MSOA boundaries for Greater Manchester as a GeoDataFrame.
    """
    logger.info("Downloading MSOA boundaries from ArcGIS REST API, this will take a minute (~250 MB).")
    msoa_url = ag.ons_layer_url(MSOA_BOUNDARY_ID)
    return ag.query_features(msoa_url)


def get_lad_boundaries() -> gpd.GeoDataFrame:
    """
    This function downloads the LAD boundaries from the ArcGIS REST API and returns them as a GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        LAD boundaries for Greater Manchester as a GeoDataFrame.
    """
    logger.info("Downloading LAD boundaries from ArcGIS REST API (~50 MB).")
    lad_url = ag.ons_layer_url(LAD_BOUNDARY_ID)
    return ag.query_features(lad_url)


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
    logger.info(f"Filtering {geo_level.name} boundaries to Greater Manchester")
    geo_h = GeoHierarchy.from_ons_lookup()
    areas = geo_h.get_areas_in_gm(geo_level)
    area_codes = [area.code for area in areas]
    code_col = "MSOA21CD" if geo_level == GeoLevel.MSOA else "LAD21CD"
    return boundaries[boundaries[code_col].isin(area_codes)]


def download_all_boundaries(msoa_path: str | Path,
                            lad_path: str | Path,
                            gm_only: bool = True) -> None:
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
    """
    for path, level in zip([msoa_path, lad_path], [GeoLevel.MSOA, GeoLevel.LAD]):
        if not os.path.exists(path):
            boundaries = get_msoa_boundaries() if level == GeoLevel.MSOA else get_lad_boundaries()
            if gm_only: boundaries = filter_to_gm(boundaries, level)
            boundaries.to_file(path)


if __name__ == "__main__":
    download_all_boundaries(
        msoa_path="data/gm_msoa_boundaries.gpkg",
        lad_path="data/gm_lad_boundaries.gpkg",
        gm_only=True
    )
