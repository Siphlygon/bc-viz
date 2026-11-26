"""
Query ArcGIS REST feature services, such as the ONS Open Geography Portal.

ONS publishes its boundaries and lookup tables as ArcGIS "feature services", which can be queried over plain HTTPS
without signing in:

    <services>/<service name>/FeatureServer/0/query?where=...&outFields=...&f=json

This module wraps that query endpoint, filtering and paging through results to return a complete table or set of
features. As a note, the ONS portal structure and response is like so:
- the root URL (ONS_SERVICES) is a folder holding ~3,850 geodatabases, which are called "services" in ArcGIS
- each service is a geodatabase downloadable from the normal portal, and contains either a layer (with geometry) or a
table (without geometry; e.g., a csv file)
- we call each row in a table or layer a "feature", which is a single record with attributes and possibly geometry

However, not every published product is a queryable service; some are just files with a different endpoint. The module
also provides functions to download those files, which are usually zipped CSVs. The ArcGIS Online portal is at
    https://www.arcgis.com/sharing/rest/content/items/<item id>/data

You can use the functions below to find things, since the portal documentation does not cover this:
    list_services("msoa")   service names containing "msoa"
    fields(url)             column names of a table or boundary layer
    count(url, where)       number of rows matching a filter
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from ..utils.logger import LoggingLevels, get_logger

# The API access point for services atthe ONS Open Geography Portal
ONS_SERVICES = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"

# Not every downloadable item is a queryable service, some are just files with a different endpoint
ARCGIS_ONLINE_ITEMS = "https://www.arcgis.com/sharing/rest/content/items"

# The integer relating to the British National Grid coordinate system, used by ONS for all its boundaries.
# See https://epsg.io/27700 for details.
BRITISH_NATIONAL_GRID = 27700

# Number of seconds to wait for a response from the server before giving up
TIMEOUT = 120

# Make a logger for this module
logger = get_logger("bcviz.acquire.arcgis_rest", LoggingLevels.INFO)


# --- FINDING SERVICES AND THEIR CONTENTS ---
def ons_layer_url(service: str, layer: int | None = None) -> str:
    """
    Get the URL of a layer (or table) in an ONS feature service, given the service name.

    Layer numbers are not always 0, so by default the service is asked for its contents and its only layer is used; if
    it has several, the error lists them.

    Parameters
    ----------
    service : str
        Name of the service, e.g. "OA_LSOA_MSOA_EW_DEC_2021_LU_v3"; corresponds to a dataset on the portal.
    layer : int | None
        Layer number, e.g. 0. If None, the service is queried for its contents and the only layer is used; if it has
        several, an error is raised.

    Returns
    -------
    str
        URL of the layer, which can be queried with `query_table` or `query_features`.
    
    Raises
    ------
    ValueError
        If the service has several layers and `layer` is None, or if the service does not exist.
    """
    service_url = f"{ONS_SERVICES}/{service}/FeatureServer"
    if layer is None:
        info = _request(service_url, {})
        contents = (info.get("layers") or []) + (info.get("tables") or [])
        if len(contents) != 1:
            listed = ", ".join(f"{c['id']}: {c['name']}" for c in contents)
            raise ValueError(f"{service} has {len(contents)} layers ({listed}); pass layer=<id>")
        layer = contents[0]["id"]
    return f"{service_url}/{layer}"


def list_services(contains: str | None = None, base: str = ONS_SERVICES) -> list[str]:
    """
    Names of the queryable services at `base`, optionally only those containing `contains` (case-insensitive).

    Only feature services are listed; the directory also contains map services, which draw maps but cannot be queried
    with this module.

    Parameters
    ----------
    contains : str | None
        If not None, only return services whose name contains this string (case-insensitive)
    base : str
        Base URL to query for services; defaults to the ONS Open Geography Portal
    
    Returns
    -------
    list[str]
        List of service names, sorted alphabetically
    """
    names = {s["name"] for s in _request(base, {})["services"] if s["type"] == "FeatureServer"}
    if contains:
        names = {n for n in names if contains.lower() in n.lower()}
    return sorted(names)


def fields(layer_url: str) -> list[str]:
    """
    Get the column names of a table or boundary layer.
    
    Parameters
    ----------
    layer_url : str
        URL of the layer or table to query
    
    Returns
    -------
    list[str]
        List of field names in the layer or table
    """
    return [f["name"] for f in _request(layer_url, {})["fields"]]


def count(layer_url: str, where: str = "1=1") -> int:
    """
    Count the number of rows in a layer matching `where`.

    Parameters
    ----------
    layer_url : str
        URL of the layer or table to query
    where : str
        SQL-style filter, e.g. "LAD22CD IN ('E08000001', 'E08000002')"
    
    Returns
    -------
    int
        Number of rows matching the filter
    """
    return _request(f"{layer_url}/query", {"where": where, "returnCountOnly": "true"})["count"]


def where_in(field: str, values: Iterable) -> str:
    """
    Build the filter `field IN ('a', 'b', ...)` for use as `where` in functions & requests.

    Parameters
    ----------
    field : str
        Name of the field to filter on
    values : Iterable
        Values to include in the filter; must be non-empty

    Returns
    -------
    str
        SQL-style filter string, e.g. "LAD22CD IN ('E08000001', 'E08000002')"
    
    Raises
    ------
    ValueError
        If `values` is empty
    """
    values = list(values)
    if not values:
        raise ValueError("where_in needs at least one value")
    quoted = ",".join("'" + str(v).replace("'", "''") + "'" for v in values)
    return f"{field} IN ({quoted})"


# --- PUBLIC QUERY FUNCTIONS ---
def query_table(
    layer_url: str,
    where: str = "1=1",
    out_fields: list[str] | None = None,
    ) -> pd.DataFrame:
    """
    Rows of a table (or a layer's attributes without geometry) matching `where`. Should only be used for tables or
    layers without geometry, as the geometry is not returned. Use `query_features` for layers with geometry.

    Parameters
    ----------
    layer_url : str
        URL of the layer or table to query
    where : str
        SQL-style filter, e.g. "LAD22CD IN ('E08000001', 'E08000002')"
    out_fields : list[str] | None
        List of fields to include in the output, or None to include all fields
    
    Returns
    -------
    pd.DataFrame
        DataFrame of rows matching the query, with columns in `out_fields` order (or the order returned by the server if
        `out_fields` is None)
    """
    logger.info(f"Querying {layer_url}")
    logger.info(f"for {where}")
    logger.info(f"with fields {out_fields or 'all'}")
    features = _query_all(layer_url, where, out_fields, {"f": "json", "returnGeometry": "false"})
    return pd.DataFrame([f["attributes"] for f in features], columns=out_fields)


def query_features(
    layer_url: str,
    where: str = "1=1",
    out_fields: list[str] | None = None,
    out_sr: int = BRITISH_NATIONAL_GRID,
    page_size: int | None = None,
    ) -> gpd.GeoDataFrame:
    """
    Features (attributes and geometry) matching `where`, in coordinate system `out_sr`. Should only be used for layers
    with geometry, as the geometry is returned. Use `query_table` for tables or layers without geometry.

    The server converts coordinates before sending them, so the default returns British National Grid regardless of how
    the data is stored. Lower `page_size` if requests for detailed boundaries time out.

    Parameters
    ----------
    layer_url : str
        URL of the layer or table to query
    where : str
        SQL-style filter, e.g. "LAD22CD IN ('E08000001', 'E08000002')"
    out_fields : list[str] | None
        List of fields to include in the output, or None to include all fields
    out_sr : int
        EPSG code of the coordinate system to return, e.g. 27700 for British National Grid, 4326 for WGS84 (lat/lon)
    page_size : int | None
        Number of features to fetch per page, by default None, which uses the server's default (usually 1,000-2,000).
        Lower this if requests for detailed boundaries time out.
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of features matching the query, with columns in `out_fields` order (or the order returned by the
        server if `out_fields` is None), and geometry in `out_sr` coordinate system
    """
    logger.info(f"Querying {layer_url}")
    logger.info(f"for {where}")
    logger.info(f"with fields {out_fields or 'all'}")
    features = _query_all(layer_url, where, out_fields,
                          {"f": "geojson", "outSR": out_sr}, page_size)
    crs = f"EPSG:{out_sr}"
    if not features:
        return gpd.GeoDataFrame(columns=[*(out_fields or []), "geometry"], geometry="geometry", crs=crs)
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    return gdf[[*out_fields, "geometry"]] if out_fields else gdf


# --- FILE ITEMS ---
def item_info(item_id: str) -> dict:
    """
    Catalogue entry for a portal item: 'title', 'type' (e.g. 'CSV Collection'), 'name' (original filename),
    'size' (bytes), 'modified' (ms since 1970), and 'url' for items that are services.
    
    Parameters
    ----------
    item_id : str
        ArcGIS Online item ID, e.g. "f3e0c1d8f3e04c8a1b0f3e0c1d8f3e0c"
    
    Returns
    -------
    dict
        Dictionary with item metadata, including 'title', 'type', 'name', 'size', 'modified', and 'url' for items that
        are services
    """
    return _request(f"{ARCGIS_ONLINE_ITEMS}/{item_id}", {})


def download_item(item_id: str, dest: str | Path, expected_title: str | None = None) -> Path:
    """
    Download a file item (e.g. a zipped CSV) to `dest`, unchanged, and return the path.

    With `expected_title`, the item's title must match exactly: a check that a pinned ID still refers to the dataset you
    intend. The file is written to '<dest>.partial' and only renamed once its size matches the catalogue's.

    Parameters
    ----------
    item_id : str
        ArcGIS Online item ID, e.g. "f3e0c1d8f3e04c8a1b0f3e0c1d8f3e0c"
    dest : str | Path
        Path to save the downloaded file, unchanged. You should use the ".zip" extension.
    expected_title : str | None
        If not None, the item's title must match exactly; otherwise a ValueError is raised
    """
    info = item_info(item_id)
    if expected_title is not None and info["title"] != expected_title:
        raise ValueError(f"Item {item_id} is titled {info['title']!r}, expected {expected_title!r}")
    if info.get("url"):
        raise ValueError(f"Item {item_id} is a {info['type']}, not a file; query it at {info['url']}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    # Streamed in 1 MB pieces, so large files never sit in memory whole.
    with requests.get(f"{ARCGIS_ONLINE_ITEMS}/{item_id}/data", stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            f.writelines(response.iter_content(chunk_size=1 << 20))
    if info.get("size") and partial.stat().st_size != info["size"]:
        raise RuntimeError(f"Downloaded {partial.stat().st_size} bytes of item {item_id}, "
                           f"expected {info['size']} (partial file kept at {partial})")
    partial.replace(dest)
    return dest


# --- INTERNALS ---
def _query_all(
    layer_url: str,
    where: str,
    out_fields: list[str] | None,
    output: dict,
    page_size: int | None = None,
    ) -> list[dict]:
    """
    Query the server to retrieve all rows of a table or layer matching `where`, paging through results if necessary.

    Parameters
    ----------
    layer_url : str
        URL of the layer or table to query
    where : str
        SQL-style filter, e.g. "LAD22CD IN ('E08000001', 'E08000002')"
    out_fields : list[str] | None
        List of fields to include in the output, or None to include all fields
    output : dict
        Additional parameters for the API request
    page_size : int | None, optional
        Number of features to fetch per page, by default None, which uses the server's default (usually 1,000-2,000).

    Returns
    -------
    list[dict]
        List of features matching the query

    Raises
    ------
    RuntimeError
        If the number of features returned does not match the expected count
    """
    info = _request(layer_url, {})
    expected = count(layer_url, where)
    params = {
        "where": where,
        "outFields": ",".join(out_fields) if out_fields else "*",
        # Paging is only reliable with a fixed sort order; the object id is unique and always present.
        "orderByFields": info["objectIdField"],
        "resultRecordCount": page_size or info.get("maxRecordCount") or 1000,
        **output,
    }
    features: list[dict] = []
    while len(features) < expected:
        page = _request(f"{layer_url}/query", {**params, "resultOffset": len(features)})["features"]
        if not page:  # No more results, even though we expected more; happens if server is busy
            break
        features += page
    if len(features) != expected:
        raise RuntimeError(f"Expected {expected} rows from {layer_url}, received {len(features)}")
    return features


def _request(url: str, params: dict) -> dict:
    """
    POST a request and return the JSON body.

    POST rather than GET so long filters (e.g. hundreds of codes) cannot exceed URL length limits. ArcGIS reports errors
    inside a normal-looking response (HTTP 200), so the body is checked too.

    Parameters
    ----------
    url : str
        URL to POST to
    params : dict
        Parameters to include in the POST body
    
    Returns
    -------
    dict
        JSON-decoded response body
    """
    response = requests.post(url, data={"f": "json", **params}, timeout=TIMEOUT)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        error = body["error"]
        details = "; ".join(error.get("details") or []) or error.get("message", "")
        raise RuntimeError(f"ArcGIS error from {url}: {details}")
    return body


if __name__ == "__main__":
    from . import arcgis_rest as ag

    GM = ["E08000001", "E08000002", "E08000003", "E08000004", "E08000005",
          "E08000006", "E08000007", "E08000008", "E08000009", "E08000010"]

    lookup = ag.query_table(
        ag.ons_layer_url("OA_LSOA_MSOA_EW_DEC_2021_LU_v3"),
        where=ag.where_in("LAD22CD", GM),
        out_fields=["OA21CD", "LSOA21CD", "MSOA21CD", "MSOA21NM", "LAD22CD", "LAD22NM"],
    )
    msoa = ag.query_features(
        ag.ons_layer_url("MSOA_2021_EW_BFE_V8_RUC"),
        where=ag.where_in("MSOA21CD", lookup["MSOA21CD"].unique()),
        out_fields=["MSOA21CD", "MSOA21NM"],
    )

    print(lookup.head())
    print(msoa.head())
