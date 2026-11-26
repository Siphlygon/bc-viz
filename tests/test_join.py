import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from bcviz.data.geography import GeoLevel
from bcviz.data.join import join_to_boundaries, standardise_columns


def msoa_boundaries(codes: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"msoa_code": codes, "msoa_name": [f"Boundary {c}" for c in codes]},
        geometry=[Point(i, i) for i in range(len(codes))],
        crs="EPSG:27700",
    )


def msoa_data(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "msoa_code": codes,
        "msoa_name": [f"Data {c}" for c in codes],
        "count": range(len(codes)),
    })


# --- standardise_columns ---
def test_standardise_columns_renames_ons_columns():
    """Test that standardise_columns renames ONS columns to the correct names."""
    table = pd.DataFrame(columns=["MSOA21CD", "MSOA21NM", "LAD22CD", "ladnm", "oa11cd", "LSOA21NM"])
    assert standardise_columns(table).columns.tolist() == [
        "msoa_code", "msoa_name", "lad_code", "lad_name", "oa_code", "lsoa_name",
    ]


@pytest.mark.parametrize("column", ["pcds", "OBJECTID", "BNG_E", "LSOA21NMW", "Shape__Area", "geometry"])
def test_standardise_columns_leaves_other_columns_alone(column):
    """
    Test that standardise_columns leaves other columns alone, even if similar to the columns that are intended to be.
    """
    assert standardise_columns(pd.DataFrame(columns=[column])).columns.tolist() == [column]


def test_standardise_columns_keeps_geodataframe_and_geometry():
    """Test that standardise_columns keeps a GeoDataFrame and geometry column if present."""
    gdf = gpd.GeoDataFrame({"MSOA21CD": ["E02000001"]}, geometry=[Point(1, 2)], crs="EPSG:27700")
    out = standardise_columns(gdf)
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.columns.tolist() == ["msoa_code", "geometry"]
    assert out.crs == gdf.crs
    assert out.geometry.iloc[0] == Point(1, 2)


def test_standardise_columns_does_not_modify_input():
    """Test that standardise_columns does not modify the input DataFrame."""
    table = pd.DataFrame(columns=["MSOA21CD"])
    standardise_columns(table)
    assert table.columns.tolist() == ["MSOA21CD"]


# --- join_to_boundaries ---
def test_join_attaches_data_to_every_polygon():
    """Test that join_to_boundaries attaches data to every polygon (one-to-one join)."""
    codes = ["E02000001", "E02000002", "E02000003"]
    joined = join_to_boundaries(msoa_boundaries(codes), msoa_data(codes), GeoLevel.MSOA)

    assert isinstance(joined, gpd.GeoDataFrame)
    assert joined.crs.to_epsg() == 27700
    assert "_merge" not in joined.columns
    joined = joined.sort_values("msoa_code")
    assert joined["count"].tolist() == [0, 1, 2]
    assert joined.geometry.tolist() == [Point(0, 0), Point(1, 1), Point(2, 2)]


def test_join_keeps_names_from_boundaries():
    """Test that join_to_boundaries keeps names from boundaries."""
    codes = ["E02000001"]
    joined = join_to_boundaries(msoa_boundaries(codes), msoa_data(codes), GeoLevel.MSOA)
    assert joined["msoa_name"].tolist() == ["Boundary E02000001"]
    assert not {"msoa_name_x", "msoa_name_y"} & set(joined.columns)


def test_join_works_without_a_name_column_in_data():
    """Test that join_to_boundaries works without a name column in data (it's optional!)"""
    codes = ["E02000001"]
    data = msoa_data(codes).drop(columns="msoa_name")
    joined = join_to_boundaries(msoa_boundaries(codes), data, GeoLevel.MSOA)
    assert joined["count"].tolist() == [0]


def test_join_does_not_modify_inputs():
    """Test that join_to_boundaries does not modify the input DataFrames."""
    codes = ["E02000001"]
    data = msoa_data(codes)
    join_to_boundaries(msoa_boundaries(codes), data, GeoLevel.MSOA)
    assert "msoa_name" in data.columns


def test_join_raises_for_polygon_without_data():
    """Test that join_to_boundaries raises a ValueError if there are polygons without data (no unjoined rows)."""
    with pytest.raises(ValueError, match=r"1 polygons without data \(e.g. \['E02000003'\]\), 0 data rows"):
        join_to_boundaries(msoa_boundaries(["E02000001", "E02000003"]), msoa_data(["E02000001"]), GeoLevel.MSOA)


def test_join_raises_for_data_without_polygon():
    """Test that join_to_boundaries raises a ValueError if there are data rows without a polygon (no unjoined rows)."""
    with pytest.raises(ValueError, match=r"0 polygons without data .* 2 data rows without a polygon"):
        join_to_boundaries(msoa_boundaries(["E02000001"]), msoa_data(["E02000001", "E02000008", "E02000009"]),
                           GeoLevel.MSOA)


def test_join_raises_for_duplicate_codes():
    """Test that join_to_boundaries raises a ValueError if there are duplicate codes."""
    codes = ["E02000001", "E02000001"]
    with pytest.raises(pd.errors.MergeError):
        join_to_boundaries(msoa_boundaries(["E02000001"]), msoa_data(codes), GeoLevel.MSOA)


def test_join_uses_the_level_code_column():
    """Test that join_to_boundaries uses the level code column to join on."""
    boundaries = gpd.GeoDataFrame({"lad_code": ["E08000001"], "lad_name": ["Bolton"]},
                                  geometry=[Point(0, 0)], crs="EPSG:27700")
    data = pd.DataFrame({"lad_code": ["E08000001"], "lad_name": ["Bolton"], "count": [1069]})
    joined = join_to_boundaries(boundaries, data, GeoLevel.LAD)
    assert joined[["lad_code", "lad_name", "count"]].values.tolist() == [["E08000001", "Bolton", 1069]]
