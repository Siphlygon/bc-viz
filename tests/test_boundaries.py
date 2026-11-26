import geopandas as gpd
import pytest
from shapely.geometry import Point

from bcviz.data import boundaries
from bcviz.data.geography import GM_DISTRICTS, GeoLevel


def points(codes: list[str], code_col: str) -> gpd.GeoDataFrame:
    """A boundary-like frame with one point per code, in British National Grid."""
    return gpd.GeoDataFrame({code_col: codes},
                            geometry=[Point(380_000 + i, 400_000) for i in range(len(codes))],
                            crs="EPSG:27700")


# --- download_boundaries ---
@pytest.fixture
def fake_arcgis(monkeypatch):
    """Record the services asked for, and return a sentinel frame from query_features."""
    calls = {"services": [], "queried": []}
    result = points(["E02000000"], "MSOA21CD")

    def ons_layer_url(service, layer=None):
        calls["services"].append(service)
        return f"https://example.test/{service}/0"

    def query_features(url, *args, **kwargs):
        calls["queried"].append(url)
        return result

    monkeypatch.setattr(boundaries.ag, "ons_layer_url", ons_layer_url)
    monkeypatch.setattr(boundaries.ag, "query_features", query_features)
    calls["result"] = result
    return calls


@pytest.mark.parametrize("level, service", [
    (GeoLevel.MSOA, boundaries.MSOA_BOUNDARY_ID),
    (GeoLevel.LAD, boundaries.LAD_BOUNDARY_ID),
])
def test_download_boundaries_queries_the_level_service(fake_arcgis, level, service):
    """Test that `download_boundaries` queries the correct service for the given geography level."""
    assert boundaries.download_boundaries(level) is fake_arcgis["result"]
    assert fake_arcgis["services"] == [service]
    assert fake_arcgis["queried"] == [f"https://example.test/{service}/0"]


@pytest.mark.parametrize("level", [GeoLevel.OA, GeoLevel.LSOA])
def test_download_boundaries_rejects_unsupported_levels(fake_arcgis, level):
    """Test that `download_boundaries` rejects unsupported levels."""
    with pytest.raises(ValueError, match="Invalid geo_level"):
        boundaries.download_boundaries(level)
    assert fake_arcgis["services"] == []


# --- filter_to_gm ---
def test_filter_to_gm_keeps_only_gm_msoas(local_lookup, gm_msoas, non_gm_district):
    """Test that `filter_to_gm` keeps only the Greater Manchester MSOAs."""
    gm_codes = gm_msoas["msoa21cd"].tolist()
    outside = local_lookup.loc[local_lookup["ladnm"] == non_gm_district[0], "msoa21cd"].unique().tolist()
    everything = points(gm_codes + outside, "MSOA21CD")

    filtered = boundaries.filter_to_gm(everything, GeoLevel.MSOA)
    assert sorted(filtered["MSOA21CD"]) == gm_codes
    assert isinstance(filtered, gpd.GeoDataFrame)
    assert filtered.crs == everything.crs


def test_filter_to_gm_keeps_only_gm_districts(local_lookup, non_gm_district):
    """Test that `filter_to_gm` keeps only the Greater Manchester LADs."""
    everything = points([*GM_DISTRICTS.values(), non_gm_district[1]], "LAD21CD")
    filtered = boundaries.filter_to_gm(everything, GeoLevel.LAD)
    assert sorted(filtered["LAD21CD"]) == sorted(GM_DISTRICTS.values())


@pytest.mark.parametrize("level", [GeoLevel.OA, GeoLevel.LSOA])
def test_filter_to_gm_rejects_unsupported_levels(level):
    """Test that `filter_to_gm` rejects unsupported geography levels."""
    with pytest.raises(ValueError, match="Invalid geo_level"):
        boundaries.filter_to_gm(points([], "X"), level)


# --- get_all_boundaries ---
@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the download and the GM filter with fakes that record which levels they were used for."""
    calls = {"downloaded": [], "filtered": []}
    frames = {GeoLevel.MSOA: points(["E02000000", "E02999999"], "MSOA21CD"),
              GeoLevel.LAD: points(["E08000001", "E06000008"], "LAD21CD")}

    def download(level):
        calls["downloaded"].append(level)
        return frames[level]

    def filter_to_gm(gdf, level):
        calls["filtered"].append(level)
        return gdf.iloc[:1]

    monkeypatch.setattr(boundaries, "download_boundaries", download)
    monkeypatch.setattr(boundaries, "filter_to_gm", filter_to_gm)
    return calls


def test_get_all_boundaries_downloads_filters_and_saves(fake_pipeline, tmp_path):
    """Test that `get_all_boundaries` downloads, filters, and saves the boundaries correctly using known codes."""
    msoa_path, lad_path = tmp_path / "msoa.gpkg", tmp_path / "lad.gpkg"
    msoa, lad = boundaries.get_all_boundaries(msoa_path, lad_path)

    assert fake_pipeline["downloaded"] == [GeoLevel.MSOA, GeoLevel.LAD]
    assert fake_pipeline["filtered"] == [GeoLevel.MSOA, GeoLevel.LAD]
    assert msoa["MSOA21CD"].tolist() == ["E02000000"]
    assert lad["LAD21CD"].tolist() == ["E08000001"]
    assert gpd.read_file(msoa_path)["MSOA21CD"].tolist() == ["E02000000"]
    assert gpd.read_file(lad_path)["LAD21CD"].tolist() == ["E08000001"]


def test_get_all_boundaries_can_skip_gm_filter(fake_pipeline, tmp_path):
    """Test that `get_all_boundaries` can skip the Greater Manchester filter if turned off in the call."""
    msoa, lad = boundaries.get_all_boundaries(tmp_path / "msoa.gpkg", tmp_path / "lad.gpkg", gm_only=False)
    assert fake_pipeline["filtered"] == []
    assert len(msoa) == 2 and len(lad) == 2


def test_get_all_boundaries_reads_existing_files_instead_of_downloading(fake_pipeline, tmp_path):
    """Test that `get_all_boundaries` reads existing files instead of downloading if they are present."""
    msoa_path, lad_path = tmp_path / "msoa.gpkg", tmp_path / "lad.gpkg"
    points(["E02000123"], "MSOA21CD").to_file(msoa_path)
    saved = msoa_path.read_bytes()

    msoa, lad = boundaries.get_all_boundaries(msoa_path, lad_path)
    assert fake_pipeline["downloaded"] == [GeoLevel.LAD]
    assert msoa["MSOA21CD"].tolist() == ["E02000123"]  # still first, so callers can pair results with levels
    assert msoa.crs.to_epsg() == 27700
    assert lad["LAD21CD"].tolist() == ["E08000001"]
    assert msoa_path.read_bytes() == saved


def test_get_all_boundaries_with_both_files_saved_downloads_nothing(fake_pipeline, tmp_path):
    """Test that `get_all_boundaries` does not download if both files are already present."""
    msoa_path, lad_path = tmp_path / "msoa.gpkg", tmp_path / "lad.gpkg"
    points(["E02000123"], "MSOA21CD").to_file(msoa_path)
    points(["E08000001"], "LAD21CD").to_file(lad_path)

    msoa, lad = boundaries.get_all_boundaries(msoa_path, lad_path)
    assert fake_pipeline["downloaded"] == []
    assert msoa["MSOA21CD"].tolist() == ["E02000123"]
    assert lad["LAD21CD"].tolist() == ["E08000001"]
