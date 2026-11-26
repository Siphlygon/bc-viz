import pandas as pd
import pytest

from bcviz.data import geography
from bcviz.data.geography import GM_DISTRICTS, GeoArea, GeoHierarchy, GeoLevel

# One known area from `ons_lookup` at each level, all inside Bolton
BOLTON = "E08000001"
BOLTON_MSOA = "E02000000"  # "Bolton 001"
BOLTON_LSOA = "E01000000"  # "Bolton 001A"
BOLTON_OA = "E00000000"


# --- GeoLevel and GeoArea ---
def test_geo_levels_are_ordered_finest_to_coarsest():
    """Test that the GeoLevels are correctly ordered from the finest scale (OA) to the coarsest (currently LAD)."""
    assert GeoLevel.OA < GeoLevel.LSOA < GeoLevel.MSOA < GeoLevel.LAD


@pytest.mark.parametrize("level, code_col, name_col", [
    (GeoLevel.OA, "oa_code", "oa_name"),
    (GeoLevel.LSOA, "lsoa_code", "lsoa_name"),
    (GeoLevel.MSOA, "msoa_code", "msoa_name"),
    (GeoLevel.LAD, "lad_code", "lad_name"),
])
def test_geo_level_column_names(level, code_col, name_col):
    """Test that the column names for each GeoLevel are correct based on the level & presets."""
    assert level.code_col == code_col
    assert level.name_col == name_col


def test_geo_area_without_name_is_named_oa():
    """
    Test that a GeoArea without a name is named 'OA'.
    
    All other area measurements have a name e.g., "Bolton" for LAD, "Bolton 001" for MSOA, "Bolton 001A" for LSOA.
    Perhaps due to how small/numerous OAs are, they don't have a name and simply have a code. Nonetheless, the names of
    GeoAreas are used in the dashboard, so this is important & cannot be left blank.
    """
    assert GeoArea(name=None, code=BOLTON_OA, level=GeoLevel.OA).name == "OA"


# --- building a GeoHierarchy ---
def test_from_ons_lookup_renames_ons_columns(hierarchy):
    """Test that creating a GeoHierarchy from the ONS lookup table renames the columns correctly."""
    assert hierarchy.table.columns.tolist() == [
        "pcds", "oa_code", "lsoa_code", "msoa_code", "lad_code", "lsoa_name", "msoa_name", "lad_name",
    ]


@pytest.mark.parametrize("ons_column, ours", [
    ("LAD22CD", "lad_code"),
    ("msoa21cd", "msoa_code"),
    ("Lsoa11Nm", "lsoa_name"),
    ("OA21CD", "oa_code"),
])
def test_from_ons_lookup_handles_any_year_and_case(ons_column, ours):
    """Test that creating a GeoHierarchy from the ONS lookup table handles any year and case."""
    table = GeoHierarchy.from_ons_lookup(pd.DataFrame({ons_column: [], "LAD22NM": []})).table
    assert {ours, "lad_name"} <= set(table.columns)


@pytest.mark.parametrize("column", ["pcds", "LSOA21NMW", "OBJECTID", "MSOA21CDX"])
def test_from_ons_lookup_leaves_other_columns_alone(column):
    """
    Test that creating a GeoHierarchy from the ONS lookup table leaves other columns alone, even if similar to the
    columns that are intended to be changed.
    """
    table = GeoHierarchy.from_ons_lookup(pd.DataFrame({column: [], "ladnm": []})).table
    assert column in table.columns


def test_from_ons_lookup_reads_local_lookup_when_none_given(local_lookup):
    """Test that creating a GeoHierarchy from the ONS lookup table reads the local lookup when none is given."""
    assert set(GeoHierarchy.from_ons_lookup().table["lad_name"]) == set(GM_DISTRICTS)


def test_hierarchy_filters_to_gm_by_default(hierarchy):
    """Test that creating a GeoHierarchy from the ONS lookup table filters to Greater Manchester by default."""
    assert set(hierarchy.table["lad_name"]) == set(GM_DISTRICTS)


def test_hierarchy_can_skip_filtering(ons_lookup, non_gm_district):
    """
    Test that creating a GeoHierarchy from the ONS lookup table can skip filtering to Greater Manchester if specified.
    """
    table = GeoHierarchy(ons_lookup.rename(columns={"ladnm": "lad_name"}), filter_districts=None).table
    assert set(table["lad_name"]) == {*GM_DISTRICTS, non_gm_district[0]}


def test_hierarchy_can_filter_to_chosen_districts(ons_lookup):
    """Test that creating a GeoHierarchy from the ONS lookup table can filter to chosen districts if specified."""
    table = GeoHierarchy(ons_lookup.rename(columns={"ladnm": "lad_name"}), filter_districts={"Bury", "Wigan"}).table
    assert set(table["lad_name"]) == {"Bury", "Wigan"}


# --- _get_level_from_code ---
@pytest.mark.parametrize("code, level", [
    ("E00000001", GeoLevel.OA),
    ("E01000001", GeoLevel.LSOA),
    ("E02000001", GeoLevel.MSOA),
    ("E06000008", GeoLevel.LAD),
    ("E07000001", GeoLevel.LAD),
    ("E08000001", GeoLevel.LAD),
    ("E09000001", GeoLevel.LAD),
])
def test_level_from_code(hierarchy, code, level):
    """Test that _get_level_from_code returns the correct GeoLevel for a given code."""
    assert hierarchy._get_level_from_code(code) == level


# Counties (E10), regions (E12) and Scottish council areas (S12) share a 3rd character with LSOAs, so must not be
# mistaken for them
@pytest.mark.parametrize("code", ["E03000001", "E04000001", "E05000001", "E10000001", "E12000002", "S12000033",
                                  "E0", ""])
def test_level_from_unrecognised_code_raises(hierarchy, code):
    """Test that _get_level_from_code raises a ValueError when given an unrecognised code."""
    with pytest.raises(ValueError, match="Cannot identify"):
        hierarchy._get_level_from_code(code)


# --- to_geo_area ---
def test_to_geo_area_single_code(hierarchy):
    """Test that to_geo_area returns the correct GeoArea for a given code."""
    assert hierarchy.to_geo_area(BOLTON_MSOA) == GeoArea("Bolton 001", BOLTON_MSOA, GeoLevel.MSOA)


def test_to_geo_area_list_of_codes(hierarchy):
    """Test that to_geo_area returns the correct GeoAreas for a list of codes."""
    assert hierarchy.to_geo_area([BOLTON, BOLTON_LSOA]) == [
        GeoArea("Bolton", BOLTON, GeoLevel.LAD),
        GeoArea("Bolton 001A", BOLTON_LSOA, GeoLevel.LSOA),
    ]


def test_to_geo_area_unknown_code_raises(hierarchy):
    """Test that to_geo_area raises a ValueError when given an unknown code."""
    with pytest.raises(ValueError, match="No area found with code: E02999999"):
        hierarchy.to_geo_area("E02999999")


def test_to_geo_area_ignores_areas_filtered_out(hierarchy, non_gm_district):
    """Test that to_geo_area ignores areas filtered out by the hierarchy."""
    with pytest.raises(ValueError, match="No area found"):
        hierarchy.to_geo_area(non_gm_district[1])


def test_to_geo_area_names_oas_with_placeholder(hierarchy):
    """Test that to_geo_area names OAs with a placeholder if no name is available."""
    # The ONS lookup has no OA names
    assert "oa_name" not in hierarchy.table.columns
    assert hierarchy.to_geo_area(BOLTON_OA) == GeoArea("OA", BOLTON_OA, GeoLevel.OA)


def test_to_geo_area_still_needs_names_above_oa(ons_lookup):
    """Test that to_geo_area still needs names above OA even if no names are available."""
    nameless = GeoHierarchy.from_ons_lookup(ons_lookup.drop(columns="msoa21nm"))
    with pytest.raises(KeyError, match="msoa_name"):
        nameless.to_geo_area(BOLTON_MSOA)


# --- get_parent ---
@pytest.mark.parametrize("code, parent_level, expected", [
    (BOLTON_LSOA, GeoLevel.MSOA, GeoArea("Bolton 001", BOLTON_MSOA, GeoLevel.MSOA)),
    (BOLTON_LSOA, GeoLevel.LAD, GeoArea("Bolton", BOLTON, GeoLevel.LAD)),
    (BOLTON_OA, GeoLevel.LSOA, GeoArea("Bolton 001A", BOLTON_LSOA, GeoLevel.LSOA)),
    (BOLTON_MSOA, GeoLevel.LAD, GeoArea("Bolton", BOLTON, GeoLevel.LAD)),
])
def test_get_parent(hierarchy, code, parent_level, expected):
    """Test that get_parent returns the correct parent GeoArea for a given code and parent level."""
    assert hierarchy.get_parent(code, parent_level) == expected


def test_get_parent_of_lad_raises(hierarchy):
    """Test that get_parent raises a ValueError when given a LAD code (LADs have no implemented parents)."""
    with pytest.raises(ValueError, match="LADs do not have any parent"):
        hierarchy.get_parent(BOLTON, GeoLevel.LAD)


@pytest.mark.parametrize("parent_level", [GeoLevel.OA, GeoLevel.LSOA])
def test_get_parent_at_same_or_finer_level_raises(hierarchy, parent_level):
    """Test that get_parent raises a ValueError when given a code at the same or finer level than the parent level."""
    with pytest.raises(ValueError, match="are not parents of"):
        hierarchy.get_parent(BOLTON_LSOA, parent_level)


# 3 is GeoLevel.MSOA's value, but a plain int has no code_col, so it must be rejected too
@pytest.mark.parametrize("not_a_level", [7, 3, "LAD", None])
def test_get_parent_rejects_non_level(hierarchy, not_a_level):
    """Test that get_parent raises a ValueError when given a non-level parent level."""
    with pytest.raises(TypeError, match="Invalid parent level"):
        hierarchy.get_parent(BOLTON_LSOA, not_a_level)


def test_get_parent_of_unknown_code_raises(hierarchy):
    """Test that get_parent raises a ValueError when given an unknown code."""
    with pytest.raises(ValueError, match="No area found with code: E01999999"):
        hierarchy.get_parent("E01999999", GeoLevel.LAD)


# --- areas_in ---
def by_code(areas):
    return sorted(areas, key=lambda a: a.code)


def test_areas_in_returns_each_child_once(hierarchy):
    """
    Test that areas_in returns each child once only (they are duplicated heavily in the raw table, as e.g., there will
    be multiple postcode rows for each LSOA, but every LSOA only has one parent).
    """
    # Each LSOA appears on 4 postcode rows of the table, but must be returned once
    assert by_code(hierarchy.areas_in(BOLTON_MSOA, GeoLevel.LSOA)) == [
        GeoArea("Bolton 001A", "E01000000", GeoLevel.LSOA),
        GeoArea("Bolton 001B", "E01000001", GeoLevel.LSOA),
    ]


def test_areas_in_district(hierarchy):
    """
    Test that areas_in returns the correct child GeoAreas for a given parent code, specifically using test districts.
    """
    assert by_code(hierarchy.areas_in(BOLTON, GeoLevel.MSOA)) == [
        GeoArea("Bolton 001", "E02000000", GeoLevel.MSOA),
        GeoArea("Bolton 002", "E02000001", GeoLevel.MSOA),
    ]


def test_areas_in_oa_raises(hierarchy):
    """Test that areas_in raises a ValueError when given an OA code (OAs have no implemented children)."""
    with pytest.raises(ValueError, match="OAs do not have any child areas"):
        hierarchy.areas_in(BOLTON_OA, GeoLevel.OA)


@pytest.mark.parametrize("child_level", [GeoLevel.MSOA, GeoLevel.LAD])
def test_areas_in_at_same_or_coarser_level_raises(hierarchy, child_level):
    """Test that areas_in raises a ValueError when given a code at the same or coarser level than the child level."""
    with pytest.raises(ValueError, match="are not children of"):
        hierarchy.areas_in(BOLTON_MSOA, child_level)


def test_areas_in_unknown_code_raises(hierarchy):
    """Test that areas_in raises a ValueError when given an unknown code."""
    with pytest.raises(ValueError, match="No area found with code: E02999999"):
        hierarchy.areas_in("E02999999", GeoLevel.LSOA)


def test_areas_in_can_return_oas(hierarchy):
    """
    Test that areas_in can return OAs if specified, specifically testing here that we can return non-direct children (as
    OAs are not direct children of LADs).
    """
    assert by_code(hierarchy.areas_in(BOLTON_LSOA, GeoLevel.OA)) == [
        GeoArea("OA", "E00000000", GeoLevel.OA),
        GeoArea("OA", "E00000001", GeoLevel.OA),
    ]


# --- get_areas_in_gm ---
def test_get_areas_in_gm_for_districts_needs_no_table():
    """
    Test that get_areas_in_gm returns the correct GeoAreas for districts if no table is provided, as it relies on a
    pre-defined dictionary of districts.
    """
    empty = GeoHierarchy(pd.DataFrame({"lad_name": []}))
    assert empty.get_areas_in_gm(GeoLevel.LAD) == [
        GeoArea(name, code, GeoLevel.LAD) for name, code in GM_DISTRICTS.items()
    ]


def test_get_areas_in_gm_for_msoas(hierarchy, gm_msoas):
    """Test that get_areas_in_gm returns the correct GeoAreas for MSOAs"""
    areas = by_code(hierarchy.get_areas_in_gm(GeoLevel.MSOA))
    assert [a.code for a in areas] == gm_msoas["msoa21cd"].tolist()
    assert [a.name for a in areas] == gm_msoas["msoa21nm"].tolist()
    assert all(a.level == GeoLevel.MSOA for a in areas)


def test_get_areas_in_gm_for_oas(hierarchy, ons_lookup):
    """Test that get_areas_in_gm returns the correct GeoAreas for OAs"""
    areas = hierarchy.get_areas_in_gm(GeoLevel.OA)
    gm_oas = ons_lookup.loc[ons_lookup["ladnm"].isin(GM_DISTRICTS), "oa21cd"].unique()
    assert sorted(a.code for a in areas) == sorted(gm_oas)
    assert {a.name for a in areas} == {"OA"}


@pytest.mark.parametrize("not_a_level", [7, 3, "LAD", None])
def test_get_areas_in_gm_rejects_non_level(hierarchy, not_a_level):
    """Test that get_areas_in_gm raises a ValueError when given a non-level geo level."""
    with pytest.raises(TypeError, match="Invalid geo level"):
        hierarchy.get_areas_in_gm(not_a_level)


def test_get_areas_in_gm_raises_if_a_district_is_missing(ons_lookup):
    """Test that get_areas_in_gm raises a ValueError if a district is missing"""
    without_wigan = GeoHierarchy.from_ons_lookup(ons_lookup[ons_lookup["ladnm"] != "Wigan"])
    with pytest.raises(ValueError, match=f"No area found with code: {GM_DISTRICTS['Wigan']}"):
        without_wigan.get_areas_in_gm(GeoLevel.MSOA)


# --- get_lad_code ---
@pytest.mark.parametrize("name", ["Bolton", "bolton", "BOLTON", "bOLTON"])
def test_get_lad_code_ignores_case(hierarchy, name):
    """Test that get_lad_code ignores case."""
    assert hierarchy.get_lad_code(name) == BOLTON


def test_get_lad_code_rejects_districts_outside_gm(hierarchy):
    """Test that get_lad_code raises a ValueError if given a district outside Greater Manchester."""
    with pytest.raises(ValueError, match="Leeds is not a valid name"):
        hierarchy.get_lad_code("Leeds")


# --- _get_geo_lookup_table ---
RAW_LOOKUP = pd.DataFrame({
    "pcd7": ["BL1 1AA"], "pcd8": ["BL1  1AA"], "pcds": ["BL1 1AA"], "usertype": [0],
    "oa21cd": [BOLTON_OA], "ladcd": [BOLTON], "ladnm": ["Bolton"], "ladnmw": [None],
})


@pytest.fixture
def lookup_paths(tmp_path, monkeypatch):
    """Point the lookup at tmp_path, and record calls to download it (which write RAW_LOOKUP as a zip)."""
    csv = tmp_path / "ons_lookup.csv"
    zipped = tmp_path / "ons_lookup.csv.zip"
    monkeypatch.setattr(geography, "GEO_LOOKUP_PATH", str(csv))
    downloads = []

    def fake_download(item_id, dest, expected_title=None):
        downloads.append({"item_id": item_id, "dest": dest, "expected_title": expected_title})
        RAW_LOOKUP.to_csv(dest, index=False, compression={"method": "zip", "archive_name": "lookup.csv"})

    monkeypatch.setattr(geography.ag, "download_item", fake_download)
    return csv, zipped, downloads


def test_lookup_is_downloaded_and_trimmed_when_absent(lookup_paths):
    """Test that the lookup table is downloaded and trimmed (to just GM) when absent."""
    csv, zipped, downloads = lookup_paths
    table = geography._get_geo_lookup_table()

    assert downloads == [{"item_id": geography.GEO_LOOKUP_ID, "dest": str(zipped),
                          "expected_title": geography.GEO_LOOKUP_NAME}]
    assert table.columns.tolist() == ["pcds", "oa21cd", "ladcd", "ladnm"]
    assert pd.read_csv(csv).columns.tolist() == ["pcds", "oa21cd", "ladcd", "ladnm"]


def test_lookup_is_unzipped_without_downloading_when_zip_exists(lookup_paths):
    """Test that the lookup table is unzipped without downloading when the zip exists."""
    csv, zipped, downloads = lookup_paths
    RAW_LOOKUP.to_csv(zipped, index=False, compression={"method": "zip", "archive_name": "lookup.csv"})

    table = geography._get_geo_lookup_table()
    assert downloads == []
    assert table["ladnm"].tolist() == ["Bolton"]
    assert csv.exists()


def test_existing_lookup_csv_is_read_as_is(lookup_paths):
    """Test that the lookup table is read as-is when the CSV exists."""
    csv, zipped, downloads = lookup_paths
    zipped.write_bytes(b"not read")
    pd.DataFrame({"ladcd": [BOLTON], "ladnm": ["Bolton"]}).to_csv(csv, index=False)

    table = geography._get_geo_lookup_table()
    assert downloads == []
    assert table.to_dict("list") == {"ladcd": [BOLTON], "ladnm": ["Bolton"]}
