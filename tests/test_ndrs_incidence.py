import pandas as pd
import pytest

from bcviz.data import ndrs_incidence as inc
from bcviz.data.geography import GM_DISTRICTS, GeoLevel

# The header NDRS's download uses, including its stray trailing space
NDRS_HEADER = ["Geography code", "Geography name ", "NDRS main", "Gender", "Year", "Type of rate", "Count", "Rate",
               "95% lower confidence interval", "95% upper confidence interval", "Flag"]

# The columns pivot_age_rates produces, as in the project's saved incidence tables
PIVOTED_COLUMNS = ["count", "crude_rate", "crude_lower_ci", "crude_upper_ci", "crude_flag",
                   "asr_rate", "asr_lower_ci", "asr_upper_ci", "asr_flag"]


def ndrs_rows(code: str, name: str, count: str, crude: tuple, asr: tuple) -> list[list[str]]:
    """The two rows NDRS gives for an area: (rate, lower, upper, flag) for each of its rate types."""
    return [[code, name, "Breast", "Females", "2018-2022", rate_type, count, *values]
            for rate_type, values in (("Non-standardised", crude), ("Age-standardised", asr))]


def write_ndrs_csv(path, rows):
    pd.DataFrame(rows, columns=NDRS_HEADER).to_csv(path, index=False)
    return path

# Two MSOA rows, one with full data and another with NDRS-marked "u" (not enough counts)
MSOA_ROWS = [
    *ndrs_rows("E02000984", "Bolton 001", "30", ("151.8", "110.7", "228.6", "NA"), ("144.3", "98.5", "204.0", "NA")),
    *ndrs_rows("E02000985", "Bolton 002", "[u]", ("[u]", "[u]", "[u]", "*"), ("[u]", "[u]", "[u]", "*")),
]


@pytest.fixture
def msoa_csv(tmp_path):
    return write_ndrs_csv(tmp_path / "msoa.csv", MSOA_ROWS)


# --- clean_incidence ---
def test_clean_uses_msoa_columns_for_msoas(msoa_csv):
    """Test that `clean_incidence` renames all columns correctly for the MSOA CSV."""
    cleaned = inc.clean_incidence(msoa_csv, GeoLevel.MSOA)
    assert {"msoa_code", "msoa_name", "count", "rate", "lower_ci", "upper_ci", "flag", "rate_type"} \
        <= set(cleaned.columns)
    assert cleaned["msoa_name"].tolist() == ["Bolton 001", "Bolton 001", "Bolton 002", "Bolton 002"]


def test_clean_uses_lad_columns_for_districts(tmp_path):
    """Test that `clean_incidence` uses the correct columns for the LAD level."""
    path = write_ndrs_csv(tmp_path / "lad.csv",
                          ndrs_rows("E08000001", "Bolton", "1069", ("142.8", "134.4", "151.6", "NA"),
                                    ("151.2", "142.3", "160.6", "NA")))
    cleaned = inc.clean_incidence(path, GeoLevel.LAD)
    assert cleaned["lad_code"].tolist() == ["E08000001", "E08000001"]
    assert "msoa_code" not in cleaned.columns


def test_clean_shortens_rate_types(msoa_csv):
    """Test that `clean_incidence` shortens the rate types to "crude" and "asr"""
    assert inc.clean_incidence(msoa_csv, GeoLevel.MSOA)["rate_type"].tolist() == ["crude", "asr", "crude", "asr"]


def test_clean_converts_numbers(msoa_csv):
    """Test that `clean_incidence` converts numbers correctly (i.e., keeping those columns numeric)."""
    cleaned = inc.clean_incidence(msoa_csv, GeoLevel.MSOA)
    assert cleaned["count"].dtype == "Int64"
    for column in ["rate", "lower_ci", "upper_ci"]:
        assert pd.api.types.is_float_dtype(cleaned[column])
    assert cleaned.loc[0, ["rate", "lower_ci", "upper_ci"]].tolist() == [151.8, 110.7, 228.6]
    assert cleaned.loc[0, "count"] == 30


def test_clean_turns_suppressed_values_into_missing(msoa_csv):
    """Test that `clean_incidence` turns suppressed values (e.g, ["u"]) into missing values (None)."""
    suppressed = inc.clean_incidence(msoa_csv, GeoLevel.MSOA).iloc[2:]
    assert suppressed[["count", "rate", "lower_ci", "upper_ci"]].isna().all().all()


def test_clean_keeps_real_flags_and_drops_na_flags(msoa_csv):
    """Test that `clean_incidence` keeps real flags (e.g., "*") and drops NA flags (e.g., "NA" becomes None)."""
    flags = inc.clean_incidence(msoa_csv, GeoLevel.MSOA)["flag"]
    assert flags.isna().tolist() == [True, True, False, False]
    assert flags.iloc[2] == "*"


def test_clean_rejects_unknown_rate_type(tmp_path):
    """Test that `clean_incidence` rejects rate types it doesn't know about (not specified in program)."""
    rows = ndrs_rows("E02000984", "Bolton 001", "30", ("1", "1", "1", "NA"), ("1", "1", "1", "NA"))
    rows[1][5] = "Directly standardised"
    with pytest.raises(ValueError, match="Unknown rate types: {'Directly standardised'}"):
        inc.clean_incidence(write_ndrs_csv(tmp_path / "x.csv", rows), GeoLevel.MSOA)


def test_clean_rejects_unexpected_text_in_numbers(tmp_path):
    """Test that `clean_incidence` rejects unexpected text in numerical columns (e.g., "[c]")."""
    rows = ndrs_rows("E02000984", "Bolton 001", "30", ("1", "1", "1", "NA"), ("1", "1", "1", "NA"))
    rows[0][7] = "[c]"
    with pytest.raises(ValueError):
        inc.clean_incidence(write_ndrs_csv(tmp_path / "x.csv", rows), GeoLevel.MSOA)


# --- pivot_age_rates ---
@pytest.fixture
def pivoted(msoa_csv):
    return inc.pivot_age_rates(inc.clean_incidence(msoa_csv, GeoLevel.MSOA), GeoLevel.MSOA)


def test_pivot_gives_one_row_per_area_with_saved_column_layout(pivoted):
    """Test that `pivot_age_rates` gives one row per area with the expected column layout."""
    assert pivoted.columns.tolist() == ["msoa_code", "msoa_name", *PIVOTED_COLUMNS]
    assert pivoted["msoa_code"].tolist() == ["E02000984", "E02000985"]


def test_pivot_puts_each_rate_type_in_its_own_columns(pivoted):
    """Test that `pivot_age_rates` puts each rate type in its own columns correctly (matches cells)."""
    row = pivoted.iloc[0]
    assert row["count"] == 30
    assert (row["crude_rate"], row["crude_lower_ci"], row["crude_upper_ci"]) == (151.8, 110.7, 228.6)
    assert (row["asr_rate"], row["asr_lower_ci"], row["asr_upper_ci"]) == (144.3, 98.5, 204.0)


def test_pivot_keeps_number_types(pivoted):
    """
    Test that `pivot_age_rates` keeps the number types (e.g., Int64 for counts), as pivoting including text columns
    normally changes them.
    """
    assert pivoted["count"].dtype == "Int64"
    for rate_type in inc.RATE_TYPES.values():
        for stat in ("rate", "lower_ci", "upper_ci"):
            assert pd.api.types.is_float_dtype(pivoted[f"{rate_type}_{stat}"])


def test_pivot_keeps_missing_values_and_flags(pivoted):
    """Test that `pivot_age_rates` keeps missing values and flags correctly (e.g., NaN for suppressed values)."""
    suppressed = pivoted.iloc[1]
    assert pd.isna(suppressed["count"])
    assert pd.isna(suppressed["asr_rate"])
    assert suppressed["crude_flag"] == "*"
    assert pd.isna(pivoted.iloc[0]["asr_flag"])


def test_pivot_rejects_repeated_rows(tmp_path):
    """Test that `pivot_age_rates` rejects repeated rows."""
    rows = [*MSOA_ROWS, MSOA_ROWS[0]]
    cleaned = inc.clean_incidence(write_ndrs_csv(tmp_path / "x.csv", rows), GeoLevel.MSOA)
    with pytest.raises(ValueError):
        inc.pivot_age_rates(cleaned, GeoLevel.MSOA)


# --- download_all_bc_incidence ---
@pytest.fixture
def fake_download(monkeypatch):
    """Replace the NDRS browser download, recording its arguments."""
    calls = []

    def download_incidence(**kwargs):
        calls.append(kwargs)
        return kwargs["dest"]

    monkeypatch.setattr(inc.ndrs, "download_incidence", download_incidence)
    return calls


def test_download_asks_for_every_gm_msoa_by_name(local_lookup, gm_msoas, fake_download):
    """Test that `download_all_bc_incidence` asks for every GM MSOA by name, as needed for the dashboard."""
    assert inc.download_all_bc_incidence(GeoLevel.MSOA, "out.csv", "2016-2020") == "out.csv"
    [call] = fake_download
    assert sorted(call.pop("areas")) == sorted(gm_msoas["msoa21nm"])
    assert call == {"geography": GeoLevel.MSOA, "dest": "out.csv", "site": "Breast", "sex": "Females",
                    "years": "2016-2020"}


def test_download_asks_for_every_gm_district_by_name(local_lookup, fake_download):
    """Test tbat `download_all_bc_incidence` asks for every GM district by name, as needed for the dashboard."""
    inc.download_all_bc_incidence(GeoLevel.LAD, "out.csv", "2018-2022")
    assert fake_download[0]["areas"] == list(GM_DISTRICTS)


@pytest.mark.parametrize("level", [GeoLevel.OA, GeoLevel.LSOA])
def test_download_rejects_levels_ndrs_does_not_publish(fake_download, level):
    """Test that `download_all_bc_incidence` rejects levels NDRS doesn't publish e.g., OAs & LSOAs."""
    with pytest.raises(ValueError, match="only has incidence data for MSOAs & LADs"):
        inc.download_all_bc_incidence(level, "out.csv", "2018-2022")
    assert fake_download == []


@pytest.mark.parametrize("level", [GeoLevel.MSOA, GeoLevel.LAD])
@pytest.mark.parametrize("years", ["2019-2023", "2018", "2018-2020"])
def test_download_rejects_unsupported_year_bands(fake_download, level, years):
    """
    Test that `download_all_bc_incidence` rejects unsupported year bands (e.g., 2018, 2019-2023).
    
    Note that there are indeed different year bands for LAD data specifically, but these are currently not supported.
    """
    with pytest.raises(ValueError, match="valid 5-year range"):
        inc.download_all_bc_incidence(level, "out.csv", years)
    assert fake_download == []


# --- get_bc_incidence ---
LAD_ROWS = ndrs_rows("E08000001", "Bolton", "1069", ("142.8", "134.4", "151.6", "NA"),
                     ("151.2", "142.3", "160.6", "NA"))


@pytest.fixture
def fake_raw_download(monkeypatch):
    """Make the download write a raw NDRS file for the level, recording the levels and years asked for."""
    calls = []

    def download(level, dest, years):
        calls.append((level, years))
        write_ndrs_csv(dest, MSOA_ROWS if level == GeoLevel.MSOA else LAD_ROWS)
        return dest

    monkeypatch.setattr(inc, "download_all_bc_incidence", download)
    return calls


def test_get_bc_incidence_replaces_raw_files_with_pivoted_tables(fake_raw_download, tmp_path):
    """Test that `get_bc_incidence` replaces raw files (tables) with pivoted tables."""
    msoa_path, lad_path = tmp_path / "msoa.csv", tmp_path / "lad.csv"
    msoa, lad = inc.get_bc_incidence(msoa_path, lad_path, years="2015-2019")

    assert fake_raw_download == [(GeoLevel.MSOA, "2015-2019"), (GeoLevel.LAD, "2015-2019")]
    assert msoa.columns.tolist() == ["msoa_code", "msoa_name", *PIVOTED_COLUMNS]
    assert lad.columns.tolist() == ["lad_code", "lad_name", *PIVOTED_COLUMNS]
    assert pd.read_csv(msoa_path).columns.tolist() == msoa.columns.tolist()
    assert pd.read_csv(lad_path)["lad_name"].tolist() == ["Bolton"]


def test_get_bc_incidence_reads_existing_files_instead_of_downloading(fake_raw_download, tmp_path):
    """Test that `get_bc_incidence` reads existing files instead of downloading new ones."""
    msoa_path, lad_path = tmp_path / "msoa.csv", tmp_path / "lad.csv"
    first_msoa, _ = inc.get_bc_incidence(msoa_path, lad_path)
    lad_path.unlink()
    saved = msoa_path.read_text()
    fake_raw_download.clear()

    msoa, lad = inc.get_bc_incidence(msoa_path, lad_path)
    assert fake_raw_download == [(GeoLevel.LAD, "2018-2022")]
    pd.testing.assert_frame_equal(msoa, first_msoa)  # still first, and typed as if just made
    assert lad["lad_name"].tolist() == ["Bolton"]
    assert msoa_path.read_text() == saved


# --- read_incidence ---
def test_read_incidence_restores_column_types(pivoted, tmp_path):
    """Test that `read_incidence` restores column types (e.g., Int64 for count)."""
    path = tmp_path / "saved.csv"
    pivoted.to_csv(path, index=False)
    pd.testing.assert_frame_equal(inc.read_incidence(path), pivoted)


def test_read_incidence_keeps_flags_as_text_when_all_missing(tmp_path):
    """Test that `read_incidence` keeps flags as text when all are missing."""
    path = tmp_path / "saved.csv"
    path.write_text("msoa_code,msoa_name,count,crude_flag,asr_flag\nE02000984,Bolton 001,,,\n")
    table = inc.read_incidence(path)
    assert table["count"].dtype == "Int64"
    assert pd.api.types.is_string_dtype(table["crude_flag"])
    assert pd.api.types.is_string_dtype(table["asr_flag"])
