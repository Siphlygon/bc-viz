"""
Tests for bcviz.acquire.ndrs. No browser is launched: Playwright's page is a MagicMock, and for `download_incidence` the
dashboard steps are replaced with fakes that record the order they run in.
"""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from bcviz.acquire import ndrs
from bcviz.data.geography import GeoLevel


def write_download(path, codes, site="Breast", sex="Females", years="2018-2022"):
    """Write a file shaped like an NDRS download (two rows per area, header with a stray space)."""
    rows = [{"Geography code": c, "Geography name ": f"Area {c}", "NDRS main": site, "Gender": sex, "Year": years,
             "Type of rate": rate_type}
            for c in codes for rate_type in ("Non-standardised", "Age-standardised")]
    pd.DataFrame(rows).to_csv(path, index=False)


# --- _check_download ---
def check(path, n_areas=2, site="Breast", sex="Females", years="2018-2022"):
    """Helper to give default options to _check_download."""
    ndrs._check_download(path, n_areas=n_areas, site=site, sex=sex, years=years)


def test_check_download_accepts_matching_file(tmp_path):
    """Test that _check_download accepts a file matching the expected selection choices."""
    path = tmp_path / "d.csv"
    write_download(path, ["E02000001", "E02000002"])
    check(path)


@pytest.mark.parametrize("field, problem", [
    ("site", "NDRS main: expected 'Breast', found ['Lung']"),
    ("sex", "Gender: expected 'Females', found ['Persons']"),
    ("years", "Year: expected '2018-2022', found ['2013-2017']"),
])
def test_check_download_rejects_wrong_selection(tmp_path, field, problem):
    """Test that _check_download rejects a file with the wrong selection choices."""
    path = tmp_path / "d.csv"
    write_download(path, ["E02000001", "E02000002"], **{field: {"site": "Lung", "sex": "Persons",
                                                                 "years": "2013-2017"}[field]})
    with pytest.raises(RuntimeError) as error:
        check(path)
    assert problem in str(error.value)


def test_check_download_rejects_mixed_values(tmp_path):
    """Test that _check_download rejects a file with mixed values (some right, some wrong) in the selection choices."""
    path = tmp_path / "d.csv"
    write_download(path, ["E02000001"])
    data = pd.read_csv(path)
    data.loc[1, "Year"] = "2017-2021"
    data.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match=r"Year: expected '2018-2022', found \['2017-2021', '2018-2022'\]"):
        check(path, n_areas=1)


def test_check_download_rejects_wrong_number_of_areas(tmp_path):
    """Test that _check_download rejects a file with the wrong number of areas."""
    path = tmp_path / "d.csv"
    write_download(path, ["E02000001", "E02000002"])
    with pytest.raises(RuntimeError, match="expected 3 areas, found 2"):
        check(path, n_areas=3)


def test_check_download_lists_every_problem(tmp_path):
    """Test that _check_download lists every problem that occurs during checking."""
    path = tmp_path / "d.csv"
    write_download(path, ["E02000001"], site="Lung")
    with pytest.raises(RuntimeError, match=r"NDRS main: .*; expected 2 areas, found 1"):
        check(path)


# --- _options, _select, _wait_for_option ---
def test_options_reads_dropdown_values():
    """Test that _options reads the values of a dropdown."""
    page = MagicMock()
    page.evaluate.return_value = ["a", "b"]
    assert ndrs._options(page, "siteChoice") == ["a", "b"]
    assert page.evaluate.call_args.args[1] == "siteChoice"


def test_select_sets_values_and_waits_for_shiny():
    """
    Test that _select sets values and waits for Shiny's server to react, as it can take a while to register the choice
    of options.
    """
    page = MagicMock()
    page.evaluate.side_effect = [["Bolton 001", "Bolton 002", "Bury 001"], None]
    ndrs._select(page, "geogChoice", ["Bolton 001", "Bury 001"])

    set_call = page.evaluate.call_args_list[1]
    assert "selectpicker('val'" in set_call.args[0]
    assert set_call.args[1] == ["geogChoice", ["Bolton 001", "Bury 001"]]
    assert page.wait_for_function.call_args.kwargs["arg"] == ["geogChoice", 2]


def test_select_rejects_values_not_offered_before_selecting():
    """
    Test that _select rejects values that are not offered before selecting (so incorrect options never reach the
    dashboard).
    """
    page = MagicMock()
    page.evaluate.return_value = ["Bolton 001"]
    with pytest.raises(ValueError, match=r"Not offered in geogChoice: \['Nowhere 001'\]"):
        ndrs._select(page, "geogChoice", ["Bolton 001", "Nowhere 001"])
    assert page.evaluate.call_count == 1
    page.wait_for_function.assert_not_called()


def test_wait_for_option_waits_for_value():
    """Test that _wait_for_option waits for the expected value to appear."""
    page = MagicMock()
    ndrs._wait_for_option(page, "siteChoice", "Breast")
    assert page.wait_for_function.call_args.kwargs["arg"] == ["siteChoice", "Breast"]


def test_wait_for_option_reports_what_is_offered_on_timeout():
    """Test that _wait_for_option reports what is offered on timeout with an error."""
    page = MagicMock()
    page.wait_for_function.side_effect = PlaywrightTimeoutError("timed out")
    page.evaluate.return_value = ["Lung", "Prostate"]
    with pytest.raises(ValueError, match=r"'Breast' is not offered in siteChoice.*\['Lung', 'Prostate'\]"):
        ndrs._wait_for_option(page, "siteChoice", "Breast")


# --- dashboard steps ---
def test_open_tab_loads_dashboard_and_waits_for_area_selector():
    """Test that _open_tab loads the dashboard and waits for the area selector to appear."""
    page = MagicMock()
    ndrs._open_tab(page, "MSOA", "geogChoiceIncMSOA")
    assert page.goto.call_args.args[0] == ndrs.DASHBOARD_URL
    page.wait_for_selector.assert_called_with("#geogChoiceIncMSOA", state="attached")


def test_choose_five_year_bands_waits_for_year_selector():
    """Test that _choose_five_year_bands waits for the year selector to appear."""
    page = MagicMock()
    ndrs._choose_five_year_bands(page, "yearTypeChoice", "yearChoice5")
    assert "value='5-year'" in page.locator.call_args.args[0]
    page.wait_for_selector.assert_called_with("#yearChoice5", state="attached")


def test_apply_and_download_waits_for_new_table_before_downloading(tmp_path):
    """Test that _apply_and_download waits for the new table to appear before downloading."""
    page = MagicMock()
    page.locator.return_value.inner_text.return_value = "old table"
    ids = ndrs._TABS[GeoLevel.MSOA]
    out = tmp_path / "d.csv.partial"

    ndrs._apply_and_download(page, ids, "Breast", out)

    steps = [c for c in page.mock_calls if c[0] in ("locator", "click", "wait_for_function")]
    assert steps[0] == call.locator(f"#{ids['table']}")
    assert steps[1] == call.click(f"#{ids['apply']}")
    assert steps[2][0] == "wait_for_function"
    assert steps[2].kwargs["arg"] == [f"#{ids['table']}", "old table", "Breast"]
    assert steps[3] == call.click(f"#{ids['download']}")
    page.expect_download.return_value.__enter__.return_value.value.save_as.assert_called_once_with(out)


# --- download_incidence ---
class FakeDashboard:
    """
    A fake browser and replacements for the dashboard steps. `steps` records each step in order, and the fake download
    writes a file for whatever was selected; set `download_codes` to override the areas it contains.
    """
    def __init__(self, monkeypatch):
        self.steps = []
        self.launched_headless = None
        self.closed = False
        self.page = MagicMock()
        self.context_kwargs = None
        self.download_codes = None
        self.fail_on = None
        self.selected = {}

        monkeypatch.setattr(ndrs, "sync_playwright", self.sync_playwright)
        monkeypatch.setattr(ndrs, "_open_tab", lambda page, tab, areas_id: self.step("open", tab, areas_id))
        monkeypatch.setattr(ndrs, "_choose_five_year_bands",
                            lambda page, radio, years_id: self.step("five_year", radio, years_id))
        monkeypatch.setattr(ndrs, "_wait_for_option", lambda page, sid, value: self.step("wait", sid, value))
        monkeypatch.setattr(ndrs, "_select", self.select)
        monkeypatch.setattr(ndrs, "_apply_and_download", self.apply_and_download)

    def step(self, *record):
        """Record a step."""
        self.steps.append(record)
        if record[0] == self.fail_on:
            raise ValueError(f"{record[0]} failed")

    def select(self, page, select_id, values):
        """Select values in a dropdown."""
        self.selected[select_id] = values
        self.step("select", select_id, values)

    def apply_and_download(self, page, ids, site, out):
        """Apply changes to a selector and download the resulting table."""
        self.step("apply", site, out.name)
        [sex], [years] = self.selected[ids["sex"]], self.selected[ids["years"]]
        codes = self.download_codes or [f"E02{i:06d}" for i in range(len(self.selected[ids["areas"]]))]
        write_download(out, codes, site=site, sex=sex, years=years)

    @contextmanager
    def sync_playwright(self):
        """Simulate a Playwright browser."""
        def launch(headless):
            self.launched_headless = headless
            return browser

        def new_context(**kwargs):
            self.context_kwargs = kwargs
            return SimpleNamespace(new_page=lambda: self.page)

        def close():
            self.closed = True

        browser = SimpleNamespace(new_context=new_context, close=close)
        yield SimpleNamespace(chromium=SimpleNamespace(launch=launch))


@pytest.fixture
def dashboard(monkeypatch):
    return FakeDashboard(monkeypatch)


def test_download_msoa_makes_selections_in_order(dashboard, tmp_path):
    """Test that `download_incidence` makes the expected selections in order."""
    dest = tmp_path / "msoa.csv"
    result = ndrs.download_incidence(GeoLevel.MSOA, ["Bolton 002", "Bolton 001", "Bolton 002"], dest)

    ids = ndrs._TABS[GeoLevel.MSOA]
    assert dashboard.steps == [
        ("open", "MSOA", ids["areas"]),
        ("select", ids["sex"], ["Females"]),
        ("select", ids["years"], ["2018-2022"]),
        ("wait", ids["site"], "Breast"),
        ("select", ids["site"], ["Breast"]),
        ("select", ids["areas"], ["Bolton 001", "Bolton 002"]),  # repeats removed, sorted
        ("apply", "Breast", "msoa.csv.partial"),
    ]
    assert result == dest
    assert dest.exists()
    assert not (tmp_path / "msoa.csv.partial").exists()


def test_download_lad_chooses_five_year_bands_before_years(dashboard, tmp_path):
    """
    Test that `download_incidence` chooses five-year bands before years, as the pooling needs to be specified for the
    LAD level prior to selecting the specific 5-year band.
    """
    ndrs.download_incidence(GeoLevel.LAD, ["Bolton"], tmp_path / "lad.csv", years="2013-2017")
    ids = ndrs._TABS[GeoLevel.LAD]
    assert dashboard.steps[:4] == [
        ("open", "LAUA", ids["areas"]),
        ("select", ids["sex"], ["Females"]),
        ("five_year", ids["year_type"], ids["years"]),
        ("select", ids["years"], ["2013-2017"]),
    ]


def test_download_sets_up_browser(dashboard, tmp_path):
    """
    Test that `download_incidence` sets up the browser as expected, respecting headless setting, allowing for downloads,
    and closing the browser after timeout.
    """
    ndrs.download_incidence(GeoLevel.MSOA, ["Bolton 001"], tmp_path / "msoa.csv", headless=False)
    assert dashboard.launched_headless is False
    assert dashboard.context_kwargs == {"accept_downloads": True}
    dashboard.page.set_default_timeout.assert_called_once_with(ndrs.TIMEOUT_MS)
    assert dashboard.closed


def test_download_creates_destination_folder(dashboard, tmp_path):
    """Test that `download_incidence` creates the destination folder if it doesn't exist. Very simple."""
    dest = tmp_path / "new" / "folder" / "msoa.csv"
    ndrs.download_incidence(GeoLevel.MSOA, ["Bolton 001"], str(dest))
    assert dest.exists()


def test_download_keeps_only_partial_file_when_check_fails(dashboard, tmp_path):
    """Test that `download_incidence` keeps only the partial file when the check fails."""
    dashboard.download_codes = ["E02000001"]  # one area, though two were asked for
    dest = tmp_path / "msoa.csv"
    with pytest.raises(RuntimeError, match="expected 2 areas, found 1"):
        ndrs.download_incidence(GeoLevel.MSOA, ["Bolton 001", "Bolton 002"], dest)
    assert not dest.exists()
    assert (tmp_path / "msoa.csv.partial").exists()


def test_download_closes_browser_when_a_step_fails(dashboard, tmp_path):
    """Test that `download_incidence` closes the browser when a step fails."""
    dashboard.fail_on = "wait"
    dest = tmp_path / "msoa.csv"
    with pytest.raises(ValueError, match="wait failed"):
        ndrs.download_incidence(GeoLevel.MSOA, ["Bolton 001"], dest)
    assert dashboard.closed
    assert not dest.exists()


@pytest.mark.parametrize("level", [GeoLevel.OA, GeoLevel.LSOA])
def test_download_rejects_unsupported_geography_without_opening_browser(dashboard, tmp_path, level):
    """Test that `download_incidence` rejects unsupported geographies without opening the browser."""
    with pytest.raises(ValueError, match="only downloaded for \\['MSOA', 'LAD'\\]"):
        ndrs.download_incidence(level, ["x"], tmp_path / "x.csv")
    assert dashboard.launched_headless is None
