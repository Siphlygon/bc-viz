"""
Download cancer incidence data from the NDRS (National Disease Registration Service) incidence and mortality dashboard.

NDRS publishes no API for its small-area incidence figures, and its publication pages sit behind a bot check that blocks
scripted downloads. This module therefore drives the public dashboard (an R Shiny app) with a headless browser using
Playwright: it makes the same selections a person would, clicks "Apply changes", and saves the file from the "Download"
button exactly as provided.

Only 5-year diagnosis bands are supported. They are the only option at MSOA level, and using them for districts too
keeps the two geographies comparable. NDRS rounds MSOA counts to the nearest 5; district (LAD) counts are not rounded.

Playwright needs a browser installed once, with:
    playwright install chromium
"""
from pathlib import Path

import pandas as pd
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from ..data.geography import GeoLevel
from ..utils.logger import LoggingLevels, get_logger

# The public NDRS dashboard
DASHBOARD_URL = "https://nhsd-ndrs.shinyapps.io/incidence_and_mortality/"

# Milliseconds to wait for the dashboard; it can take a while to start up and to build large tables
TIMEOUT_MS = 120_000

# For each geography: the dashboard's name for its tab on the Incidence page, and the element ids of the controls on
# that tab. Only the district tab offers a choice of year band, so only it has a year-type control.
_TABS = {
    GeoLevel.MSOA: {
        "tab": "MSOA",
        "areas": "geogChoiceIncMSOA",
        "sex": "genderChoiceIncMSOA",
        "year_type": None,
        "years": "yearChoiceInc5MSOA",
        "site": "siteChoiceIncMSOA",
        "apply": "goButtonMSOA",
        "download": "downloadButtonMSOA",
        "table": "incidenceTableMSOA",
    },
    GeoLevel.LAD: {
        "tab": "LAUA",
        "areas": "geogChoiceIncLAUA",
        "sex": "genderChoiceIncLAUA",
        "year_type": "yearTypeChoiceIncLAUA",
        "years": "yearChoiceInc5LAUA",  # only appears once 5-year bands are chosen
        "site": "siteChoiceIncLAUA",
        "apply": "goButtonLAUA",
        "download": "downloadButtonLAUA",
        "table": "incidenceTableLAUA",
    },
}

# Make a logger for this module
logger = get_logger("bcviz.acquire.ndrs", LoggingLevels.INFO)


def download_incidence(
    geography: GeoLevel,
    areas: list[str],
    dest: str | Path,
    site: str = "Breast",
    sex: str = "Females",
    years: str = "2018-2022",
    headless: bool = True,
    ) -> Path:
    """
    Download cancer incidence counts and rates for the given areas from the NDRS dashboard.

    The file is saved exactly as the dashboard provides it. It is first written to '<dest>.partial' and only renamed to
    `dest` once it has been checked against the request, so a failed download is never left at `dest`.

    Parameters
    ----------
    geography : GeoLevel
        GeoLevel.MSOA or GeoLevel.LAD.
    areas : list[str]
        Area names as the dashboard lists them, e.g. "Bolton 001" for an MSOA or "Bolton" for a district. The dashboard
        selects areas by name, not by code.
    dest : str | Path
        Path to save the CSV to.
    site : str
        Cancer site as named in the dashboard, by default "Breast".
    sex : str
        "Females", "Males" or "Persons", by default "Females".
    years : str
        5-year diagnosis band, e.g. "2013-2017"; by default "2018-2022".
    headless : bool
        If False, show the browser window so each step can be watched, by default True.

    Returns
    -------
    Path
        Path of the saved CSV.

    Raises
    ------
    ValueError
        If `geography` is not MSOA or LAD, or a requested value is not offered by the dashboard.
    RuntimeError
        If the downloaded file does not match the request.
    """
    if geography not in _TABS:
        raise ValueError(f"NDRS incidence is only downloaded for {[g.name for g in _TABS]}, not {geography.name}")
    ids = _TABS[geography]
    areas = sorted(set(areas))  # the dashboard ignores repeats, so they would never be registered as chosen
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")

    logger.info(f"Downloading NDRS incidence ({site}, {sex}, {years}) for {len(areas)} {geography.name}s to {dest}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            page = browser.new_context(accept_downloads=True).new_page()
            page.set_default_timeout(TIMEOUT_MS)
            _open_tab(page, ids["tab"], ids["areas"])

            # Order matters: the list of cancer sites only fills in once sex and the year band have been chosen
            _select(page, ids["sex"], [sex])
            if ids["year_type"]:
                _choose_five_year_bands(page, ids["year_type"], ids["years"])
            _select(page, ids["years"], [years])
            _wait_for_option(page, ids["site"], site)
            _select(page, ids["site"], [site])
            _select(page, ids["areas"], areas)

            _apply_and_download(page, ids, site, partial)
        finally:
            browser.close()

    _check_download(partial, n_areas=len(areas), site=site, sex=sex, years=years)
    partial.replace(dest)
    return dest


# --- DASHBOARD STEPS ---
def _open_tab(page: Page, tab: str, areas_id: str) -> None:
    """
    Load the dashboard and open its Incidence page at the given geography tab.

    Parameters
    ----------
    page : Page
        The browser page to use.
    tab : str
        The dashboard's name for the geography tab, e.g. "MSOA".
    areas_id : str
        Element id of the tab's area selector, which is waited for to know the tab has loaded.
    """
    page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
    # Shiny apps never stop network activity, so wait for the app itself to report that it is connected
    page.wait_for_function("() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()")
    for name in ("Incidence", tab):
        page.locator("a[data-toggle='tab']").filter(has_text=name).locator("visible=true").first.click()
    page.wait_for_selector(f"#{areas_id}", state="attached")


def _choose_five_year_bands(page: Page, radio_name: str, years_id: str) -> None:
    """
    Click the '5-year' button of a year-type control, then wait for the matching list of year bands to appear.

    Parameters
    ----------
    page : Page
        The browser page to use.
    radio_name : str
        Name of the year-type radio inputs.
    years_id : str
        Element id of the year-band selector that appears once 5-year bands are chosen.
    """
    radio = page.locator(f"input[name='{radio_name}'][value='5-year']")
    # The radio input is hidden inside a styled button, which is what receives the click
    radio.locator("xpath=ancestor::*[contains(@class, 'btn')][1]").click()
    page.wait_for_selector(f"#{years_id}", state="attached")


def _apply_and_download(page: Page, ids: dict, site: str, out: Path) -> None:
    """
    Click 'Apply changes', wait until the results table has been rebuilt, then save the file from 'Download'.

    The Download button serves whatever result the server currently holds, so clicking it before the table updates
    could save the previous selection.

    Parameters
    ----------
    page : Page
        The browser page to use.
    ids : dict
        Element ids for the current tab, from `_TABS`.
    site : str
        The chosen cancer site, which must appear in the rebuilt table.
    out : Path
        Path to save the downloaded file to.
    """
    table = f"#{ids['table']}"
    before = page.locator(table).inner_text()
    page.click(f"#{ids['apply']}")
    page.wait_for_function(
        "([sel, before, site]) => { const t = document.querySelector(sel);"
        " return t && t.innerText !== before && t.innerText.includes(site); }",
        arg=[table, before, site],
    )
    with page.expect_download() as download:
        page.click(f"#{ids['download']}")
    download.value.save_as(out)


def _check_download(path: Path, n_areas: int, site: str, sex: str, years: str) -> None:
    """
    Check the downloaded file contains what was requested; the file is left in place for inspection if not.

    Parameters
    ----------
    path : Path
        Path of the downloaded file.
    n_areas : int
        Number of areas requested.
    site, sex, years : str
        The requested cancer site, sex and year band.

    Raises
    ------
    RuntimeError
        If the file's contents do not match the request.
    """
    data = pd.read_csv(path, dtype=str)
    data.columns = data.columns.str.strip()  # NDRS's header has stray spaces, e.g. "Geography name "

    problems = []
    for column, expected in (("NDRS main", site), ("Gender", sex), ("Year", years)):
        found = set(data[column])
        if found != {expected}:
            problems.append(f"{column}: expected {expected!r}, found {sorted(found)}")
    n_found = data["Geography code"].nunique()
    if n_found != n_areas:
        problems.append(f"expected {n_areas} areas, found {n_found}")
    if problems:
        raise RuntimeError(f"Download from NDRS did not match the request ({path}): " + "; ".join(problems))


# --- UTILITY ---
def _options(page: Page, select_id: str) -> list[str]:
    """
    Get the values offered by one of the dashboard's dropdowns.
    """
    return page.evaluate("id => [...document.getElementById(id).options].map(o => o.value)", select_id)


def _select(page: Page, select_id: str, values: list[str]) -> None:
    """
    Choose `values` in one of the dashboard's dropdowns, and wait until the app has registered them.

    Values are checked against the dropdown first: an unoffered value would otherwise only show up as a timeout, without
    saying which of possibly hundreds of values was wrong.

    Parameters
    ----------
    page : Page
        The browser page to use.
    select_id : str
        Element id of the dropdown.
    values : list[str]
        Values to choose.

    Raises
    ------
    ValueError
        If any value is not offered by the dropdown.
    """
    offered = _options(page, select_id)
    missing = [v for v in values if v not in offered]
    if missing:
        raise ValueError(f"Not offered in {select_id}: {missing}. Offered (first 30): {offered[:30]}")

    # The dropdowns are bootstrap-select widgets; setting them through their own jQuery plugin fires the same events as a
    # person clicking. Shiny's record of the inputs then confirms the choice has reached the server.
    page.evaluate("([id, v]) => jQuery('#' + id).selectpicker('val', v)", [select_id, values])
    page.wait_for_function(
        "([id, n]) => { const v = Shiny.shinyapp.$inputValues[id];"
        " return Array.isArray(v) ? v.length === n : v != null && n === 1; }",
        arg=[select_id, len(values)],
    )


def _wait_for_option(page: Page, select_id: str, value: str) -> None:
    """
    Wait for a dropdown to offer `value`, for dropdowns whose options depend on earlier choices.

    Raises
    ------
    ValueError
        If `value` has not been offered after 30 seconds, listing what is offered instead.
    """
    try:
        page.wait_for_function(
            "([id, v]) => [...document.getElementById(id).options].some(o => o.value === v)",
            arg=[select_id, value],
            timeout=30_000,
        )
    except PlaywrightTimeoutError:
        raise ValueError(f"{value!r} is not offered in {select_id} for these choices. "
                         f"Offered: {_options(page, select_id)}") from None


if __name__ == "__main__":
    # Example usage
    from . import ndrs
    from ..data.geography import GeoLevel, GeoHierarchy
    import os

    geoh = GeoHierarchy.from_ons_lookup()
    bolton_code = geoh.get_lad_code("Bolton")
    msoas = geoh.areas_in(bolton_code, GeoLevel.MSOA)
    msoa_names = [msoa.name for msoa in msoas]

    ndrs.download_incidence(
        geography = GeoLevel.MSOA,
        areas = msoa_names,
        dest = "bc-viz/acquire/temp.csv",
        site = "Breast",
        sex = "Females",
        years = "2018-2022",
        headless = True
    )

    temp = pd.read_csv("bc-viz/acquire/temp.csv")
    print(temp.head())
    os.remove("bc-viz/acquire/temp.csv")
