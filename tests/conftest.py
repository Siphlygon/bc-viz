"""
Shared fixtures for the bcviz test suite.

Every test runs in its own empty working directory, with the network and the browser blocked, so nothing can read the
real `data/` folder or reach ONS/NDRS by accident. Tests that need a download replace the relevant function themselves.
"""
import pandas as pd
import pytest
import requests

from bcviz.acquire import ndrs
from bcviz.data import geography
from bcviz.data.geography import GM_DISTRICTS, GeoHierarchy

# A district outside Greater Manchester, to check filtering keeps it out
NON_GM_DISTRICT = ("Blackburn with Darwen", "E06000008")


def _blocked(*args, **kwargs):
    raise AssertionError("Tests must not touch the network or a real browser; patch the call instead.")


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Run in an empty directory, and make any real HTTP request or browser launch fail loudly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    monkeypatch.setattr(ndrs, "sync_playwright", _blocked)


@pytest.fixture
def ons_lookup() -> pd.DataFrame:
    """
    A small lookup table laid out like the real ONS postcode lookup (after its unused columns are dropped).

    Every GM district is present, because `get_areas_in_gm` looks up all ten, plus one district outside GM. Each
    district has 2 MSOAs, each MSOA 2 LSOAs, each LSOA 2 OAs, and each OA 2 postcodes, so every area appears on several
    rows as it does in the real table. Like the real table, OAs have codes but no names.
    """
    rows = []
    districts = [*GM_DISTRICTS.items(), NON_GM_DISTRICT]
    for d, (lad_name, lad_code) in enumerate(districts):
        for m in range(2):
            msoa_code = f"E02{d:04d}{m:02d}"
            msoa_name = f"{lad_name} {m + 1:03d}"
            for l in range(2):
                lsoa_code = f"E01{d:04d}{m}{l}"
                lsoa_name = f"{msoa_name}{'AB'[l]}"
                for o in range(2):
                    oa_code = f"E00{d:03d}{m}{l}{o}"
                    for p in range(2):
                        rows.append({
                            "pcds": f"XX{d} {m}{l}{o}{p}",
                            "oa21cd": oa_code,
                            "lsoa21cd": lsoa_code,
                            "msoa21cd": msoa_code,
                            "ladcd": lad_code,
                            "lsoa21nm": lsoa_name,
                            "msoa21nm": msoa_name,
                            "ladnm": lad_name,
                        })
    return pd.DataFrame(rows)


@pytest.fixture
def non_gm_district() -> tuple[str, str]:
    """(name, code) of the one district in `ons_lookup` outside GM."""
    return NON_GM_DISTRICT


@pytest.fixture
def hierarchy(ons_lookup) -> GeoHierarchy:
    """A GeoHierarchy over `ons_lookup`, filtered to GM as by default."""
    return GeoHierarchy.from_ons_lookup(ons_lookup)


@pytest.fixture
def local_lookup(ons_lookup, monkeypatch) -> pd.DataFrame:
    """Make `GeoHierarchy.from_ons_lookup()` (with no argument) use `ons_lookup` instead of the real ONS file."""
    monkeypatch.setattr(geography, "_get_geo_lookup_table", lambda: ons_lookup.copy())
    return ons_lookup


@pytest.fixture
def gm_msoas(ons_lookup) -> pd.DataFrame:
    """Unique (code, name) of the MSOAs inside GM in `ons_lookup`, sorted by code."""
    gm = ons_lookup[ons_lookup["ladnm"].isin(GM_DISTRICTS)]
    return gm[["msoa21cd", "msoa21nm"]].drop_duplicates().sort_values("msoa21cd").reset_index(drop=True)
