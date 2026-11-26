"""
A module for working with geographical data in the bcviz project.
"""
import os
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import overload

import pandas as pd

from ..acquire import arcgis_rest as ag
from ..utils.logger import LoggingLevels, get_logger

# Every LAD-level geography in Greater Manchester, for use in filtering data to only include these areas. This is not
# expected to ever change. The names are connected to their respective LAD code
GM_DISTRICTS = {
    "Bolton": "E08000001",
    "Bury": "E08000002",
    "Manchester": "E08000003",
    "Oldham": "E08000004",
    "Rochdale": "E08000005",
    "Salford": "E08000006",
    "Stockport": "E08000007",
    "Tameside": "E08000008",
    "Trafford": "E08000009",
    "Wigan": "E08000010",
}

# The name & ID of the API endpoint providing the ONS lookup table, and its local name in our project.
GEO_LOOKUP_PATH = "data/ons_lookup.csv"
GEO_LOOKUP_NAME = "Postcode to OA (2021) to LSOA to MSOA to LAD (May 2026) Best Fit Lookup in the UK"
GEO_LOOKUP_ID = "2713c3a627454ff88add814226264d77"

class GeoLevel(IntEnum):
    """
    Geography levels supported by this project, ordered from finest to coarsest (so OA < MSOA).
    
    Allowed values are:
    - OA: Output Area  (Census)
    - LSOA: Lower Layer Super Output Area  (Census; ~3-4 OAs)
    - MSOA: Middle Layer Super Output Area  (Census; ~5-6 LSOAs)
    - LAD: Local Authority District (Government; also known as LAUA or LTLA)

    Each unit has a geographical code (e.g., S00137176) and a name (e.g., Bolton 016E).
    """
    OA = 1
    LSOA = 2
    MSOA = 3
    LAD = 4

    @property
    def code_col(self) -> str:
        return f"{self.name.lower()}_code"

    @property
    def name_col(self) -> str:
        return f"{self.name.lower()}_name"


@dataclass
class GeoArea:
    """
    A simple dataclass containing information on a specific area, including its name, code, and GeoLevel.

    OAs do not have names, hence that column being optional.
    
    Example:
        name = "Bolton 016"
        code = "E02000999"
        level = GeoLevel.MSOA
    """
    name: str | None
    code: str
    level: GeoLevel

    def __post_init__(self):
        if self.name is None:
            self.name = "OA"  # overwrite the None with just a default "OA", which may help in future debugging

    def __str__(self):
        return f"name: {self.name}, code: {self.code}, level: {self.level}"


def _get_geo_lookup_table() -> pd.DataFrame:
    """
    Get a lookup table of geographical areas in the UK directly from the ONS Open Geography portal, which allows for
    exact matching of areas to their parent geographies.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows represent geographical areas and columns represent their properties, including their
        names and parent geography codes.
    """
    # Download the ONS lookup table if it doesn't already exist
    zip_path = GEO_LOOKUP_PATH + ".zip"
    if not os.path.exists(zip_path):
        logger = get_logger(__name__, LoggingLevels.INFO.value)
        logger.info(f"Downloading ONS lookup table to {GEO_LOOKUP_PATH}...")
        ag.download_item(GEO_LOOKUP_ID, GEO_LOOKUP_PATH + ".zip", expected_title=GEO_LOOKUP_NAME)

    if not os.path.exists(GEO_LOOKUP_PATH):
        # Unzip the downloaded file and read it into a DataFrame, dropping unnecessary columns for file-size
        lookup_table = pd.read_csv(zip_path, compression="zip")
        lookup_table = lookup_table.drop(columns=[
            "pcd7", "pcd8", # 7 and 8 char versions of postcodes; here for older data, "pcds" is the standard way
            "usertype",     # "1" is a "large user" and 0 is a "small user", purely based off Royal Mail post volume
            "ladnmw"        # the name of the LAD in Welsh; we don't need this for our purposes & blank mostly anyway
        ])
        lookup_table.to_csv(GEO_LOOKUP_PATH, index=False)
        return lookup_table

    return pd.read_csv(GEO_LOOKUP_PATH)


class GeoHierarchy:
    """
    A lookup table for nested geographical hierarchies in a pandas DataFrame representing a larger area.

    For example, in Greater Manchester, this would contain all the OAs, LSOAs, MSOAs, and LADs in the area, along with
    their names and parent geography codes. This allows for easy filtering of data to only include areas within a
    specific district or geography level.
    """
    def __init__(self, table: pd.DataFrame, filter_districts: set["str"] | None = set(GM_DISTRICTS.keys())):
        """
        Set up a GeoHierarchy object with a lookup table of geographical areas. Optionally filter to only include areas
        within Greater Manchester.

        Parameters
        ----------
        table : pd.DataFrame
            A DataFrame where rows represent geographical areas and columns represent their properties.
        filter_districts : bool, optional
            A list of districts to filter the table to only include data for, by default only the LADs within Greater
            Manchester. If None, will not perform any filtering.
        """
        self.table = table
        if filter_districts:
            self.table = self.table[self.table["lad_name"].isin(filter_districts)]

    @classmethod
    def from_ons_lookup(cls, lookup: pd.DataFrame | None = None) -> "GeoHierarchy":
        """
        Create a GeoHierarchy object from an ONS lookup table (for OA to LSOA to MSOA to District).

        If no lookup table is provided, the function will attempt to use a local version from the ONS Open Geography
        portal, downloading it if necessary.

        Parameters
        ----------
        lookup : pd.DataFrame | None
            A DataFrame where rows represent geographical areas and columns represent their properties. If None, the
            function will attempt to load a local version of the ONS lookup table.

        Returns
        -------
        GeoHierarchy
            A GeoHierarchy object representing the nested geographical hierarchy.
        """
        if lookup is None:
            lookup = _get_geo_lookup_table()

        # Tables downloaded from ONS will have specific years attached to the column names, so we need to strip those
        # off & rename the columns to get the standard column names we expect.
        rename = {}
        for column in lookup.columns:
            # Split the column name into its base name ("lsoa") and kind ("cd" for code or "nm" for name), ignoring any
            # year suffixes.
            match = re.fullmatch(r"(OA|LSOA|MSOA|LAD)\d*(CD|NM)", column, flags=re.IGNORECASE)
            if match:
                level, kind = GeoLevel[match.group(1).upper()], match.group(2).upper()
                rename[column] = level.code_col if kind == "CD" else level.name_col
        return cls(lookup.rename(columns=rename))


    # --- RETRIEVE INFORMATION ---
    def get_parent(self, area_code: str, parent_level: GeoLevel) -> GeoArea:
        """
        Get the parent area code of a given area code at a specified parent geography level.

        Parameters
        ----------
        area_code : str
            The code of the area for which to find the parent, e.g., "E010000001"
        parent_level : GeoLevel
            The geography level of the parent area.

        Returns
        -------
        str
            The code of the parent area at the specified level.
        
        Raises
        ------
        ValueError
            `area_code` refers to an LAD (no parents), or there is no row in the table with `area_code`.
        """
        if parent_level not in GeoLevel:
            raise ValueError(f"Invalid parent level: {parent_level}. Must be one of {list(GeoLevel)}.")

        # Find the level of the area code we're given
        level = self._get_level_from_code(area_code)
        if level == GeoLevel.LAD:
            raise ValueError("LADs do not have any parent areas in this database.")

        if not parent_level > level:
            raise ValueError(f"{parent_level} are not parents of {level}.")

        
        # Get the row corresponding to the given area code
        row = self.table[self.table[level.code_col] == area_code]
        if row.empty:
            raise ValueError(f"No area found with code: {area_code}")
        return self.to_geo_area(row[parent_level.code_col].values[0])

    def areas_in(self, area_code: str, child_level: GeoLevel) -> list[GeoArea]:
        """
        Get all the GeoAreas of `child_level` inside `area_code`.

        Parameters
        ----------
        area_code : str
            The area code of the target area.
        child_level : GeoLevel
            The `GeoLevel` of the specific children you want to find (e.g., GeoLevel.LSOA for a MSOA code)

        Returns
        -------
        list[GeoArea]
            A list of GeoArea all of `child_level` who have `area_code` as a parent.

        Raises
        ------
        ValueError
            If `area_code` refers to an OA, which has no children, or if no values can be found for `area_code`.
        """
        # Get the level of the area code entered
        level = self._get_level_from_code(area_code)
        if level == GeoLevel.OA:
            raise ValueError("OAs do not have any child areas inside them.")

        if not child_level < level:
            raise ValueError(f"{child_level} are not children of {level}.")

        rows = self.table[self.table[level.code_col] == area_code]
        if rows.empty:
            raise ValueError(f"No area found with code: {area_code}")
        return self.to_geo_area(list(set(rows[child_level.code_col].values)))

    def get_areas_in_gm(self, geo_level: GeoLevel) -> list[GeoArea]:
        """
        Get all the GeoAreas of `geo_level` inside Greater Manchester (combined authority).

        This is purely a convenience function, as areas within Greater Manchester are expected to be the main focus
        of this project.

        Parameters
        ----------
        geo_level : GeoLevel
            The `GeoLevel` of the GeoAreas in GM you want to find (e.g., GeoLevel.MSOA for MSOAs)

        Returns
        -------
        list[GeoArea]
            A list of GeoArea all of `geo_level` who are in Greater Manchester.
        """
        assert geo_level in GeoLevel, f"Invalid geo level: {geo_level}. Must be one of {list(GeoLevel)}."
        if geo_level == GeoLevel.LAD:
            return [GeoArea(name=lad_name,
                            code=lad_code,
                            level=GeoLevel.LAD) for lad_name, lad_code in GM_DISTRICTS.items()]

        # Exact same process for each census statistical geography
        gm_lad_codes = GM_DISTRICTS.values()
        census_areas = [self.areas_in(code, geo_level) for code in gm_lad_codes] # find all GeoAreas for each LAD
        return [item for sublist in census_areas for item in sublist] # unpacking to 1d list


    # --- UTILITY ---
    def _get_level_from_code(self, area_code: str) -> "GeoLevel":
        """
        Area codes for the different geographical areas follow strict allowance rules, which therefore allows the
        automatic determination of which area a specific code is describing, notably:

        * OAs are always in the form "E00" (3rd digit is always 0)
        * LSOAs are always in the form "E01" (3rd digit is always 1)
        * MSOAs are always in the form "E02" (3rd digit is always 2)
        * LADs are (for GM) usually "E06", "E07, "E08", or "E09" (3rd digit is always in [6, 7, 8, 9])

        Parameters
        ----------
        area_code : str
            The code of a specific area.

        Returns
        -------
        GeoLevel
            The geography level of the area.
        
        Raises
        ------
        ValueError
            The code cannot be identified using the above criteria on its 3rd digit.
        """
        match area_code[2]:
            case "0": return GeoLevel.OA
            case "1": return GeoLevel.LSOA
            case "2": return GeoLevel.MSOA
            case "6" | "7" | "8" | "9": return GeoLevel.LAD
            case _:
                raise ValueError(f"Cannot identify the level of area code: {area_code}")

    @overload
    def to_geo_area(self, area_code: str) -> GeoArea: ...
    @overload
    def to_geo_area(self, area_code: list[str]) -> list[GeoArea]: ...
    def to_geo_area(self, area_code):
        """
        Converts area code(s) into a GeoArea(s) using values from the table.

        Parameters
        ----------
        area_code : str | list[str]
            The target area code(s).

        Returns
        -------
        GeoArea | list[GeoArea]
            The geographical area(s) represented by the code(s).

        Raises
        ------
        ValueError
            If an area code cannot be identified or is not present in the table.
        """
        geo_areas = []
        single_value = False

        if isinstance(area_code, str):
            area_code = [area_code]
            single_value = True
        
        for code in area_code:
            level = self._get_level_from_code(code)
            rows = self.table[self.table[level.code_col] == code]
            if rows.empty:
                raise ValueError(f"No area found with code: {code}")

            name = rows[level.name_col].values[0]
            geo_areas.append(GeoArea(name=name, code=code, level=level))
        return geo_areas if not single_value else geo_areas[0]

    def get_lad_code(self, lad_name: str) -> str:
        """
        Converts a common-name for an LAD (e.g., "Bolton") into its area code using GM_DISTRICTS.

        Parameters
        ----------
        lad_name : str
            The name of a LAD in Greater Manchester.

        Returns
        -------
        str
            The area code corresponding to the LAD.
        
        Raises
        ------
        ValueError
            If the LAD name is not in GM_DISTRICTS.
        """
        # Convert to CapitalCase style (bolton --> Bolton)
        lad_name = lad_name[0].upper() + lad_name[1:].lower() 
        if lad_name not in GM_DISTRICTS:
            raise ValueError(f"{lad_name} is not a valid name for a LAD in Greater Manchester")
        return GM_DISTRICTS[lad_name]


if __name__ == "__main__":
    # Example usage of the GeoHierarchy class
    geo_hierarchy = GeoHierarchy.from_ons_lookup()
    print(geo_hierarchy.table.head())
