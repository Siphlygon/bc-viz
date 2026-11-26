"""
Tests for bcviz.acquire.arcgis_rest. `requests` is replaced by small fakes, so the tests describe what the module sends
and how it reads the answers, without contacting ArcGIS.
"""
import geopandas as gpd
import pandas as pd
import pytest
import requests

from bcviz.acquire import arcgis_rest as ag

# Fake layer URL
LAYER = f"{ag.ONS_SERVICES}/Some_Service/FeatureServer/0"


class FakeResponse:
    """Stands in for a `requests.Response`: constructs a response with the given body and status."""
    def __init__(self, body=None, status=200, chunks=()):
        self.body = body
        self.status = status
        self.chunks = chunks

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.body

    def iter_content(self, chunk_size):
        return iter(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePost:
    """Stands in for `requests.post`: answers with `respond(url, data)` and records each call."""
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def __call__(self, url, data, timeout):
        self.calls.append({"url": url, "data": data, "timeout": timeout})
        return FakeResponse(self.respond(url, data))


@pytest.fixture
def post(monkeypatch):
    """Install a fake `requests.post` with the given answering function, and return it to inspect its calls."""
    def install(respond):
        fake = FakePost(respond)
        monkeypatch.setattr(ag.requests, "post", fake)
        return fake
    return install


class FakeLayer:
    """
    Answers like an ArcGIS layer holding `features`: its info, a count, and pages of at most `max_record_count`.

    `short_pages` makes every query return nothing, as a busy server sometimes does.
    """
    def __init__(self, features, max_record_count=2, object_id_field="FID", short_pages=False):
        self.features = features
        self.info = {"objectIdField": object_id_field, "maxRecordCount": max_record_count}
        self.short_pages = short_pages

    def __call__(self, url, data):
        if not url.endswith("/query"):
            return self.info
        if data.get("returnCountOnly") == "true":
            return {"count": len(self.features)}
        if self.short_pages:
            return {"features": []}
        start = data["resultOffset"]
        return {"features": self.features[start:start + data["resultRecordCount"]]}


def query_calls(fake):
    """The paged query requests a FakePost received (not the info or count requests)."""
    return [c for c in fake.calls if c["url"].endswith("/query") and "returnCountOnly" not in c["data"]]


def table_rows(n):
    return [{"attributes": {"FID": i, "CODE": f"E0{i}", "NAME": f"Area {i}"}} for i in range(n)]


def geojson_features(n):
    return [
        {"type": "Feature",
         "properties": {"FID": i, "CODE": f"E0{i}", "NAME": f"Area {i}"},
         "geometry": {"type": "Point", "coordinates": [380_000 + i, 400_000 + i]}}
        for i in range(n)
    ]


# --- _request ---
def test_request_posts_json_format_and_params(post):
    """Test that _request sends the right POST data and returns JSON according to the params."""
    fake = post(lambda url, data: {"ok": True})
    assert ag._request("https://example.test/x", {"where": "1=1"}) == {"ok": True}
    assert fake.calls == [{"url": "https://example.test/x", "data": {"f": "json", "where": "1=1"},
                           "timeout": ag.TIMEOUT}]


def test_request_params_can_override_format(post):
    """Test that params given to _request override the default format."""
    fake = post(lambda url, data: {})
    ag._request("https://example.test/x", {"f": "geojson"})
    assert fake.calls[0]["data"]["f"] == "geojson"


def test_request_raises_on_arcgis_error_with_details(post):
    """Test that _request raises on ArcGIS errors, with details if available."""
    post(lambda url, data: {"error": {"code": 400, "message": "Bad", "details": ["Invalid field", "Try again"]}})
    with pytest.raises(RuntimeError, match="Invalid field; Try again"):
        ag._request("https://example.test/x", {})


def test_request_falls_back_to_error_message_without_details(post):
    """Test that _request falls back to the error message if there are no details."""
    post(lambda url, data: {"error": {"code": 400, "message": "Bad request", "details": []}})
    with pytest.raises(RuntimeError, match="Bad request"):
        ag._request("https://example.test/x", {})


def test_request_raises_on_http_error(monkeypatch):
    """Test that _request raises on HTTP errors (e.g., 500)."""
    monkeypatch.setattr(ag.requests, "post", lambda url, data, timeout: FakeResponse({}, status=500))
    with pytest.raises(requests.HTTPError):
        ag._request("https://example.test/x", {})


# --- ons_layer_url ---
def test_ons_layer_url_with_explicit_layer_makes_no_request(post):
    """Test that ons_layer_url with an explicit layer makes no request -- it just returns the URL."""
    fake = post(lambda url, data: pytest.fail("should not query the service"))
    assert ag.ons_layer_url("Some_Service", layer=2) == f"{ag.ONS_SERVICES}/Some_Service/FeatureServer/2"
    assert fake.calls == []


@pytest.mark.parametrize("contents", [
    {"layers": [{"id": 3, "name": "boundaries"}], "tables": []},
    {"layers": [], "tables": [{"id": 3, "name": "lookup"}]},
    {"tables": [{"id": 3, "name": "lookup"}]},              # key missing entirely
    {"layers": None, "tables": [{"id": 3, "name": "lookup"}]},  # key present but null
])
def test_ons_layer_url_uses_the_only_layer_or_table(post, contents):
    """Test that ons_layer_url uses the only layer or table in the service."""
    fake = post(lambda url, data: contents)
    assert ag.ons_layer_url("Some_Service") == f"{ag.ONS_SERVICES}/Some_Service/FeatureServer/3"
    assert fake.calls[0]["url"] == f"{ag.ONS_SERVICES}/Some_Service/FeatureServer"


def test_ons_layer_url_with_several_layers_lists_them(post):
    """Test that ons_layer_url lists several layers in an error message if there are several."""
    post(lambda url, data: {"layers": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}], "tables": []})
    with pytest.raises(ValueError, match=r"2 layers \(0: a, 1: b\)"):
        ag.ons_layer_url("Some_Service")


def test_ons_layer_url_with_no_layers_raises(post):
    """Test that ons_layer_url raises if there are no layers in the service."""
    post(lambda url, data: {"layers": [], "tables": []})
    with pytest.raises(ValueError, match="0 layers"):
        ag.ons_layer_url("Some_Service")


# --- list_services, fields, count ---
def test_list_services_returns_sorted_unique_feature_services(post):
    """Test that list_services returns sorted, unique, feature services."""
    post(lambda url, data: {"services": [
        {"name": "MSOA_b", "type": "FeatureServer"},
        {"name": "MSOA_a", "type": "FeatureServer"},
        {"name": "MSOA_a", "type": "FeatureServer"},
        {"name": "MSOA_map", "type": "MapServer"},
    ]})
    assert ag.list_services() == ["MSOA_a", "MSOA_b"]


def test_list_services_filters_case_insensitively(post):
    """Test that list_services filters case-insensitively."""
    post(lambda url, data: {"services": [
        {"name": "MSOA_2021", "type": "FeatureServer"},
        {"name": "LAD_2021", "type": "FeatureServer"},
    ]})
    assert ag.list_services("msoa") == ["MSOA_2021"]


def test_list_services_queries_given_base(post):
    """Test that list_services queries the given base if non-default (not ONS_SERVICES)."""
    fake = post(lambda url, data: {"services": []})
    ag.list_services(base="https://example.test/services")
    assert fake.calls[0]["url"] == "https://example.test/services"


def test_fields_returns_field_names_in_order(post):
    """Test that fields returns field names in order."""
    post(lambda url, data: {"fields": [{"name": "FID"}, {"name": "CODE"}, {"name": "NAME"}]})
    assert ag.fields(LAYER) == ["FID", "CODE", "NAME"]


def test_count_asks_for_count_only(post):
    """Test that count asks for a count only of records matching the where clause."""
    fake = post(lambda url, data: {"count": 42})
    assert ag.count(LAYER, "CODE = 'x'") == 42
    assert fake.calls[0]["url"] == f"{LAYER}/query"
    assert fake.calls[0]["data"]["where"] == "CODE = 'x'"
    assert fake.calls[0]["data"]["returnCountOnly"] == "true"


# --- where_in ---
def test_where_in_quotes_values():
    """Test that where_in quotes values, as they need to be in string form for the query URL."""
    assert ag.where_in("LAD22CD", ["E08000001", "E08000002"]) == "LAD22CD IN ('E08000001','E08000002')"


def test_where_in_escapes_apostrophes():
    """Test that where_in escapes apostrophes in strings, as they are special characters."""
    assert ag.where_in("NAME", ["King's Lynn"]) == "NAME IN ('King''s Lynn')"


def test_where_in_accepts_any_iterable_and_non_strings():
    """Test that where_in accepts any iterable and non-strings, converting them to strings."""
    assert ag.where_in("ID", (n for n in [1, 2])) == "ID IN ('1','2')"


def test_where_in_needs_a_value():
    """Test that where_in needs at least one value to be in the list."""
    with pytest.raises(ValueError, match="at least one value"):
        ag.where_in("ID", [])


# --- _query_all (paging) ---
def test_query_all_pages_through_every_feature(post):
    """Test that _query_all pages through every feature in the layer."""
    fake = post(FakeLayer(table_rows(5), max_record_count=2))
    features = ag._query_all(LAYER, "1=1", None, {})
    assert features == table_rows(5)
    assert [c["data"]["resultOffset"] for c in query_calls(fake)] == [0, 2, 4]


def test_query_all_sorts_by_object_id_and_requests_fields(post):
    """Test that _query_all sorts by object ID and requests fields in the specified order."""
    fake = post(FakeLayer(table_rows(1), object_id_field="OBJECTID"))
    ag._query_all(LAYER, "CODE = 'E00'", ["CODE", "NAME"], {"returnGeometry": "false"})
    data = query_calls(fake)[0]["data"]
    assert data["orderByFields"] == "OBJECTID"
    assert data["outFields"] == "CODE,NAME"
    assert data["where"] == "CODE = 'E00'"
    assert data["returnGeometry"] == "false"


def test_query_all_requests_all_fields_by_default(post):
    """Test that _query_all requests all fields by default."""
    fake = post(FakeLayer(table_rows(1)))
    ag._query_all(LAYER, "1=1", None, {})
    assert query_calls(fake)[0]["data"]["outFields"] == "*"


def test_query_all_page_size_overrides_server_maximum(post):
    """Test that _query_all overrides the server's maximum page size if specified."""
    fake = post(FakeLayer(table_rows(3), max_record_count=1000))
    ag._query_all(LAYER, "1=1", None, {}, page_size=1)
    assert [c["data"]["resultRecordCount"] for c in query_calls(fake)] == [1, 1, 1]


def test_query_all_defaults_page_size_when_server_gives_none(post):
    """Test that _query_all defaults to the server's maximum page size if not specified."""
    layer = FakeLayer(table_rows(1))
    layer.info = {"objectIdField": "FID"}
    fake = post(layer)
    ag._query_all(LAYER, "1=1", None, {})
    assert query_calls(fake)[0]["data"]["resultRecordCount"] == 1000


def test_query_all_with_no_matches_makes_no_page_requests(post):
    """Test that _query_all makes no page requests when there are no matches."""
    fake = post(FakeLayer([]))
    assert ag._query_all(LAYER, "1=0", None, {}) == []
    assert query_calls(fake) == []


def test_query_all_raises_when_server_stops_early(post):
    """Test that _query_all raises an error when the server stops early, as can happen if busy."""
    post(FakeLayer(table_rows(3), short_pages=True))
    with pytest.raises(RuntimeError, match="Expected 3 rows .* received 0"):
        ag._query_all(LAYER, "1=1", None, {})


# --- query_table ---
def test_query_table_returns_attributes_without_geometry(post):
    """Test that query_table returns attributes without geometry (so not a GeoDataFrame)."""
    fake = post(FakeLayer(table_rows(3)))
    table = ag.query_table(LAYER)
    assert isinstance(table, pd.DataFrame)
    assert table["CODE"].tolist() == ["E00", "E01", "E02"]
    data = query_calls(fake)[0]["data"]
    assert data["returnGeometry"] == "false"
    assert data["f"] == "json"


def test_query_table_orders_columns_as_requested(post):
    """Test that query_table orders columns as requested if specified."""
    post(FakeLayer(table_rows(2)))
    table = ag.query_table(LAYER, out_fields=["NAME", "CODE"])
    assert table.columns.tolist() == ["NAME", "CODE"]


# --- query_features ---
def test_query_features_returns_geodataframe_in_british_national_grid(post):
    """Test that query_features returns a GeoDataFrame in British National Grid coordinate system (27700) by default."""
    fake = post(FakeLayer(geojson_features(3)))
    gdf = ag.query_features(LAYER)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == ag.BRITISH_NATIONAL_GRID
    assert gdf["CODE"].tolist() == ["E00", "E01", "E02"]
    assert gdf.geometry.iloc[1].x == 380_001
    data = query_calls(fake)[0]["data"]
    assert data["f"] == "geojson"
    assert data["outSR"] == ag.BRITISH_NATIONAL_GRID


def test_query_features_uses_requested_coordinate_system(post):
    """Test that query_features uses the requested coordinate system if BNG is not requested / not left as default."""
    fake = post(FakeLayer(geojson_features(1)))
    gdf = ag.query_features(LAYER, out_sr=4326)
    assert gdf.crs.to_epsg() == 4326
    assert query_calls(fake)[0]["data"]["outSR"] == 4326


def test_query_features_keeps_requested_fields_then_geometry(post):
    """Test that query_features keeps the requested fields, then geometry at the end (standard geo data format)."""
    post(FakeLayer(geojson_features(2)))
    gdf = ag.query_features(LAYER, out_fields=["NAME", "CODE"])
    assert gdf.columns.tolist() == ["NAME", "CODE", "geometry"]


def test_query_features_with_no_matches_returns_empty_frame(post):
    """Test that query_features returns an empty GeoDataFrame when there are no matches, rather than raising."""
    post(FakeLayer([]))
    gdf = ag.query_features(LAYER, where="1=0", out_fields=["CODE"])
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty
    assert gdf.columns.tolist() == ["CODE", "geometry"]
    assert gdf.crs.to_epsg() == ag.BRITISH_NATIONAL_GRID


def test_query_features_passes_page_size(post):
    """Test that query_features passes the requested page size to the server, and so not the server default."""
    fake = post(FakeLayer(geojson_features(3), max_record_count=1000))
    ag.query_features(LAYER, page_size=2)
    assert [c["data"]["resultRecordCount"] for c in query_calls(fake)] == [2, 2]


# --- item_info and download_item ---
ITEM_ID = "abc123"
PAYLOAD = [b"PK\x03\x04", b"rest of the file"]
PAYLOAD_SIZE = sum(len(c) for c in PAYLOAD)


@pytest.fixture
def item(post, monkeypatch):
    """
    Serve a file item: `info` overrides its catalogue entry, and the returned dict records the download requests.
    """
    state = {"info": {"title": "Lookup", "type": "CSV Collection", "size": PAYLOAD_SIZE}, "gets": []}
    post(lambda url, data: state["info"])

    def fake_get(url, stream, timeout):
        state["gets"].append({"url": url, "stream": stream, "timeout": timeout})
        return FakeResponse(chunks=PAYLOAD)

    monkeypatch.setattr(ag.requests, "get", fake_get)
    return state


def test_item_info_reads_catalogue_entry(post):
    """Test that item_info reads the catalogue entry (metadata) not the actual file."""
    fake = post(lambda url, data: {"title": "Lookup"})
    assert ag.item_info(ITEM_ID) == {"title": "Lookup"}
    assert fake.calls[0]["url"] == f"{ag.ARCGIS_ONLINE_ITEMS}/{ITEM_ID}"


def test_download_item_saves_file_and_returns_path(item, tmp_path):
    """Test that download_item saves the file and returns the path, and that the partial file is removed."""
    dest = tmp_path / "nested" / "lookup.zip"
    assert ag.download_item(ITEM_ID, str(dest), expected_title="Lookup") == dest
    assert dest.read_bytes() == b"".join(PAYLOAD)
    assert not dest.with_name("lookup.zip.partial").exists()
    assert item["gets"] == [{"url": f"{ag.ARCGIS_ONLINE_ITEMS}/{ITEM_ID}/data", "stream": True,
                             "timeout": ag.TIMEOUT}]


def test_download_item_rejects_unexpected_title(item, tmp_path):
    """
    Test that download_item rejects a file with an unexpected title, as a measure to avoid downloading the wrong file.
    """
    with pytest.raises(ValueError, match="titled 'Lookup', expected 'Something else'"):
        ag.download_item(ITEM_ID, tmp_path / "x.zip", expected_title="Something else")
    assert item["gets"] == []


def test_download_item_rejects_services(item, tmp_path):
    """Test that download_item rejects a service, as they should be queried using the query functions."""
    item["info"] = {"title": "Lookup", "type": "Feature Service", "url": "https://example.test/FeatureServer"}
    with pytest.raises(ValueError, match="not a file"):
        ag.download_item(ITEM_ID, tmp_path / "x.zip")
    assert item["gets"] == []


def test_download_item_keeps_partial_file_on_size_mismatch(item, tmp_path):
    """Test that download_item keeps the partial file if the download size does not match the catalogue entry."""
    item["info"]["size"] = PAYLOAD_SIZE + 1
    dest = tmp_path / "lookup.zip"
    with pytest.raises(RuntimeError, match=f"Downloaded {PAYLOAD_SIZE} bytes"):
        ag.download_item(ITEM_ID, dest)
    assert not dest.exists()
    assert (tmp_path / "lookup.zip.partial").read_bytes() == b"".join(PAYLOAD)


def test_download_item_skips_size_check_when_size_unknown(item, tmp_path):
    """Test that download_item skips the size check if the size is unknown (not specified in metadata)."""
    del item["info"]["size"]
    dest = tmp_path / "lookup.zip"
    ag.download_item(ITEM_ID, dest)
    assert dest.exists()
