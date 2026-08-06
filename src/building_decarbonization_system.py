"""
NYC Building Electrification Targeting System
=============================================
Screens NYC LL84 benchmarking data to rank buildings by heat pump retrofit
potential, using HVAC replacement-cycle age and neighborhood heat vulnerability.

DATA SOURCES
  Buildings : NYC Open Data 5zyy-y8am (LL84 energy benchmarking disclosure)
  Thermal   : NYC Open Data 4mhf-duep (NYC DOHMH Heat Vulnerability Index,
              ZIP-level ordinal tiers 1-5). NOTE: this is a NYC agency index,
              not a NASA satellite product. Label it accordingly.
  Geometry  : NYC ZCTA polygons (GeoJSON)

METHOD NOTES / KNOWN LIMITS -- state these if asked:
  - "Electrified" means all-electric by fuel mix (electricity reported, no gas
    or oil). LL84 does not report equipment type, so this is a proxy for heat
    pump presence, not a direct observation.
  - HVAC age is inferred from year_built. Buildings that already replaced
    equipment are indistinguishable from those that have not.
  - Heat vulnerability is joined at ZIP level, so all buildings in a ZIP share
    a tier. This is coarse and correlates with income and housing age.
  - Targeting is rule-based filtering and ranking. There is no trained model.
"""

import json
import warnings

import folium
import geopandas as gpd
import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from IPython.display import clear_output, display

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
BUILDING_API_URL = "https://data.cityofnewyork.us/resource/5zyy-y8am.json?$limit=50000"
THERMAL_API_URL = "https://data.cityofnewyork.us/resource/4mhf-duep.json?$limit=500"
GEOJSON_POLY_URL = (
    "https://raw.githubusercontent.com/fedhere/PUI2015_EC/master/"
    "mam1612_EC/nyc-zip-code-tabulation-areas-polygons.geojson"
)

CURRENT_YEAR = 2026
HVAC_WINDOW = (2006, 2011)      # build years now inside the 15-20yr replacement cycle
METRIC_CRS = "EPSG:32618"        # UTM 18N, metres -- for real distance in NYC
MIN_ZIP_SAMPLE = 10              # ZIPs below this are excluded from rate stats

DATA_IS_SYNTHETIC = False        # set True only by the fallback path


# =============================================================================
# PHASE 1: INGESTION
# =============================================================================
def load_live_data():
    buildings = pd.read_json(BUILDING_API_URL)
    thermal = pd.read_json(THERMAL_API_URL)
    polygons = requests.get(GEOJSON_POLY_URL, timeout=60).json()
    return buildings, thermal, polygons


def load_demo_data():
    """Clearly-labelled demo data. Only for offline UI testing."""
    rng = np.random.default_rng(42)
    zips = ["10026", "10035", "10451", "11101", "11211", "11236"]
    df = pd.DataFrame({
        "primary_property_type": rng.choice(
            ["Office", "Hotel", "Retail Store", "Multifamily Housing"], 500),
        "property_name": [f"DEMO-DATA Asset #{i}" for i in range(500)],
        "address_1": [f"{rng.integers(10, 2500)} Demo Street" for _ in range(500)],
        "year_built": rng.integers(2006, 2012, 500),
        "postal_code": rng.choice(zips, 500),
        "latitude": rng.uniform(40.63, 40.85, 500),
        "longitude": rng.uniform(-73.96, -73.88, 500),
        "electricity_use": rng.uniform(5000, 50000, 500),
        "natural_gas_use": rng.choice([np.nan, 20000], 500, p=[0.3, 0.7]),
        "fuel_oil_1": np.nan,
        "fuel_oil_2": np.nan,
    })
    thermal = pd.DataFrame({"zcta": zips, "hvi": [5, 4, 5, 3, 2, 5]})
    return df, thermal, {"type": "FeatureCollection", "features": []}


print("Loading data...")
try:
    raw_building_df, raw_thermal_df, nyc_zip_geojson = load_live_data()
    print(f"  Live API: {len(raw_building_df):,} building records retrieved.")
except Exception as exc:
    DATA_IS_SYNTHETIC = True
    raw_building_df, raw_thermal_df, nyc_zip_geojson = load_demo_data()
    print("\n" + "!" * 70)
    print("!! API FAILED -- RUNNING ON DEMO DATA. RESULTS ARE NOT REAL.")
    print(f"!! Reason: {exc}")
    print("!" * 70 + "\n")


# =============================================================================
# PHASE 2: COLUMN RESOLUTION
# =============================================================================
def find_best_column(candidates, all_columns):
    for name in candidates:
        if name in all_columns:
            return name
    for name in candidates:
        for col in all_columns:
            if name in col:
                return col
    return None


cols = list(raw_building_df.columns)
type_col = find_best_column(["primary_property_type", "property_type"], cols)
name_col = find_best_column(["property_name", "building_name"], cols)
addr_col = find_best_column(["address_1", "street_address"], cols)
year_col = find_best_column(["year_built", "construction_year"], cols)
zip_col = find_best_column(["postal_code", "zipcode", "zip_code"], cols)
lat_col = find_best_column(["latitude"], cols)
lon_col = find_best_column(["longitude"], cols)
elec_col = find_best_column(["electricity_use", "electricity"], cols)
gas_col = find_best_column(["natural_gas_use", "natural_gas"], cols)
oil1_col = find_best_column(["fuel_oil_1"], cols)
oil2_col = find_best_column(["fuel_oil_2"], cols)

print("\nResolved columns:")
for label, c in [("type", type_col), ("name", name_col), ("address", addr_col),
                 ("year", year_col), ("zip", zip_col), ("lat", lat_col),
                 ("lon", lon_col), ("electricity", elec_col), ("gas", gas_col),
                 ("oil_1", oil1_col), ("oil_2", oil2_col)]:
    print(f"  {label:<12} -> {c}")

missing = [n for n, c in [("year", year_col), ("zip", zip_col), ("lat", lat_col),
                          ("lon", lon_col), ("electricity", elec_col)] if c is None]
if missing:
    raise RuntimeError(f"Required columns unresolved: {missing}. Check the API schema.")

fossil_cols = [c for c in (gas_col, oil1_col, oil2_col) if c]
if not fossil_cols:
    raise RuntimeError("No fossil fuel columns resolved -- cannot classify fuel mix.")


# =============================================================================
# PHASE 3: CLEANING
# =============================================================================
# LL84 encodes unreported values as "Not Available" / "Not Applicable".
# to_numeric coerces those to NaN. That distinction matters below.
for c in [year_col, lat_col, lon_col, elec_col] + fossil_cols:
    raw_building_df[c] = pd.to_numeric(raw_building_df[c], errors="coerce")

n_raw = len(raw_building_df)
raw_building_df = raw_building_df.dropna(subset=[lat_col, lon_col])
n_geo = len(raw_building_df)

raw_building_df[zip_col] = (
    raw_building_df[zip_col].astype(str).str.strip().str.split(".").str[0]
)
raw_building_df = raw_building_df.drop_duplicates(subset=[addr_col]).copy()
n_clean = len(raw_building_df)

print(f"\nCleaning: {n_raw:,} raw -> {n_geo:,} geocoded -> {n_clean:,} deduped by address")


# =============================================================================
# PHASE 4: FUEL-MIX CLASSIFICATION
# =============================================================================
# BUGFIX: the original tested `col == 0`, which is False for NaN. All-electric
# buildings report blanks rather than zeros for gas and oil, so every genuine
# all-electric building was excluded and the detection count was zero.
#
# Two definitions, because the blank is genuinely ambiguous:
#   loose  -- blank OR zero counts as "fuel not used"   (upper bound)
#   strict -- only an explicit zero counts               (lower bound)

def fuel_absent_loose(s):
    return s.isna() | (s <= 0)


def fuel_absent_strict(s):
    return s.fillna(-1) == 0


has_elec = raw_building_df[elec_col] > 0
no_fossil_loose = np.logical_and.reduce([fuel_absent_loose(raw_building_df[c]) for c in fossil_cols])
no_fossil_strict = np.logical_and.reduce([fuel_absent_strict(raw_building_df[c]) for c in fossil_cols])

raw_building_df["all_electric"] = (has_elec & no_fossil_loose).astype(int)
raw_building_df["all_electric_strict"] = (has_elec & no_fossil_strict).astype(int)

n_loose = int(raw_building_df["all_electric"].sum())
n_strict = int(raw_building_df["all_electric_strict"].sum())

print("\nFuel-mix classification (proxy for electrification):")
print(f"  Loose  (blank or zero fossil): {n_loose:,}  ({n_loose / n_clean * 100:.1f}%)")
print(f"  Strict (explicit zero only)  : {n_strict:,}  ({n_strict / n_clean * 100:.1f}%)")
if n_loose == 0:
    print("  WARNING: zero detections. Inspect the fuel columns before trusting output.")

# NOTE: the original script contained a block that randomly flagged 25% of rows
# as electrified whenever real detections fell below 10. It has been removed.
# It silently replaced a broken classifier with noise.


# =============================================================================
# PHASE 5: THERMAL JOIN
# =============================================================================
t_cols = list(raw_thermal_df.columns)
t_zip_col = find_best_column(["zcta", "zip"], t_cols)
hvi_col = find_best_column(["hvi", "heat_index"], t_cols)
raw_thermal_df[t_zip_col] = (
    raw_thermal_df[t_zip_col].astype(str).str.strip().str.split(".").str[0]
)
raw_thermal_df[hvi_col] = pd.to_numeric(raw_thermal_df[hvi_col], errors="coerce")

real_gdf = raw_building_df.merge(
    raw_thermal_df[[t_zip_col, hvi_col]].drop_duplicates(subset=[t_zip_col]),
    left_on=zip_col, right_on=t_zip_col, how="left",
).rename(columns={hvi_col: "hvi"})

# Do NOT impute a middle tier. Flag unmatched rows and exclude them from
# HVI-driven ranking, so imputed values cannot masquerade as measurements.
real_gdf["hvi_matched"] = real_gdf["hvi"].notna()
n_matched = int(real_gdf["hvi_matched"].sum())
print(f"\nThermal join: {n_matched:,} of {len(real_gdf):,} matched "
      f"({n_matched / len(real_gdf) * 100:.1f}%)")
print(f"  Tier distribution: {real_gdf['hvi'].value_counts().sort_index().to_dict()}")

real_gdf = gpd.GeoDataFrame(
    real_gdf,
    geometry=gpd.points_from_xy(real_gdf[lon_col], real_gdf[lat_col]),
    crs="EPSG:4326",
)
# Projected copy for distance work -- degrees are anisotropic at NYC latitude
# (1 deg lon is ~0.76x 1 deg lat), so raw lat/lon Euclidean picks wrong neighbours.
_projected = real_gdf.to_crs(METRIC_CRS)
real_gdf["x_m"] = _projected.geometry.x
real_gdf["y_m"] = _projected.geometry.y


# =============================================================================
# PHASE 6: CANDIDATE POOL AND RANKING
# =============================================================================
in_window = real_gdf[year_col].between(*HVAC_WINDOW)
candidate_mask = in_window & (real_gdf["all_electric"] == 0)
candidate_pool = real_gdf[candidate_mask].copy()
candidate_pool["hvac_age"] = CURRENT_YEAR - candidate_pool[year_col]

print(f"\nCandidates: {int(in_window.sum()):,} in {HVAC_WINDOW[0]}-{HVAC_WINDOW[1]} "
      f"window -> {len(candidate_pool):,} not already all-electric")

ranked = candidate_pool[candidate_pool["hvi_matched"]]
macro_summary = (
    ranked.groupby([zip_col, "hvi"]).size().reset_index(name="candidates")
)
macro_summary["priority_score"] = macro_summary["hvi"] * macro_summary["candidates"]
macro_summary = macro_summary.sort_values("priority_score", ascending=False).rename(
    columns={zip_col: "ZIP Code", "hvi": "Heat Vulnerability Tier",
             "candidates": "Retrofit Candidates", "priority_score": "Priority Score"}
)


def zip_adoption_table():
    """ZIP-level electrification rates. Excludes thin ZIPs and unmatched HVI."""
    rows = []
    for z, g in real_gdf[real_gdf["hvi_matched"]].groupby(zip_col):
        if str(z).strip() in ("", "nan") or len(g) < MIN_ZIP_SAMPLE:
            continue
        rows.append({
            "zip": z,
            "hvi": int(g["hvi"].iloc[0]),
            "adoption_pct": g["all_electric"].mean() * 100,
            "n_buildings": len(g),
        })
    return pd.DataFrame(rows)


# =============================================================================
# PHASE 7: PROXIMITY
# =============================================================================
def nearest_electrified(target_row, exclude_address=None):
    """Nearest all-electric building, in metres, using projected coordinates."""
    pool = real_gdf[real_gdf["all_electric"] == 1]
    if exclude_address is not None:
        pool = pool[pool[addr_col].astype(str).str.strip().str.lower()
                    != str(exclude_address).strip().lower()]
    if pool.empty:
        return None, None
    d = np.hypot(pool["x_m"] - target_row["x_m"], pool["y_m"] - target_row["y_m"])
    idx = d.idxmin()
    return real_gdf.loc[[idx]], float(d.loc[idx])


# =============================================================================
# PHASE 8: SEARCH
# =============================================================================
TYPE_GROUPS = {
    "commercial": ["Office", "Medical Office", "Hotel", "Retail Store",
                   "Wholesale Club/Supercenter", "Supermarket/Grocery Store"],
    "office": ["Office", "Medical Office"],
    "hotel": ["Hotel"],
    "retail": ["Retail Store", "Wholesale Club/Supercenter", "Supermarket/Grocery Store"],
    "grocery": ["Supermarket/Grocery Store"],
}
RESIDENTIAL_TERMS = "multifamily|housing|apartment|residential"


def search_candidates(query, top_n):
    q = str(query).lower().strip()
    pool = candidate_pool

    if q in TYPE_GROUPS:
        pool = pool[pool[type_col].isin(TYPE_GROUPS[q])]
    elif q in ("multifamily", "residential", "apartment", "housing"):
        pool = pool[pool[type_col].str.lower().str.contains(RESIDENTIAL_TERMS, na=False)]
    else:
        pool = pool[pool[type_col].str.lower().str.contains(q, na=False)]

    if pool.empty:
        return pool

    # Rank matched-HVI buildings first, then by heat tier, then by equipment age
    return pool.sort_values(
        ["hvi_matched", "hvi", "hvac_age"], ascending=[False, False, False]
    ).head(int(top_n)).copy()


# =============================================================================
# PHASE 9: OUTREACH DRAFT
# =============================================================================
TIER_LABEL = {1: "low", 2: "mild", 3: "moderate", 4: "high", 5: "severe"}
LANDMARKS = {
    "10026": "Morningside Park", "10035": "Marcus Garvey Park",
    "10451": "Mill Pond Park", "11236": "Canarsie Beach Park",
    "11101": "Gantry Plaza State Park", "11211": "McCarren Park",
}


def draft_outreach(row):
    """Template-filled draft. Not a generative model -- string substitution."""
    name = row[name_col]
    addr = row[addr_col]
    age = int(row["hvac_age"])
    b_zip = str(row[zip_col]).strip()

    sibling, dist_m = nearest_electrified(row, exclude_address=addr)
    if sibling is not None and dist_m is not None and dist_m < 1000:
        proof = (f"A building {dist_m:,.0f}m away at {sibling.iloc[0][addr_col]} "
                 f"already runs all-electric, so local performance data is available.")
    else:
        proof = "Comparable all-electric conversions are underway across the borough."

    if row["hvi_matched"]:
        tier = int(row["hvi"])
        heat_line = (f"This ZIP sits at tier {tier} of 5 on the NYC Heat Vulnerability "
                     f"Index ({TIER_LABEL.get(tier, 'elevated')} vulnerability), so peak "
                     f"summer cooling load is a growing cost exposure.")
    else:
        heat_line = "Peak summer cooling load is a growing cost exposure."

    landmark = LANDMARKS.get(b_zip)
    opener = (f"I was near {landmark} recently and noticed your building at {addr}."
              if landmark else f"I came across your building at {addr} in city records.")

    return f"""Subject: HVAC replacement planning at {name}

Dear Property Manager,

{opener}

City benchmarking records show the building was constructed {age} years ago,
which typically places central heating and cooling equipment at or past the
15-20 year replacement window. (Note: this is inferred from construction year,
not equipment records.)

{heat_line}

{proof}

Would you be open to a short call to review the existing plant layout and
current incentive programs?

Best regards,
"""


# =============================================================================
# PHASE 10: UI
# =============================================================================
search_box = widgets.Text(value="commercial", description="Search:",
                          layout=widgets.Layout(width="38%"))
limit_slider = widgets.IntSlider(value=10, min=1, max=25, description="Top N:",
                                 layout=widgets.Layout(width="32%"))
search_button = widgets.Button(description="Find Candidates", button_style="success",
                               layout=widgets.Layout(width="22%"))

status_view = widgets.HTML()
table_view = widgets.HTML()
target_box = widgets.Text(description="Target:", layout=widgets.Layout(width="58%"))
draft_button = widgets.Button(description="Draft Outreach", button_style="info",
                              layout=widgets.Layout(width="24%"))
draft_pane = widgets.HTML(layout=widgets.Layout(padding="12px", border="1px dashed #999"))
micro_map_pane = widgets.Output(layout=widgets.Layout(height="340px", border="1px solid #ccc"))
map_output_pane = widgets.Output(layout=widgets.Layout(height="560px", overflow_y="auto"))
macro_table_view = widgets.HTML(layout=widgets.Layout(width="98%"))

banner = widgets.HTML(
    "<div style='background:#c0392b;color:#fff;padding:8px;font-weight:bold;'>"
    "DEMO DATA -- results are not real</div>" if DATA_IS_SYNTHETIC else ""
)

current_slice = pd.DataFrame()

DISPLAY_COLS = {
    name_col: "Property", addr_col: "Address", type_col: "Type",
    year_col: "Year Built", "hvac_age": "HVAC Age", "hvi": "Heat Tier",
}


def render_micro_map(row, sibling, dist_m):
    fig = folium.Figure(height=320)
    m = folium.Map(location=[row.geometry.y, row.geometry.x],
                   zoom_start=16, tiles="OpenStreetMap").add_to(fig)
    folium.Marker([row.geometry.y, row.geometry.x],
                  popup=f"TARGET: {row[name_col]}",
                  icon=folium.Icon(color="red")).add_to(m)
    if sibling is not None:
        s = sibling.iloc[0]
        folium.Marker([s.geometry.y, s.geometry.x],
                      popup=f"ALL-ELECTRIC: {s[addr_col]} ({dist_m:,.0f}m)",
                      icon=folium.Icon(color="green", icon="leaf")).add_to(m)
        folium.PolyLine([[row.geometry.y, row.geometry.x],
                         [s.geometry.y, s.geometry.x]],
                        color="#34495e", weight=2.5, dash_array="5,10").add_to(m)
    with micro_map_pane:
        clear_output(wait=True)
        display(fig)


def render_city_map(focus_zip=None):
    if focus_zip and str(focus_zip).strip() in set(real_gdf[zip_col]):
        sel = real_gdf[real_gdf[zip_col] == str(focus_zip).strip()]
        center, zoom = [sel.geometry.y.mean(), sel.geometry.x.mean()], 14
    else:
        center, zoom = [40.72, -73.95], 11

    fig = folium.Figure(height=540)
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron").add_to(fig)

    hvi_by_zip = real_gdf.dropna(subset=["hvi"]).groupby(zip_col)["hvi"].first().to_dict()
    palette = {1: "#fee5d9", 2: "#fcae91", 3: "#fb6a4a", 4: "#de2d26", 5: "#a50f15"}

    def style(feature):
        props = feature["properties"]
        z = str(props.get("postalCode") or props.get("ZIPCODE") or "").strip()
        tier = hvi_by_zip.get(z)
        return {"fillColor": palette.get(int(tier), "#cccccc") if tier else "#e8e8e8",
                "color": "#2c3e50", "weight": 1.0, "fillOpacity": 0.45}

    if nyc_zip_geojson.get("features"):
        folium.GeoJson(json.loads(json.dumps(nyc_zip_geojson)), style_function=style).add_to(m)

    for z, g in real_gdf.groupby(zip_col):
        if str(z).strip() in ("", "nan") or len(g) < MIN_ZIP_SAMPLE:
            continue
        pct = g["all_electric"].mean() * 100
        n_cand = int((g[year_col].between(*HVAC_WINDOW) & (g["all_electric"] == 0)).sum())
        tier = g["hvi"].iloc[0]
        tier_txt = f"Tier {int(tier)}/5" if pd.notna(tier) else "unmatched"
        icon = (f"<div style='width:30px;height:30px;border-radius:50%;"
                f"background:conic-gradient(#2ecc71 0% {pct}%,#e74c3c {pct}% 100%);"
                f"border:2px solid #fff;display:flex;align-items:center;"
                f"justify-content:center;font:bold 10px Arial;color:#fff;'>{len(g)}</div>")
        popup = (f"<b>ZIP {z}</b><br>Heat vulnerability: {tier_txt}<br>"
                 f"Buildings: {len(g)}<br>Retrofit candidates: {n_cand}<br>"
                 f"All-electric: {pct:.1f}%")
        folium.Marker([g.geometry.y.mean(), g.geometry.x.mean()],
                      icon=folium.DivIcon(html=icon, icon_size=(30, 30)),
                      popup=folium.Popup(popup, max_width=260)).add_to(m)

    with map_output_pane:
        clear_output(wait=True)
        display(fig)


def on_search(_):
    global current_slice
    status_view.value = "<i>Searching...</i>"
    result = search_candidates(search_box.value, limit_slider.value)
    if result.empty:
        status_view.value = "<b style='color:#c0392b;'>No matches.</b>"
        table_view.value = ""
        return
    current_slice = result
    status_view.value = f"<h3>Top {len(result)} candidates</h3>"
    shown = result[[c for c in DISPLAY_COLS if c in result.columns]].rename(columns=DISPLAY_COLS)
    table_view.value = (f"<div style='max-height:220px;overflow-y:auto;'>"
                        f"{shown.to_html(index=False, border=0)}</div>")
    target_box.value = str(result.iloc[0][name_col])
    render_city_map()
    on_draft(None)


def on_draft(_):
    if current_slice.empty:
        draft_pane.value = "<i>Run a search first.</i>"
        return
    q = str(target_box.value).lower().strip()
    match = current_slice[
        current_slice[name_col].astype(str).str.lower().str.contains(q, na=False)
        | current_slice[addr_col].astype(str).str.lower().str.contains(q, na=False)
    ]
    row = (match if not match.empty else current_slice).iloc[0]
    sibling, dist_m = nearest_electrified(row, exclude_address=row[addr_col])
    render_micro_map(row, sibling, dist_m)
    draft_pane.value = f"<pre style='white-space:pre-wrap;font:13px monospace;'>{draft_outreach(row)}</pre>"


search_button.on_click(on_search)
draft_button.on_click(on_draft)

page_1 = widgets.VBox([
    widgets.HTML("<h2>Candidate Search</h2>"),
    widgets.HBox([search_box, limit_slider, search_button]),
    status_view, table_view,
    widgets.HTML("<hr><h3>Outreach Draft</h3>"),
    widgets.HBox([target_box, draft_button]),
    draft_pane,
    widgets.HTML("<h4>Nearest All-Electric Building</h4>"),
    micro_map_pane,
])

map_focus_box = widgets.Text(value="10035", description="ZIP:",
                             layout=widgets.Layout(width="52%"))
map_focus_btn = widgets.Button(description="Focus", button_style="warning",
                               layout=widgets.Layout(width="28%"))
map_focus_btn.on_click(lambda b: render_city_map(map_focus_box.value))

page_2 = widgets.HBox([
    widgets.VBox([widgets.HTML("<h2>ZIP Leaderboard</h2>"), macro_table_view],
                 layout=widgets.Layout(width="38%")),
    widgets.VBox([widgets.HTML("<h2>Heat Vulnerability Map</h2>"),
                  widgets.HBox([map_focus_box, map_focus_btn]), map_output_pane],
                 layout=widgets.Layout(width="62%")),
])

tabs = widgets.Tab(children=[page_1, page_2])
tabs.set_title(0, "Candidate Search")
tabs.set_title(1, "Neighborhood Dashboard")

display(widgets.VBox([banner, tabs]))
on_search(None)
macro_table_view.value = (f"<div style='max-height:480px;overflow-y:auto;'>"
                          f"{macro_summary.to_html(index=False, border=0)}</div>")
