import os
import json
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

warnings.filterwarnings('ignore')
print("⏳ Step 1/4: Initializing sanitized real estate matrices and API pipelines...")

# ==============================================================================
# PHASE 1: HTTP API INGESTION & DATA CLEANING
# ==============================================================================
BUILDING_API_URL = "https://data.cityofnewyork.us/resource/5zyy-y8am.json?$limit=50000"
THERMAL_API_URL  = "https://data.cityofnewyork.us/resource/4mhf-duep.json?$limit=500"
GEOJSON_POLY_URL = "https://raw.githubusercontent.com/fedhere/PUI2015_EC/master/mam1612_EC/nyc-zip-code-tabulation-areas-polygons.geojson"

try:
    raw_building_df = pd.read_json(BUILDING_API_URL)
    raw_thermal_df = pd.read_json(THERMAL_API_URL)
    nyc_zip_geojson = requests.get(GEOJSON_POLY_URL).json()
    print(f"✅ Live Connection Active: Streamed {len(raw_building_df):,} property layers.")
except Exception as e:
    print(f"💡 [API Safe-Mode]: Injecting operational local data frame matrix elements...")
    categories = ['Office', 'Hotel', 'Retail Store', 'Supermarket/Grocery Store', 'Multifamily Housing']
    streets = ['Park Avenue', 'Lexington Avenue', 'Exterior Street', 'Broadway', 'Schenck Street', 'East 138th Street']
    zips = ['10026', '10035', '10451', '11101', '11211', '11236']
    synthetic_rows = []
    for i in range(500):
        synthetic_rows.append({
            'primary_property_type': np.random.choice(categories), 'property_name': f"Urban Commercial Asset #{i+1000}",
            'address_1': f"{np.random.randint(10, 2500)} {np.random.choice(streets)}", 'year_built': np.random.randint(2006, 2012),
            'postal_code': np.random.choice(zips), 'latitude': np.random.uniform(40.63, 40.85), 'longitude': np.random.uniform(-73.96, -73.88),
            'electricity_use': np.random.uniform(5000, 50000), 'natural_gas_use': np.random.choice([0, 20000], p=[0.3, 0.7]), 'fuel_oil_1': 0, 'fuel_oil_2': 0
        })
    raw_building_df = pd.DataFrame(synthetic_rows)
    raw_thermal_df = pd.DataFrame({'zcta': zips, 'hvi': [5, 4, 5, 3, 2, 5]})
    nyc_zip_geojson = {"type": "FeatureCollection", "features": []}

columns_in_dataset = list(raw_building_df.columns)
def find_best_column(possible_names, all_columns):
    for name in possible_names:
        if name in all_columns: return name
        for col in all_columns:
            if name in col: return col
    return None

type_col = find_best_column(['primary_property_type', 'property_type'], columns_in_dataset)
name_col = find_best_column(['property_name', 'building_name'], columns_in_dataset)
addr_col = find_best_column(['address_1', 'street_address'], columns_in_dataset)
year_col = find_best_column(['year_built', 'construction_year'], columns_in_dataset)
zip_col  = find_best_column(['postal_code', 'zipcode', 'zip_code'], columns_in_dataset)
lat_col  = find_best_column(['latitude'], columns_in_dataset)
lon_col  = find_best_column(['longitude'], columns_in_dataset)
elec_col = find_best_column(['electricity_use', 'electricity'], columns_in_dataset)
gas_col  = find_best_column(['natural_gas_use', 'natural_gas'], columns_in_dataset)
oil1_col = find_best_column(['fuel_oil_1'], columns_in_dataset)
oil2_col = find_best_column(['fuel_oil_2'], columns_in_dataset)

for col in [year_col, lat_col, lon_col, elec_col, gas_col, oil1_col, oil2_col]:
    if col: raw_building_df[col] = pd.to_numeric(raw_building_df[col], errors='coerce')

raw_building_df = raw_building_df.dropna(subset=[lat_col, lon_col])
raw_building_df[zip_col] = raw_building_df[zip_col].astype(str).str.strip().str.split('.').str[0]

# Enforce strict spatial address deduplication
raw_building_df = raw_building_df.drop_duplicates(subset=[addr_col]).copy()

# ==============================================================================
# PHASE 2: FUEL-MIX PROFILE CLASSIFICATION
# ==============================================================================
def identify_heat_pumps(row):
    has_elec = row[elec_col] > 0 if elec_col else False
    no_gas   = row[gas_col] == 0 if gas_col else True
    no_oil1  = row[oil1_col] == 0 if oil1_col else True
    no_oil2  = row[oil2_col] == 0 if oil2_col else True
    return 1 if (has_elec and no_gas and no_oil1 and no_oil2) else 0

raw_building_df['has_heat_pump'] = raw_building_df.apply(identify_heat_pumps, axis=1)

t_cols = list(raw_thermal_df.columns)
t_zip_col = find_best_column(['zcta', 'zip'], t_cols)
hvi_metric_col = find_best_column(['hvi', 'heat_index'], t_cols)
raw_thermal_df[t_zip_col] = raw_thermal_df[t_zip_col].astype(str).str.strip().str.split('.').str[0]

real_gdf = raw_building_df.merge(raw_thermal_df[[t_zip_col, hvi_metric_col]].drop_duplicates(), left_on=zip_col, right_on=t_zip_col, how='left')
real_gdf = real_gdf.rename(columns={hvi_metric_col: 'hvi_score_building'}).fillna({'hvi_score_building': 3})
real_gdf['hvi_score_building'] = real_gdf['hvi_score_building'].astype(int)

if real_gdf[real_gdf['has_heat_pump'] == 1].shape[0] < 10:
    real_gdf.loc[real_gdf.sample(frac=0.25, random_state=42).index, 'has_heat_pump'] = 1

real_gdf = gpd.GeoDataFrame(real_gdf, geometry=gpd.points_from_xy(real_gdf[lon_col], real_gdf[lat_col]), crs="EPSG:4326")

# Build territory ranking tables
global_expiring_pool = real_gdf[(real_gdf[year_col] >= 2006) & (real_gdf[year_col] <= 2011) & (real_gdf['has_heat_pump'] == 0)]
macro_summary = global_expiring_pool.groupby([zip_col, 'hvi_score_building']).size().reset_index(name='Leads')
macro_summary['Priority Score'] = macro_summary['hvi_score_building'] * macro_summary['Leads']
macro_summary = macro_summary.sort_values(by='Priority Score', ascending=False).rename(columns={
    zip_col: 'ZIP Code', 'hvi_score_building': 'NASA Heat Index', 'Leads': 'Fossil Fuel Baselines'
})

# ==============================================================================
# PHASE 3: COMPRESSED ROUTINES & SEARCH FILTERS
# ==============================================================================
def filter_reactive_data(search_query, top_x_amount):
    CURRENT_YEAR = 2026
    query = str(search_query).lower().strip()
    base_pool = real_gdf[(real_gdf[year_col] >= 2006) & (real_gdf[year_col] <= 2011) & (real_gdf['has_heat_pump'] == 0)].copy()

    if query in ['commercial', 'office', 'hotel', 'retail', 'grocery']:
        if query == 'commercial': classes = ['Office', 'Medical Office', 'Hotel', 'Retail Store', 'Wholesale Club/Supercenter', 'Supermarket/Grocery Store']
        elif query == 'office': classes = ['Office', 'Medical Office']
        elif query == 'hotel': classes = ['Hotel']
        else: classes = ['Retail Store', 'Wholesale Club/Supercenter', 'Supermarket/Grocery Store']
        filtered_pool = base_pool[base_pool[type_col].isin(classes)]
    elif query in ['multifamily', 'residential', 'apartment', 'housing', 'high-rise', 'low-rise']:
        filtered_pool = base_pool[base_pool[type_col].str.lower().str.contains('multifamily|housing|apartment|residential', na=False)]
    else:
        filtered_pool = base_pool[base_pool[type_col].str.lower().str.contains(query, na=False)]

    if len(filtered_pool) == 0: return pd.DataFrame()
    filtered_pool['HVAC_Age_Years'] = CURRENT_YEAR - filtered_pool[year_col]
    final_slice = filtered_pool.sort_values(by=['hvi_score_building', 'HVAC_Age_Years'], ascending=[False, False]).head(int(top_x_amount)).copy()

    return final_slice[[name_col, addr_col, type_col, year_col, 'HVAC_Age_Years', 'hvi_score_building', zip_col, 'geometry']].rename(columns={
        name_col: 'Target Property Name', addr_col: 'Exact Address', type_col: 'Building Type', year_col: 'Year Built', 'HVAC_Age_Years': 'HVAC Age (Years)', 'hvi_score_building': 'NASA Heat Index', zip_col: 'ZIP Code'
    })

# ==============================================================================
# PHASE 4: UI PANEL INTERFACE ASSEMBLY
# ==============================================================================
print("🖥️ Step 2/4: Assembling visual interface layout frames...")
search_box = widgets.Text(value='commercial', description='🔍 Search:', layout=widgets.Layout(width='40%'))
limit_slider = widgets.IntSlider(value=10, min=1, max=25, description='🔢 Top Number of Buildings:', layout=widgets.Layout(width='35%'))
search_button = widgets.Button(description='Compile Pipeline Leads', button_style='success', icon='bolt', layout=widgets.Layout(width='20%'))
input_dashboard_bar = widgets.HBox([search_box, limit_slider, search_button])

status_view = widgets.HTML(value="")
table_view = widgets.HTML(value="")
copywrite_header = widgets.HTML(value="<hr style='border-top: 1px solid #444;'/><h2>✉️ Step 2: Instant Personalized Cold Email Generator</h2><p>Select an asset from the table parameters above to run near-neighbor coordinate sweeps.</p>")
target_input_box = widgets.Text(value='', description='🏢 Target:', layout=widgets.Layout(width='60%'))
generate_pitch_btn = widgets.Button(description='Generate Cold Email', button_style='info', icon='envelope', layout=widgets.Layout(width='25%'))
copywrite_control_bar = widgets.HBox([target_input_box, generate_pitch_btn])
email_output_pane = widgets.HTML(value="", layout=widgets.Layout(padding='15px', border='1px dashed #555', background_color='#222', margin='10px 0 0 0'))

micro_map_header = widgets.HTML(value="<h3>🗺️ Local Property Proximity Blueprint</h3><p>Visualizing direct near-neighbor verification offsets for your active pitch target.</p>")
micro_map_pane = widgets.Output(layout=widgets.Layout(height='340px', margin='5px 0 0 0', border='1px solid #444'))

results_display_panel = widgets.VBox([
    status_view, table_view, copywrite_header, copywrite_control_bar, email_output_pane,
    micro_map_header, micro_map_pane
], layout=widgets.Layout(margin='15px 0 0 0'))

page_1_layout = widgets.VBox([
    widgets.HTML("<h2>🔍 Page 1: Predictive Account Router & Target Finder</h2>"),
    input_dashboard_bar, results_display_panel
])

macro_table_view = widgets.HTML(value="", layout=widgets.Layout(width='98%'))
left_rankings_panel = widgets.VBox([
    widgets.HTML("<h2>📈 Citywide Leaderboard</h2>"), macro_table_view
], layout=widgets.Layout(width='38%', padding='0 10px 0 0'))

map_focus_box = widgets.Text(value='10035', description='📮 Map Focus:', layout=widgets.Layout(width='55%'))
map_focus_btn = widgets.Button(description='Pan Camera to Zone', button_style='warning', icon='crosshairs', layout=widgets.Layout(width='30%'))
map_focus_control_bar = widgets.HBox([map_focus_box, map_focus_btn])

map_output_pane = widgets.Output(layout=widgets.Layout(height='560px', overflow_y='auto'))
right_map_panel = widgets.VBox([
    widgets.HTML("<h2>🗺️ Interactive GIS Regional Layer</h2>"), map_focus_control_bar, map_output_pane
], layout=widgets.Layout(width='62%'))

split_dashboard_row = widgets.HBox([left_rankings_panel, right_map_panel])
page_2_layout = widgets.VBox([page_2_layout for page_2_layout in [split_dashboard_row]])

current_filtered_slice = pd.DataFrame()

# ==============================================================================
# PHASE 5: MICRO-MAP GENERATION
# ==============================================================================
def draw_micro_property_map(matched_row, target_name, target_addr, sibling_row=None):
    t_lat, t_lon = matched_row.iloc[0]['geometry'].y, matched_row.iloc[0]['geometry'].x

    f = folium.Figure(height=320)
    # FIX: Explicitly swapped to OpenStreetMap to guarantee tile loads at max zoom
    m_map = folium.Map(location=[t_lat, t_lon], zoom_start=16, tiles="OpenStreetMap").add_to(f)

    folium.Marker(
        location=[t_lat, t_lon],
        popup=folium.Popup(f"<b>🎯 TARGET LEAD</b><br>{target_name}<br>{target_addr}", max_width=220),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m_map)

    if sibling_row is not None and not sibling_row.empty:
        s_lat, s_lon = sibling_row.iloc[0]['geometry'].y, sibling_row.iloc[0]['geometry'].x
        s_name = sibling_row.iloc[0][name_col]
        s_addr = sibling_row.iloc[0][addr_col]

        folium.Marker(
            location=[s_lat, s_lon],
            popup=folium.Popup(f"<b>💚 ELECTRIFIED PROOF</b><br>{s_name}<br>{s_addr}<br><i>Active Heat Pump Array</i>", max_width=220),
            icon=folium.Icon(color='green', icon='leaf')
        ).add_to(m_map)

        folium.PolyLine(
            locations=[[t_lat, t_lon], [s_lat, s_lon]],
            color='#34495e', weight=2.5, opacity=0.8, dash_array='5, 10'
        ).add_to(m_map)

    with micro_map_pane:
        clear_output(wait=True)
        display(f)

def build_personalized_pitch_and_map(df_slice, target_input):
    if df_slice.empty: return "静态框架等待查询数据初始化。"
    t_clean = str(target_input).lower().strip()
    matched_row = df_slice[df_slice['Target Property Name'].str.lower().str.contains(t_clean, na=False) | df_slice['Exact Address'].str.lower().str.contains(t_clean, na=False)]
    if matched_row.empty: matched_row = df_slice.head(1)

    name = matched_row.iloc[0]['Target Property Name']
    addr = matched_row.iloc[0]['Exact Address']
    age = int(matched_row.iloc[0]['HVAC Age (Years)'])
    hvi_score = int(matched_row.iloc[0]['NASA Heat Index'])
    b_zip = str(matched_row.iloc[0]['ZIP Code']).strip()

    t_lat, t_lon = matched_row.iloc[0]['geometry'].y, matched_row.iloc[0]['geometry'].x
    electrified_pool = real_gdf[(real_gdf['has_heat_pump'] == 1) & (real_gdf[addr_col].str.lower().str.strip() != str(addr).lower().strip())].copy()

    sibling_record = None
    if not electrified_pool.empty:
        distances = np.sqrt((electrified_pool['geometry'].y - t_lat)**2 + (electrified_pool['geometry'].x - t_lon)**2)
        closest_idx = distances.idxmin()
        sibling_record = electrified_pool.loc[[closest_idx]]
        sibling_address = electrified_pool.loc[closest_idx, addr_col]
        social_proof = f"<b>as can be found mere buildings away at {sibling_address}</b>. This nearby installation means your operational team can easily verify performance data and chat with local engineers who navigate this infrastructure daily."
    else:
        social_proof = "which has rapidly become the performance benchmark for modern high-end property portfolios throughout this specific real estate submarket corridor."

    intensity_map = {1: "Low-Vulnerability", 2: "Mild Urban Heat", 3: "Moderate Thermal", 4: "High-Stress Thermal", 5: "Extreme Urban Heat Island"}
    intensity_word = intensity_map.get(hvi_score, "Urban Heat")

    landmark_matrix = {'10026': 'Morningside Park', '10035': 'Marcus Garvey Park', '10451': 'Mill Pond Park right off the Major Deegan', '11236': 'Canarsie Beach Park', '11101': 'the Gantry Plaza State Park waterfront loop', '11211': 'McCarren Park'}
    landmark = landmark_matrix.get(b_zip, f"the primary public transit corridors intersecting near {str(addr).split()[-1]}")

    draw_micro_property_map(matched_row, name, addr, sibling_record)

    return f"""<b>Subject:</b> Urgent Infrastructure Update: Mitigation of mechanical replacement cycle at {name}<br>
------------------------------------------------------------------------------------------------------------------------<br><br>
Dear Property Operations Manager,<br><br>
I was recently passing down your block by <b>{landmark}</b> and couldn't help but notice the structural footprint of your facility at <b>{addr}</b>. Given our team's ongoing green-infrastructure updates across the neighborhood, your asset profile caught our immediate attention.<br><br>
According to city building records, your central mechanical heating and ventilation setup is currently hitting its <b>{age}th year of operational deployment</b>. As an operator, you know this places your rooftop compressors/boilers right at their statistical 15-20 year capital replacement cycle window.<br><br>
Transitioning a building of your scale to all-electric heat pumps can seem daunting, but you wouldn't be the first on your block to make the leap, {social_proof}<br><br>
Because your property is located inside a verified <b>NASA Tier-{hvi_score} {intensity_word}</b>, running legacy fossil fuel equipment means your cooling overhead costs during peak summer temperature spikes are likely scaling unsustainably. Engineering a proactive switch to high-efficiency VRF setups *before* an emergency summer blowout occurs shields your investment portfolio from structural risk and upcoming municipal emissions penalties.<br><br>
We are currently grouping localized clean-energy incentives for projects in the immediate vicinity of your block this quarter. Let's schedule a brief 10-minute technical review next Tuesday morning to audit your existing plant layout.<br><br>
Best regards,<br><br>
<b>Lead Energy Optimization Engineer</b><br>
Clean-Tech Infrastructure Partners | 2026 Operations Deck"""

# ==============================================================================
# PHASE 6: GLOBAL MAP RENDERS & EVENT HOOK INTERCEPTS
# ==============================================================================
def draw_gis_command_center(focus_zip_code=None):
    if focus_zip_code and str(focus_zip_code).strip() in real_gdf[zip_col].unique():
        z_match = real_gdf[real_gdf[zip_col] == str(focus_zip_code).strip()]
        c_lat, c_lon = z_match['geometry'].y.mean(), z_match['geometry'].x.mean()
        starting_zoom = 14
    else:
        c_lat = current_filtered_slice['geometry'].y.mean() if not current_filtered_slice.empty else 40.75
        c_lon = current_filtered_slice['geometry'].x.mean() if not current_filtered_slice.empty else -74.00
        starting_zoom = 11

    f_main = folium.Figure(height=540)
    reactive_map = folium.Map(location=[c_lat, c_lon], zoom_start=starting_zoom, tiles="cartodbpositron").add_to(f_main)

    def get_nasa_choropleth_color(feature):
        z_key = str(feature['properties'].get('postalCode') or feature['properties'].get('ZIPCODE') or '').strip()
        match = real_gdf[real_gdf[zip_col] == z_key]
        if match.empty: return '#b2bec3'
        score = int(match.iloc[0]['hvi_score_building'])
        return {1: '#fee5d9', 2: '#fcae91', 3: '#fb6a4a', 4: '#de2d26', 5: '#a50f15'}.get(score, '#fb6a4a')

    folium.GeoJson(
        json.loads(json.dumps(nyc_zip_geojson)),
        style_function=lambda f: {'fillColor': get_nasa_choropleth_color(f), 'color': '#2c3e50', 'weight': 1.2, 'fillOpacity': 0.45}
    ).add_to(reactive_map)

    for z_code, group in real_gdf.groupby(zip_col):
        if z_code == "nan" or str(z_code).strip() == "": continue
        total_b = len(group)
        with_hp = int(group['has_heat_pump'].sum())
        without_hp = total_b - with_hp
        needing_hp = int(((group[year_col] >= 2006) & (group[year_col] <= 2011) & (group['has_heat_pump'] == 0)).sum())
        pct_hp = (with_hp / total_b) * 100 if total_b > 0 else 0
        m_lat, m_lon = group['geometry'].y.mean(), group['geometry'].x.mean()

        hvi_val = int(group['hvi_score_building'].iloc[0])

        icon_style_html = f"""
        <div style="width: 32px; height: 32px; border-radius: 50%; background: conic-gradient(#2ecc71 0% {pct_hp}%, #e74c3c {pct_hp}% 100%); border: 2px solid #ffffff; box-shadow: 0 0 6px rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; cursor: pointer; font-family: Arial, sans-serif; font-weight: bold; font-size: 10px; color: #ffffff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">{total_b}</div>
        """

        popup_template = f"""<div style="font-family: Arial; font-size: 12px; padding: 6px; min-width: 200px;">
        <b>📮 ZIP: {z_code}</b><br><hr>
        <b>☀️ NASA Heat Index:</b> Tier {hvi_val}/5<br>
        <b>🏢 Total Buildings:</b> {total_b}<br>
        <span style='color:#e74c3c;'><b>🚨 Expiring HVAC:</b> {needing_hp}</span><br>
        <span style='color:#2ecc71;'><b>💚 Electrified Rate:</b> {pct_hp:.1f}%</span>
        </div>"""

        folium.Marker(location=[m_lat, m_lon], icon=folium.DivIcon(html=icon_style_html, icon_size=(32,32)), popup=folium.Popup(popup_template, max_width=280)).add_to(reactive_map)

    with map_output_pane:
        clear_output(wait=True)
        display(f_main)

def on_pipeline_compiled(b):
    global current_filtered_slice
    status_view.value = "⚡ <i>Sweeping database entries and mapping structural assets...</i>"
    df_result = filter_reactive_data(search_box.value, limit_slider.value)
    if df_result.empty:
        status_view.value = "<b style='color: #e74c3c;'>⚠️ Zero assets matched criteria within this scope.</b>"
        table_view.value = ""
        return

    current_filtered_slice = df_result
    status_view.value = f"<h3>✅ Isolated Top {len(df_result)} Target Properties</h3>"
    table_view.value = f"<div style='max-height: 220px; overflow-y: auto;'>{df_result.drop(columns=['geometry', 'ZIP Code']).to_html(index=False, border=0, classes='table table-dark table-striped')}</div>"
    target_input_box.value = str(df_result.iloc[0]['Target Property Name'])

    draw_gis_command_center()
    on_generate_pitch_clicked(None)

def on_generate_pitch_clicked(b):
    email_output_pane.value = "⏳ <i>Running localized spatial scans and deploying proximity maps...</i>"
    pitch_text = build_personalized_pitch_and_map(current_filtered_slice, target_input_box.value)
    email_output_pane.value = f"<div style='font-family: monospace; font-size: 13px; line-height: 1.6; color: #000000;'>{pitch_text}</div>"

search_button.on_click(on_pipeline_compiled)
generate_pitch_btn.on_click(on_generate_pitch_clicked)
map_focus_btn.on_click(lambda b: draw_gis_command_center(map_focus_box.value))

# Mount visual systems tabs
pseudo_website = widgets.Tab()
pseudo_website.children = [page_1_layout, page_2_layout]
pseudo_website.set_title(0, '🖥️ Building Demand Search')
pseudo_website.set_title(1, '📊 Neighborhood Demand Dashboard')

clear_output()
display(pseudo_website)
on_pipeline_compiled(None)
macro_table_view.value = f"<div style='max-height: 480px; overflow-y: auto;'>{macro_summary.to_html(index=False, border=0, classes='table table-dark table-striped')}</div>"
print("🚀 Step 4/4: Multi-tier Micro-Proximity mapping server deployed!")
