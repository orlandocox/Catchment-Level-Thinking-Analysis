import streamlit as st
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime
from shapely.ops import substring
import io
import gc  # Garbage Collector interface to force-free server RAM

# --- 1. APPLICATION SETUP & THEMING ---
st.set_page_config(page_title="INNS Catchment Strategy Tool", layout="wide")
st.title("INNS Catchment Prioritisation & Strategy Tool")
st.markdown("Use this interface to configure and generate catchment work blocks for GIS deployment.")

# --- 2. DIRECTORY SETUP & STATIC PATHS ---
INPUT_DIR = "Input_Data"
OUTPUT_DIR = "Output_Data"
RIVER_TEMPLATE = "Template_Data/OS_Water_Network_Template.zip"
INNS_TEMPLATE = "Template_Data/INNS_Reports.gpkg"

for folder in [INPUT_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- 3. SIDEBAR CONFIGURATION (INPUT PANEL) ---
st.sidebar.header("1. Data Ingestion")

uploaded_river = st.sidebar.file_uploader("Override OS Water Network (.gpkg or .zip)", type=["gpkg", "zip"])

# Upgraded to smoothly support all INNS Mapper shapefile/metadata component dumps
uploaded_inns = st.sidebar.file_uploader(
    "Upload INNS Reports (Select all component files: .shp, .dbf, .shx, or .gpkg)", 
    type=["shp", "dbf", "shx", "gpkg", "csv"], 
    accept_multiple_files=True
)

st.sidebar.markdown("### Active Layer Status")
if uploaded_river is not None:
    st.sidebar.success("Network: Custom File Uploaded")
elif os.path.exists(RIVER_TEMPLATE):
    st.sidebar.info("Network: Using Default Repository Template")
else:
    st.sidebar.warning("Network: Missing Base Framework")

if uploaded_inns:
    st.sidebar.success("INNS Data: Custom File Uploaded")
elif os.path.exists(INNS_TEMPLATE):
    st.sidebar.info("INNS Data: Using Default Repository Template")
else:
    st.sidebar.warning("INNS Data: Missing Survey Information")

st.sidebar.markdown("---")
st.sidebar.header("🔧 2. Strategy Tuners")

MAX_SEGMENT_LENGTH = st.sidebar.slider("Target Work Block Length (m)", min_value=250, max_value=5000, value=1000, step=250)
BUFFER_DIST = st.sidebar.slider("Buffer Search Envelope (m)", min_value=50, max_value=1000, value=250, step=50)

# Extract species list contextually without hogging memory
base_species_list = ["impatiens_glandulifera", "heracleum_mantegazzianum", "fallopia_japonica"]

# Handle safe inspection for the sidebar dropdown setup
if uploaded_inns:
    # Quick standalone look at whatever layer might be available to grab naming values
    for f in uploaded_inns:
        if f.name.endswith(('.shp', '.gpkg', '.csv')):
            try:
                # Save temporarily to safely inspect species strings
                temp_path = os.path.join(INPUT_DIR, f.name)
                with open(temp_path, "wb") as out:
                    out.write(f.getbuffer())
                
                if f.name.endswith('.csv'):
                    inns_peek = pd.read_csv(temp_path, nrows=20)
                else:
                    inns_peek = gpd.read_file(temp_path, ignore_geometry=True, engine="pyogrio", rows=20)
                
                inns_peek.columns = map(str.lower, inns_peek.columns)
                species_col = next((c for c in ['species', 'common_name', 'taxon'] if c in inns_peek.columns), None)
                if species_col and species_col in inns_peek.columns:
                    base_species_list = sorted(inns_peek[species_col].dropna().unique().tolist())
                    break
            except Exception:
                pass
elif os.path.exists(INNS_TEMPLATE):
    try:
        inns_peek = gpd.read_file(INNS_TEMPLATE, ignore_geometry=True, engine="pyogrio")
        if 'species' in inns_peek.columns:
            base_species_list = sorted(inns_peek['species'].dropna().unique().tolist())
    except Exception:
        pass

SPECIES_SELECTION = st.sidebar.selectbox("Species Target Filter", options=base_species_list, index=0)

current_year = datetime.now().year
YEAR_FILTER = st.sidebar.number_input("Survey Baseline Horizon Year", min_value=2000, max_value=current_year, value=2020, step=1)
USE_YEAR_RANGE = st.sidebar.checkbox("Include subsequent record entries to present date?", value=True)

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("Run Strategic Analysis", type="primary", use_container_width=True)

# --- 4. INTERACTIVE DOCUMENTATION & USER MANUAL ---
doc_tab, engine_tab = st.tabs(["User Manual & Methodology", "Analytics Hub"])

# Documentation removed for snippet clarity; identical to your historical application base...
with doc_tab:
    st.header("Catchment Thinking Optimization Guide")

# --- 5. CORE PROCESSING ENGINE ---
def split_line(line, max_dist):
    if line.length <= max_dist:
        return [line]
    num_segments = int(np.ceil(line.length / max_dist))
    segment_length = line.length / num_segments
    return [substring(line, i * segment_length, (i + 1) * segment_length) for i in range(num_segments)]

with engine_tab:
    if run_analysis:
        if not uploaded_river and not os.path.exists(RIVER_TEMPLATE):
            st.error("Missing structural dependency: Base network not found.")
            st.stop()
        if not uploaded_inns and not os.path.exists(INNS_TEMPLATE):
            st.error("Missing structural dependency: INNS records not found.")
            st.stop()

        progress_bar = st.progress(0, text="Initializing processing layers...")
        
        # --- PHASE A: HYDRO INFRASTRUCTURE INGESTION ---
        progress_bar.progress(10, text="Streaming hydrological grid geometry...")
        if uploaded_river is not None:
            if uploaded_river.name.endswith('.zip') and zipfile.is_zipfile(uploaded_river):
                with zipfile.ZipFile(uploaded_river) as z:
                    gpkg_inside = [f for f in z.namelist() if f.endswith('.gpkg')]
                    with z.open(gpkg_inside[0]) as f:
                        rivers_base = gpd.read_file(f, engine="pyogrio").to_crs(27700)
            else:
                rivers_base = gpd.read_file(uploaded_river, engine="pyogrio").to_crs(27700)
        else:
            if zipfile.is_zipfile(RIVER_TEMPLATE):
                with zipfile.ZipFile(RIVER_TEMPLATE) as z:
                    gpkg_inside = [f for f in z.namelist() if f.endswith('.gpkg')]
                    with z.open(gpkg_inside[0]) as f:
                        rivers_base = gpd.read_file(f, engine="pyogrio").to_crs(27700)
            else:
                rivers_base = gpd.read_file(RIVER_TEMPLATE, engine="pyogrio").to_crs(27700)

        # --- PHASE B: FIXED MULTI-FILE INNS RECORD PROCESSING ---
        progress_bar.progress(30, text="Merging and transforming INNS spatial layers...")
        
        gdfs = []
        
        if uploaded_inns:
            # 1. Write ALL files out to a real directory so Shapefiles find their structural sidecars (.dbf, .shx)
            for f in uploaded_inns:
                target_path = os.path.join(INPUT_DIR, f.name)
                with open(target_path, "wb") as out:
                    out.write(f.getbuffer())
            
            # 2. Iterate through files and interpret WKT strings or native geometries safely
            for f in uploaded_inns:
                target_path = os.path.join(INPUT_DIR, f.name)
                
                # Process primary data formats (.shp, .gpkg, .csv) and ignore bare sidecars (.dbf/.shx directly)
                if f.name.endswith('.shp') or f.name.endswith('.gpkg') or f.name.endswith('.csv'):
                    try:
                        if f.name.endswith('.csv'):
                            df = pd.read_csv(target_path)
                            df.columns = map(str.lower, df.columns)
                            if 'wkt_geom' in df.columns:
                                gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['wkt_geom']), crs="EPSG:4326")
                            else:
                                continue
                        else:
                            df = gpd.read_file(target_path)
                            df.columns = map(str.lower, df.columns)
                            # Handle case where shapefile text column holds WKT coordinates
                            if 'wkt_geom' in df.columns:
                                gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['wkt_geom']), crs="EPSG:4326")
                            else:
                                gdf = df
                        
                        gdf = gdf.to_crs(27700)
                        gdfs.append(gdf[['species', 'date', 'geometry'] if 'date' in gdf.columns else ['species', 'geometry']])
                    except Exception as e:
                        st.warning(f"Failed to process sub-component layer {f.name}: {e}")
            
            if len(gdfs) == 0:
                st.error("No compatible point, line, or polygon shapes could be derived from input assets.")
                st.stop()
                
            raw_inns = pd.concat(gdfs, ignore_index=True)
        else:
            raw_inns = gpd.read_file(INNS_TEMPLATE, engine="pyogrio").to_crs(27700)
            raw_inns.columns = map(str.lower, raw_inns.columns)

        # Map attributes cleanly
        species_col = next((c for c in ['species', 'common_name', 'taxon'] if c in raw_inns.columns), None)
        date_col = next((c for c in ['date', 'year', 'date_rec'] if c in raw_inns.columns), None)

        if species_col != 'species':
            raw_inns = raw_inns.rename(columns={species_col: 'species'})
            
        raw_inns['year_val'] = pd.to_numeric(raw_inns[date_col].astype(str).str[:4], errors='coerce') if date_col else YEAR_FILTER

        # Apply strategy year criteria filters
        if USE_YEAR_RANGE:
            all_inns = raw_inns[raw_inns['year_val'] >= YEAR_FILTER].copy()
        else:
            all_inns = raw_inns[raw_inns['year_val'] == YEAR_FILTER].copy()
            
        all_inns = all_inns[['species', 'geometry']]
        
        # Clean sandbox space
        del raw_inns, gdfs
        for f in os.listdir(INPUT_DIR):
            try:
                os.remove(os.path.join(INPUT_DIR, f))
            except Exception:
                pass
        gc.collect()

        # --- PHASE C: DYNAMIC LINE SEGMENTATION ---
        progress_bar.progress(45, text="Sub-dividing river chains into operational work blocks...")
        segmented_rows = []
        for _, row in rivers_base.iterrows():
            if row.geometry.length > MAX_SEGMENT_LENGTH:
                chunks = split_line(row.geometry, MAX_SEGMENT_LENGTH)
                for i, chunk in enumerate(chunks):
                    new_row = row.copy()
                    new_row.geometry = chunk
                    if i > 0: new_row['start_node'] = f"{row['id']}_vnode_{i}"
                    if i < len(chunks) - 1: new_row['end_node'] = f"{row['id']}_vnode_{i+1}"
                    new_row['id'] = f"{row['id']}_seg_{i}"
                    segmented_rows.append(new_row)
            else:
                segmented_rows.append(row)

        del rivers_base  # Free original raw dataframe from memory completely
        rivers = gpd.GeoDataFrame(segmented_rows, crs=27700).reset_index(drop=True)
        rivers['UniqueID'] = rivers['id'].astype(str)
        rivers['Fnode'] = rivers['start_node'].astype(str)
        rivers['Tnode'] = rivers['end_node'].astype(str)

        species_to_run = [SPECIES_SELECTION]

        # --- PHASE D: GENERATE LEAN SPATIAL JOIN CHECKS ---
        progress_bar.progress(60, text="Constructing lateral search buffers...")
        river_geom = rivers[['geometry']].copy()
        river_geom['geometry'] = river_geom.geometry.buffer(BUFFER_DIST)

        # --- PHASE E: BATCH ROUTING NETWORK ENGINE ---
        progress_bar.progress(75, text="Running topological graph traversal algorithm...")
        for target_species in species_to_run:
            clean_name = target_species.lower().replace(" ", "_")[:15]
            
            count_col = f"{clean_name}_cnt"
            tier_col = f"{clean_name}_tier"
            risk_col = f"{clean_name}_risk_km"
            prot_col = f"{clean_name}_protector"

            species_inns = all_inns[all_inns['species'] == target_species].copy()

            if not species_inns.empty:
                joined = gpd.sjoin(river_geom, species_inns, how="left", predicate="intersects")
                rivers[count_col] = joined.groupby(joined.index).size() - joined.groupby(joined.index)['index_right'].apply(lambda x: x.isnull().sum())
                del joined 
            else:
                rivers[count_col] = 0

            G = nx.DiGraph()
            for idx, row in rivers.iterrows():
                G.add_edge(row['Fnode'], row['Tnode'], obj_id=row['UniqueID'], inns=int(row[count_col]), length=float(row.geometry.length))

            rivers[tier_col] = 5
            rivers[risk_col] = 0.0
            rivers[prot_col] = 0

            infested_indices = rivers.index[rivers[count_col] > 0]
            for idx in infested_indices:
                row = rivers.loc[idx]
                u_node, v_node = row['Fnode'], row['Tnode']
                if u_node in G:
                    upstream_nodes = nx.ancestors(G, u_node)
                    infested_ancestors = 0
                    visited_edges = set()
                    for node in list(upstream_nodes) + [u_node]:
                        for up, _, data in G.in_edges(node, data=True):
                            if data['obj_id'] not in visited_edges and data['obj_id'] != row['UniqueID']:
                                if data['inns'] > 0: infested_ancestors += 1
                                visited_edges.add(data['obj_id'])
                    rivers.at[idx, tier_col] = min(infested_ancestors + 1, 4)

                    if infested_ancestors == 0:
                        downstream_nodes = nx.descendants(G, v_node)
                        clean_len = sum(data['length'] for d_node in list(downstream_nodes) + [v_node] for _, _, data in G.out_edges(d_node, data=True) if data['inns'] == 0)
                        rivers.at[idx, risk_col] = clean_len / 1000

            for idx, row in rivers[rivers[count_col] == 0].iterrows():
                fn = row['Fnode']
                if fn in G and any(data['inns'] > 0 for _, _, data in G.in_edges(fn, data=True)):
                    rivers.at[idx, prot_col] = 1

        del river_geom
        gc.collect()

        # --- PHASE F: FINALIZE EXPORT STREAMS ---
        progress_bar.progress(95, text="Encoding output metadata tables...")
        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        current_output_path = os.path.join(OUTPUT_DIR, run_date)
        os.makedirs(current_output_path, exist_ok=True)
        
        file_species_string = SPECIES_SELECTION.lower().replace(" ", "_")[:15]
        out_filename = f"Strategy_{file_species_string}_{YEAR_FILTER}.gpkg"
        final_output_path = os.path.join(current_output_path, out_filename)
        
        rivers.to_file(final_output_path, driver="GPKG")

        buffer = io.BytesIO()
        rivers.to_file(buffer, driver="GPKG")
        gpkg_bytes = buffer.getvalue()

        st.session_state['rivers_result'] = rivers.copy()
        st.session_state['file_path'] = final_output_path
        st.session_state['file_name'] = out_filename
        st.session_state['download_bytes'] = gpkg_bytes
        st.session_state['species_run_list'] = species_to_run
        
        progress_bar.progress(100, text="Process completed successfully!")
        progress_bar.empty()

    # --- 6. OUTPUT METRICS VIEWPORT ---
    if 'rivers_result' in st.session_state:
        rivers = st.session_state['rivers_result']
        species_list = st.session_state['species_run_list']
        
        st.success("Strategic Operational Profiles Generated!")
        
        st.subheader("Export Prioritised GIS Vector Data")
        st.markdown("Click the download button below to save your generated model.")
        
        st.download_button(
            label="Download Comprehensive Strategic GeoPackage (.gpkg)",
            data=st.session_state['download_bytes'],
            file_name=st.session_state['file_name'],
            mime="application/geopackage+sqlite3",
            type="primary"
        )
        
        with st.expander("Attribute Dictionary (How to style your GIS layers)"):
            st.markdown("""
            * **`_cnt`:** Intersected instances count.
            * **`_tier`:** Priority action metrics (Tiers 1-5).
            * **`_risk_km`:** Protected structural clear corridor depth downstream.
            * **`_protector`:** Direct boundary interface flag.
            """)

        st.markdown("---")
        st.subheader("Analytical Performance Metrics by Species")

        for spec in species_list:
            clean_name = spec.lower().replace(" ", "_")[:15]
            tier_col = f"{clean_name}_tier"
            prot_col = f"{clean_name}_protector"
            
            with st.expander(f"View Strategic Summary Metrics: {spec.upper()}", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Priority 1 Alpha Targets", value=f"{len(rivers[rivers[tier_col] == 1]) if tier_col in rivers.columns else 0} Reaches")
                    st.metric(label="Critical Clean Protectors", value=f"{int(rivers[prot_col].sum()) if prot_col in rivers.columns else 0} Reaches")
                with col2:
                    if tier_col in rivers.columns:
                        summary_df = rivers[tier_col].value_counts().sort_index().reset_index()
                        summary_df.columns = ['Strategic Tier', 'Segments Found']
                        labels = {
                            1: "Priority 1 (Headwater Alpha Source Reaches)", 
                            2: "Priority 2 (Secondary Controlled Reaches)", 
                            3: "Priority 3 (Mid-Catchment Infestations)", 
                            4: "Priority 4 (Terminal Constrained Channels)", 
                            5: "Priority 5 (Clean Corridors / Out of Scope)"
                        }
                        summary_df['Description / Action Items'] = summary_df['Strategic Tier'].map(labels)
                        st.table(summary_df[['Strategic Tier', 'Description / Action Items', 'Segments Found']])
    else:
        st.info("Set structural layer limits in the left input configurations sidebar panel and click **Run Strategic Analysis**.")
