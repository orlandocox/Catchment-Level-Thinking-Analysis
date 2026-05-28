import streamlit as st
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import os
from datetime import datetime
from shapely.ops import substring
import io

st.set_page_config(page_title="INNS Catchment Strategy Tool", layout="wide")
st.title("🌊 INNS Catchment Prioritisation & Strategy Tool")
st.markdown("Use this interface to configure and generate catchment work blocks for GIS deployment.")

# --- 1. DIRECTORY SETUP ---
INPUT_DIR = "Input_Data"
OUTPUT_DIR = "Output_Data"
for folder in [INPUT_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("🔧 Analysis Parameters")

available_files = os.listdir(INPUT_DIR) if os.path.exists(INPUT_DIR) else []
gpkg_files = [f for f in available_files if f.endswith('.gpkg')]

river_file_name = st.sidebar.selectbox("Select OS Water Network File", gpkg_files, index=gpkg_files.index("OS_Water_Network.gpkg") if "OS_Water_Network.gpkg" in gpkg_files else 0)
inns_file_name = st.sidebar.selectbox("Select INNS Reports File", gpkg_files, index=gpkg_files.index("INNS_Reports.gpkg") if "INNS_Reports.gpkg" in gpkg_files else 0)

RIVER_FILE = os.path.join(INPUT_DIR, river_file_name) if river_file_name else ""
INNS_FILE = os.path.join(INPUT_DIR, inns_file_name) if inns_file_name else ""

MAX_SEGMENT_LENGTH = st.sidebar.slider("Work Block Size (Meters)", min_value=250, max_value=5000, value=1000, step=250)
BUFFER_DIST = st.sidebar.slider("INNS Search Buffer (Meters)", min_value=50, max_value=1000, value=250, step=50)

# Load dynamic species options
base_species_list = []
if inns_file_name and os.path.exists(INNS_FILE):
    try:
        inns_peek = gpd.read_file(INNS_FILE, ignore_geometry=True, engine="pyogrio")
        if 'species' in inns_peek.columns:
            base_species_list = sorted(inns_peek['species'].dropna().unique().tolist())
    except Exception:
        base_species_list = ["impatiens_glandulifera", "heracleum_mantegazzianum", "fallopia_japonica"]

species_options = ["All Species (Run Individually)"] + base_species_list
SPECIES_SELECTION = st.sidebar.selectbox("Species Target", options=species_options, index=0)

current_year = datetime.now().year
YEAR_FILTER = st.sidebar.number_input("Filter Start Year", min_value=2000, max_value=current_year, value=2015, step=1)
USE_YEAR_RANGE = st.sidebar.checkbox("Include all years from this start year onwards?", value=True)

run_analysis = st.sidebar.button("🚀 Run Strategic Analysis", type="primary")

# --- 3. CORE PROCESSING ENGINE ---
def split_line(line, max_dist):
    if line.length <= max_dist:
        return [line]
    num_segments = int(np.ceil(line.length / max_dist))
    segment_length = line.length / num_segments
    return [substring(line, i * segment_length, (i + 1) * segment_length) for i in range(num_segments)]

if run_analysis:
    if not river_file_name or not inns_file_name:
        st.error("Please ensure both input GeoPackages are selected.")
        st.stop()

    with st.spinner("Running high-speed network and spatial calculation..."):
        # Load datasets
        rivers_base = gpd.read_file(uploaded_river, engine="pyogrio").to_crs(27700)
        all_inns = gpd.read_file(INNS_FILE, engine="pyogrio").to_crs(27700)

        # Segmenting base lines
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

        rivers = gpd.GeoDataFrame(segmented_rows, crs=27700).reset_index(drop=True)
        rivers['UniqueID'] = rivers['id']
        rivers['Fnode'] = rivers['start_node']
        rivers['Tnode'] = rivers['end_node']

        # Year Filtering
        all_inns['year_val'] = pd.to_numeric(all_inns['date'].astype(str).str[:4], errors='coerce')
        if USE_YEAR_RANGE:
            all_inns = all_inns[all_inns['year_val'] >= YEAR_FILTER]
        else:
            all_inns = all_inns[all_inns['year_val'] == YEAR_FILTER]

        # Determine loops based on user choice
        if SPECIES_SELECTION == "All Species (Run Individually)":
            species_to_run = base_species_list
        else:
            species_to_run = [SPECIES_SELECTION]

        # Buffered river geom for spatial joins
        river_geom = rivers[['geometry']].copy()
        river_geom['geometry'] = river_geom.geometry.buffer(BUFFER_DIST)

        # --- BATCH RUN LOOP ---
        for target_species in species_to_run:
            # Short clean code names for columns (e.g., "Himalayan Balsam" -> "himalayan_balsam")
            clean_name = target_species.lower().replace(" ", "_")[:15]
            
            count_col = f"{clean_name}_cnt"
            tier_col = f"{clean_name}_tier"
            risk_col = f"{clean_name}_risk_km"
            prot_col = f"{clean_name}_protector"

            # Filter for this specific loop
            species_inns = all_inns[all_inns['species'] == target_species].copy()

            # Spatial join count
            if not species_inns.empty:
                joined = gpd.sjoin(river_geom, species_inns, how="left", predicate="intersects")
                rivers[count_col] = joined.groupby(joined.index).size() - joined.groupby(joined.index)['index_right'].apply(lambda x: x.isnull().sum())
            else:
                rivers[count_col] = 0

            # Graph building for this species
            G = nx.DiGraph()
            for idx, row in rivers.iterrows():
                G.add_edge(str(row['Fnode']), str(row['Tnode']), obj_id=row['UniqueID'], inns=row[count_col], length=row.geometry.length)

            # Default assignments
            rivers[tier_col] = 5
            rivers[risk_col] = 0.0
            rivers[prot_col] = 0

            # Calculate network metrics
            infested_indices = rivers.index[rivers[count_col] > 0]
            for idx in infested_indices:
                row = rivers.loc[idx]
                u_node, v_node = str(row['Fnode']), str(row['Tnode'])
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
                fn = str(row['Fnode'])
                if fn in G and any(data['inns'] > 0 for _, _, data in G.in_edges(fn, data=True)):
                    rivers.at[idx, prot_col] = 1

        # --- EXPORT & STATE PERSISTENCE ---
        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        current_output_path = os.path.join(OUTPUT_DIR, run_date)
        os.makedirs(current_output_path, exist_ok=True)
        
        file_species_string = "Multi_Species" if SPECIES_SELECTION == "All Species (Run Individually)" else SPECIES_SELECTION
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
        st.session_state['total_sightings'] = len(all_inns)

# --- 4. DISPLAY METRICS & SUMMARY ---
if 'rivers_result' in st.session_state:
    rivers = st.session_state['rivers_result']
    species_list = st.session_state['species_run_list']
    
    st.success("🎉 Multi-Species Strategic Profiles Generated!")
    
    st.subheader("📥 Download Strategic GIS Layers")
    st.download_button(
        label="💾 Download Comprehensive GeoPackage (.gpkg)",
        data=st.session_state['download_bytes'],
        file_name=st.session_state['file_name'],
        mime="application/geopackage+sqlite3",
        type="primary"
    )
    st.info(f"📁 Network-wide file archived at: `{st.session_state['file_path']}`")

    st.markdown("---")
    st.subheader("📊 Individual Species Metrics")

    # Display an analytical summary table for every species executed in the run
    for spec in species_list:
        clean_name = spec.lower().replace(" ", "_")[:15]
        tier_col = f"{clean_name}_tier"
        prot_col = f"{clean_name}_protector"
        
        with st.expander(f"👁️ View Strategic Summary for: {spec.upper()}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                p1_count = len(rivers[rivers[tier_col] == 1])
                protectors = int(rivers[prot_col].sum())
                st.metric("Priority 1 Alpha Fronts", f"{p1_count}")
                st.metric("Critical Clean Protectors", f"{protectors}")
                
            with col2:
                summary_df = rivers[tier_col].value_counts().sort_index().reset_index()
                summary_df.columns = ['Strategic Tier', 'Segments Found']
                labels = {1: "Priority 1 (Alpha Source)", 2: "Priority 2", 3: "Priority 3", 4: "Priority 4", 5: "Priority 5 (Clean / Out of Scope)"}
                summary_df['Description'] = summary_df['Strategic Tier'].map(labels)
                st.table(summary_df[['Strategic Tier', 'Description', 'Segments Found']])
else:
    st.info("👈 Use the parameters on the left sidebar and click **Run Strategic Analysis** to generate the data.")
