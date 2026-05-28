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
st.title("🌊 INNS Catchment Prioritisation & Strategy Tool")
st.markdown("Use this interface to configure and generate catchment work blocks for GIS deployment.")

# --- 2. DIRECTORY SETUP & STATIC PATHS ---
INPUT_DIR = "Input_Data"
OUTPUT_DIR = "Output_Data"
RIVER_TEMPLATE = "Template_Data/OS_Water_Network_Template.zip"
INNS_TEMPLATE = "Template_Data/INNS_Reports.gpkg"

for folder in [INPUT_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- 3. SIDEBAR CONFIGURATION (INPUT PANEL) ---
st.sidebar.header("📁 1. Data Ingestion")

uploaded_river = st.sidebar.file_uploader("Override OS Water Network (.gpkg or .zip)", type=["gpkg", "zip"])
uploaded_inns = st.sidebar.file_uploader("Override INNS Reports (.gpkg)", type=["gpkg"])

st.sidebar.markdown("### 🔍 Active Layer Status")
if uploaded_river is not None:
    st.sidebar.success("🟢 Network: Custom File Uploaded")
elif os.path.exists(RIVER_TEMPLATE):
    st.sidebar.info("🔵 Network: Using Default Repository Template")
else:
    st.sidebar.warning("⚠️ Network: Missing Base Framework")

if uploaded_inns is not None:
    st.sidebar.success("🟢 INNS Data: Custom File Uploaded")
elif os.path.exists(INNS_TEMPLATE):
    st.sidebar.info("🔵 INNS Data: Using Default Repository Template")
else:
    st.sidebar.warning("⚠️ INNS Data: Missing Survey Information")

st.sidebar.markdown("---")
st.sidebar.header("🔧 2. Strategy Tuners")

MAX_SEGMENT_LENGTH = st.sidebar.slider("Target Work Block Length (m)", min_value=250, max_value=5000, value=1000, step=250)
BUFFER_DIST = st.sidebar.slider("Buffer Search Envelope (m)", min_value=50, max_value=1000, value=250, step=50)

# Extract species list contextually without hogging memory
base_species_list = ["impatiens_glandulifera", "heracleum_mantegazzianum", "fallopia_japonica"]
active_inns_source = uploaded_inns if uploaded_inns is not None else (INNS_TEMPLATE if os.path.exists(INNS_TEMPLATE) else None)

if active_inns_source is not None:
    try:
        inns_peek = gpd.read_file(active_inns_source, ignore_geometry=True, engine="pyogrio")
        if 'species' in inns_peek.columns:
            base_species_list = sorted(inns_peek['species'].dropna().unique().tolist())
    except Exception:
        pass

# REMOVED "All Species" option to safeguard container hardware resources
SPECIES_SELECTION = st.sidebar.selectbox("Species Target Filter", options=base_species_list, index=0)

current_year = datetime.now().year
YEAR_FILTER = st.sidebar.number_input("Survey Baseline Horizon Year", min_value=2000, max_value=current_year, value=2015, step=1)
USE_YEAR_RANGE = st.sidebar.checkbox("Include subsequent record entries to present date?", value=True)

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Strategic Analysis", type="primary", use_container_width=True)

# --- 4. INTERACTIVE DOCUMENTATION & USER MANUAL ---
doc_tab, engine_tab = st.tabs(["📖 User Manual & Methodology", "💻 Analytics Hub"])

with doc_tab:
    st.header("📘 Catchment Thinking Optimization Guide")
    st.markdown("""
    Welcome to the **INNS Catchment Strategy Tool**. This system uses directed graph algorithms 
    to organize invasive species field operations. By evaluating river segments from headwaters to sea, 
    it identifies exactly where to intervene to stop downstream re-infestation.
    """)
    
    with st.expander("🔍 Step 1: Input Data Requirements", expanded=True):
        st.markdown("""
        The engine accepts custom GIS files via the sidebar. If none are provided, it automatically falls back to default preloaded datasets. If you use custom overrides, ensure they meet these constraints:
        * **OS Water Network Link Geometry:** Must be provided in **British National Grid (EPSG:27700)**. The layer requires structural connectivity identifiers, specifically an edge ID (`id`), a starting point node (`start_node`), and a terminating point node (`end_node`).
        * **INNS Survey Reports:** A spatial GeoPackage (`.gpkg`) layer containing species observation coordinates. The attribute table must contain a text column titled `species` and a temporal column titled `date` (formatted cleanly as `YYYY-MM-DD` or starting with a 4-digit year string).
        """)

    with st.expander("🎛️ Step 2: Understanding Side Panel Parameters", expanded=False):
        st.markdown("""
        Adjusting the sidebar configurations fundamentally shifts how your field operations are grouped and evaluated:
        1. **Target Work Block Length (meters):** Long, continuous river reaches are split into standardized management stretches. Setting this to `1000m` means a continuous 5km river section will be neatly partitioned into 5 independent operational zones.
        2. **Buffer Search Envelope (meters):** Survey records rarely snap perfectly to a river centerline due to GPS variance. This parameter builds a temporary lateral buffer around the channel to grab nearby observations. If weeds grow far up the banks, increase this value to ensure they are captured.
        3. **Survey Horizon Year:** Allows you to isolate recent data. Setting this to `2015` with the subsequent checkbox active will completely ignore historical data from 2014 and older, focusing exclusively on active modern threats.
        """)

    with st.expander("👑 Step 3: Deciphering Strategic Output Classifications", expanded=False):
        st.markdown("""
        When the calculation finishes, every river reach is assigned a management **Tier** from 1 to 5. These tiers indicate how you should prioritize field labor:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            * **Tier 1 (Alpha Source):** Reaches containing active target populations with **zero** identified infestations anywhere upstream. **Action item: Treat immediately.** Eradicating these removes the root seed source.
            * **Tier 2:** Infested reaches with exactly one active cluster located upstream. These are your immediate secondary objectives.
            * **Tier 3 & 4:** Mid-catchment and terminal channels choked by multiple upstream source populations. Postpone operations here until upstream sources are cleared, as these zones face constant re-infestation pressure.
            """)
        with col2:
            st.markdown("""
            * **Tier 5 (Clean Corridors):** Safe zones where no target species were found. No remediation action required.
            * **Critical Clean Protectors:** Clean river reaches located **directly downstream** of an active infestation. These act as your environmental line in the sand—if field teams do not monitor these points, the upstream infestation will soon move into clear water.
            * **Downstream Risk (km):** The length of continuous uninfested river corridor extending below a Tier 1 source. Reaches with higher numbers should be prioritized first, as clearing them protects a larger downstream area.
            """)

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
        
        # --- PHASE A: MEMORY-SAFE HYDRO INFRASTRUCTURE INGESTION ---
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

        # --- PHASE B: INNS RECORD PROCESSING ---
        progress_bar.progress(30, text="Parsing environmental spatial records database...")
        if uploaded_inns is not None:
            all_inns = gpd.read_file(uploaded_inns, engine="pyogrio").to_crs(27700)
        else:
            all_inns = gpd.read_file(INNS_TEMPLATE, engine="pyogrio").to_crs(27700)

        all_inns['year_val'] = pd.to_numeric(all_inns['date'].astype(str).str[:4], errors='coerce')
        if USE_YEAR_RANGE:
            all_inns = all_inns[all_inns['year_val'] >= YEAR_FILTER]
        else:
            all_inns = all_inns[all_inns['year_val'] == YEAR_FILTER]

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

        # Configured context cleanly for isolated target runs only
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
        
        st.success("🎉 Strategic Operational Profiles Generated!")
        
        st.subheader("📥 Export Prioritised GIS Vector Data")
        st.markdown("""
        Click the download button below to save your generated model. 
        Import this `.gpkg` file into desktop software like QGIS or ArcGIS Pro to map out your catchment works.
        """)
        
        st.download_button(
            label="💾 Download Comprehensive Strategic GeoPackage (.gpkg)",
            data=st.session_state['download_bytes'],
            file_name=st.session_state['file_name'],
            mime="application/geopackage+sqlite3",
            type="primary"
        )
        
        with st.expander("📝 Attribute Dictionary (How to style your GIS layers)"):
            st.markdown("""
            When you open the attribute table of the downloaded GeoPackage, you will find columns dynamically generated for each species run (using the template prefix `[species_name]_...`):
            * **`_cnt` (Count):** Integer showing the exact number of survey points that intersected this segment.
            * **`_tier` (Action Priority):** Values from 1 to 5. Style with a categorical color ramp (e.g., Tier 1 as bright red, Tier 5 as light blue) to easily identify your priority targets.
            * **`_risk_km` (Downstream Risk Value):** Floating point number indicating the kilometers of clean river network lying downstream. Sort descending on this column within Tier 1 reaches to rank your highest-stakes targets.
            * **`_protector` (Buffer Shield Flag):** Binary switch (`1` or `0`). Filter for rows where this is `1` to highlight clean segments that directly border upstream infestations.
            """)

        st.markdown("---")
        st.subheader("📊 Analytical Performance Metrics by Species")

        for spec in species_list:
            clean_name = spec.lower().replace(" ", "_")[:15]
            tier_col = f"{clean_name}_tier"
            prot_col = f"{clean_name}_protector"
            
            with st.expander(f"👁️ View Strategic Summary Metrics: {spec.upper()}", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    p1_count = len(rivers[rivers[tier_col] == 1]) if tier_col in rivers.columns else 0
                    protectors = int(rivers[prot_col].sum()) if prot_col in rivers.columns else 0
                    st.metric(label="Priority 1 Alpha Targets", value=f"{p1_count} Reaches")
                    st.metric(label="Critical Clean Protectors", value=f"{protectors} Reaches")
                    
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
        st.info("👈 Set structural layer limits in the left input configurations sidebar panel and click **Run Strategic Analysis**.")
