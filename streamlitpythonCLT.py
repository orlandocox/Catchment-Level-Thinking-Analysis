import streamlit as st
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import os
import zipfile
import gc
import folium
from streamlit_folium import st_folium
from datetime import datetime
from shapely.ops import substring

# --- 1. SETUP & UI CONFIG ---
st.set_page_config(page_title="INNS Catchment Strategy", layout="wide")
st.title("INNS Catchment Prioritisation Tool")

# Ensure directories exist
for folder in ["Input_Data", "Output_Data"]:
    os.makedirs(folder, exist_ok=True)

RIVER_TMPL = "Template_Data/OS_Water_Network_Template.zip"
INNS_TMPL = "Template_Data/INNS_Reports.gpkg"

# --- 2. SIDEBAR PARAMETERS ---
with st.sidebar:
    st.header("1. Data Inputs")
    up_river = st.file_uploader("Network (.gpkg/.zip)", type=["gpkg", "zip"])
    up_inns = st.file_uploader("INNS Records (.gpkg)", type=["gpkg"])

    st.header("2. Strategy Settings")
    MAX_SEG = st.slider("Segment Length (m)", 250, 5000, 1000, 250)
    BUF_DIST = st.slider("Buffer Search (m)", 50, 1000, 250, 50)
    
    # Extract species dynamically for the dropdown
    species_opts = ["impatiens_glandulifera", "heracleum_mantegazzianum", "fallopia_japonica"]
    active_inns = up_inns if up_inns else (INNS_TMPL if os.path.exists(INNS_TMPL) else None)
    if active_inns:
        try:
            peek = gpd.read_file(active_inns, ignore_geometry=True, engine="pyogrio", rows=100)
            if 'species' in peek.columns: species_opts = sorted(peek['species'].dropna().unique().tolist())
        except Exception: pass

    TARGET_SPECIES = st.selectbox("Target Species", species_opts, index=0)
    YEAR_FILTER = st.number_input("Baseline Year", 2000, datetime.now().year, 2020)
    USE_RANGE = st.checkbox("Include records up to present?", True)
    
    st.divider()
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# --- 3. UI TABS ---
tab_how, tab_results, tab_strategy, tab_hub = st.tabs([
    "📖 How to Use", 
    "📊 Understanding Results", 
    "🎯 Catchment Strategy", 
    "⚙️ Analytics Hub"
])

with tab_how:
    st.markdown("""
    ### Quick Start Guide
    1. **Upload Data (Optional):** Use the sidebar to upload a custom river network or INNS records. If left blank, default templates are used.
    2. **Set Parameters:**
        * **Segment Length:** Breaks long rivers into manageable work blocks (e.g., 1000m stretches).
        * **Buffer Search:** How far from the river bank to search for species records (accounts for GPS inaccuracy).
        * **Baseline Year:** Filters out outdated historical records.
    3. **Select Target:** Pick the specific invasive species you want to model.
    4. **Run Engine:** Click **Run Analysis** in the sidebar.
    """)

with tab_results:
    st.markdown("""
    ### GIS Export Dictionary
    When you import the output `.gpkg` into QGIS or ArcGIS, style your layers using these generated attributes:
    * **`_tier` (Action Priority):** 1 to 5. Color-code these to easily see your primary targets (Tier 1) vs safe zones (Tier 5).
    * **`_cnt` (Record Count):** The number of actual survey observations that fell within this river segment.
    * **`_risk_km` (Downstream Risk):** For Tier 1 reaches, this shows how many kilometers of clean river are at immediate risk below it.
    * **`_protector` (Shield Flag):** A `1` means this clean segment sits directly below an infestation. Monitor these closely.
    """)

with tab_strategy:
    st.markdown("""
    ### How to Treat INNS Effectively
    Invasive species spread via water flow. Treating downstream populations while upstream sources remain active is ineffective as floods will re-infest cleared areas. 

    **Follow the Top-Down Approach:**
    * **Target Tier 1 First:** These are "Alpha Sources"—infestations at the very top of the catchment with *no* upstream sources. Eradicating these cuts off seed supply.
    * **Move to Tier 2:** Once Tier 1 is cleared, Tier 2 segments become the new Tier 1s.
    * **Ignore Tiers 3 & 4 (For Now):** These are heavily pressured mid-to-lower catchment areas. Do not spend budget here until upper catchments are under control.
    * **Defend Protectors:** Deploy monitoring teams to "Protector" reaches to ensure infestations haven't spread into clean corridors.
    """)

# --- 4. ENGINE LOGIC ---
def split_line(line, max_dist):
    if line.length <= max_dist: return [line]
    n = int(np.ceil(line.length / max_dist))
    length = line.length / n
    return [substring(line, i * length, (i + 1) * length) for i in range(n)]

with tab_hub:
    if run_btn:
        if not up_river and not os.path.exists(RIVER_TMPL): st.error("Missing River Network data."); st.stop()
        if not up_inns and not os.path.exists(INNS_TMPL): st.error("Missing INNS data."); st.stop()

        with st.spinner("Processing network topologies..."):
            # A. Load Rivers
            if up_river:
                if up_river.name.endswith('.zip'):
                    with zipfile.ZipFile(up_river) as z:
                        gpkg = [f for f in z.namelist() if f.endswith('.gpkg')][0]
                        with z.open(gpkg) as f: rivers = gpd.read_file(f, engine="pyogrio").to_crs(27700)
                else: rivers = gpd.read_file(up_river, engine="pyogrio").to_crs(27700)
            else:
                if zipfile.is_zipfile(RIVER_TMPL):
                    with zipfile.ZipFile(RIVER_TMPL) as z:
                        gpkg = [f for f in z.namelist() if f.endswith('.gpkg')][0]
                        with z.open(gpkg) as f: rivers = gpd.read_file(f, engine="pyogrio").to_crs(27700)
                else: rivers = gpd.read_file(RIVER_TMPL, engine="pyogrio").to_crs(27700)

            # B. Load & Filter INNS
            inns = gpd.read_file(up_inns or INNS_TMPL, engine="pyogrio").to_crs(27700)
            inns['year'] = pd.to_numeric(inns['date'].astype(str).str[:4], errors='coerce')
            inns = inns[inns['year'] >= YEAR_FILTER] if USE_RANGE else inns[inns['year'] == YEAR_FILTER]

            # C. Segment Rivers
            segs = []
            for _, r in rivers.iterrows():
                if r.geometry.length > MAX_SEG:
                    chunks = split_line(r.geometry, MAX_SEG)
                    for i, c in enumerate(chunks):
                        nr = r.copy()
                        nr.geometry = c
                        if i > 0: nr['start_node'] = f"{r['id']}_vnode_{i}"
                        if i < len(chunks)-1: nr['end_node'] = f"{r['id']}_vnode_{i+1}"
                        nr['id'] = f"{r['id']}_seg_{i}"
                        segs.append(nr)
                else: segs.append(r)
                
            del rivers
            rvrs = gpd.GeoDataFrame(segs, crs=27700).reset_index(drop=True)
            rvrs['uid'], rvrs['fn'], rvrs['tn'] = rvrs['id'].astype(str), rvrs['start_node'].astype(str), rvrs['end_node'].astype(str)

            # D. Spatial Join
            buf = rvrs[['geometry']].copy()
            buf['geometry'] = buf.geometry.buffer(BUF_DIST)
            
            pfx = TARGET_SPECIES.lower().replace(" ", "_")[:15]
            c_cnt, c_tier, c_rsk, c_prt = f"{pfx}_cnt", f"{pfx}_tier", f"{pfx}_risk_km", f"{pfx}_protector"
            
            spec_inns = inns[inns['species'] == TARGET_SPECIES]
            rvrs[c_cnt] = 0
            if not spec_inns.empty:
                jn = gpd.sjoin(buf, spec_inns, how="left", predicate="intersects")
                rvrs[c_cnt] = jn.groupby(jn.index).size() - jn.groupby(jn.index)['index_right'].apply(lambda x: x.isnull().sum())
            del buf; gc.collect()

            # E. Graph Analytics
            G = nx.DiGraph()
            for _, r in rvrs.iterrows(): G.add_edge(r['fn'], r['tn'], uid=r['uid'], i_cnt=int(r[c_cnt]), l=float(r.geometry.length))

            rvrs[c_tier], rvrs[c_rsk], rvrs[c_prt] = 5, 0.0, 0
            
            for idx in rvrs.index[rvrs[c_cnt] > 0]:
                u, v, uid = rvrs.at[idx, 'fn'], rvrs.at[idx, 'tn'], rvrs.at[idx, 'uid']
                ups = list(nx.ancestors(G, u)) + [u]
                
                inf_up = len(set(d['uid'] for n in ups for _, _, d in G.in_edges(n, data=True) if d['i_cnt'] > 0 and d['uid'] != uid))
                rvrs.at[idx, c_tier] = min(inf_up + 1, 4)

                if inf_up == 0:
                    dns = list(nx.descendants(G, v)) + [v]
                    rvrs.at[idx, c_rsk] = sum(d['l'] for n in dns for _, _, d in G.out_edges(n, data=True) if d['i_cnt'] == 0) / 1000.0

            for idx in rvrs.index[rvrs[c_cnt] == 0]:
                if rvrs.at[idx, 'fn'] in G and any(d['i_cnt'] > 0 for _, _, d in G.in_edges(rvrs.at[idx, 'fn'], data=True)):
                    rvrs.at[idx, c_prt] = 1

            # F. Disk Export & Session State
            out_name = f"Strategy_{pfx}_{YEAR_FILTER}.gpkg"
            out_dir = os.path.join("Output_Data", datetime.now().strftime("%Y-%m-%d_%H-%M"))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, out_name)
            
            # Export to file directly to avoid memory warnings/spikes
            rvrs.to_file(out_path, driver="GPKG")
            
            st.session_state.update({
                'res': rvrs, 'out_path': out_path, 'fname': out_name, 
                'spec': TARGET_SPECIES, 'cols': (c_tier, c_prt, c_cnt)
            })

    # --- 5. RESULTS UI & OPTIMIZED MAPPING ---
    if 'res' in st.session_state:
        df = st.session_state['res']
        t_col, p_col, cnt_col = st.session_state['cols']
        
        st.success("Analysis Complete.")
        
        # Top Metrics
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.metric("Priority 1 Targets (Alpha Sources)", f"{len(df[df[t_col] == 1])}")
        with c2:
            st.metric("Clean Protectors At Risk", f"{int(df[p_col].sum())}")
        with c3:
            with open(st.session_state['out_path'], "rb") as f:
                st.download_button(
                    "📥 Download GIS Data (.gpkg)", 
                    data=f.read(), 
                    file_name=st.session_state['fname'], 
                    type="primary", 
                    use_container_width=True
                )

        st.divider()

        # Generate Folium Map (Lightweight)
        st.subheader("Interactive Catchment Prioritisation Map")
        with st.spinner("Optimizing map layers..."):
            # 1. Project to Lat/Lon
            df_map = df.to_crs(4326).copy()
            
            # 2. SIMPLIFY GEOMETRY: Massive memory saver for Folium rendering
            df_map['geometry'] = df_map.geometry.simplify(0.0002, preserve_topology=True)
            
            # 3. Select strictly required columns for rendering
            df_map = df_map[[t_col, cnt_col, p_col, 'geometry']]
            
            def get_style(feature):
                tier = feature['properties'][t_col]
                if tier == 1: 
                    return {'color': '#FF0000', 'weight': 4, 'opacity': 0.9} # Tier 1 - Red
                elif tier == 2:
                    return {'color': '#FF8C00', 'weight': 3, 'opacity': 0.85} # Tier 2 - Orange
                elif tier in [3, 4]:
                    return {'color': '#FFD700', 'weight': 2, 'opacity': 0.75} # Tiers 3/4 - Yellow
                return {'color': '#1E90FF', 'weight': 1, 'opacity': 0.4} # Tier 5 - Blue

            bounds = df_map.total_bounds
            m = folium.Map(
                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2], 
                zoom_start=11, 
                tiles="CartoDB positron"
            )
            
            folium.GeoJson(
                df_map,
                style_function=get_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=[t_col, cnt_col, p_col], 
                    aliases=['Action Tier:', 'Record Count:', 'Protector Flag:'],
                    localize=True
                )
            ).add_to(m)
            
            st_folium(m, use_container_width=True, height=500, returned_objects=[])

        # Summary Table
        st.subheader("Segment Summary")
        smry = df[t_col].value_counts().sort_index().reset_index()
        smry.columns = ['Tier', 'Segments']
        smry['Action'] = smry['Tier'].map({1: "Alpha Source", 2: "Secondary", 3: "Mid-Catchment", 4: "Terminal", 5: "Clean"})
        st.dataframe(smry, hide_index=True, use_container_width=True)
        
    elif not run_btn:
        st.info("Configure parameters in the sidebar and click 'Run Analysis' to view results.")
