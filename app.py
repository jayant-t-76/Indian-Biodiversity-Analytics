import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io
import warnings

# Suppress minor warnings for clean console
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Indian Biodiversity Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium Nature Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px -4px rgba(6, 78, 59, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.4rem;
        opacity: 0.92;
        max-width: 900px;
        line-height: 1.45;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(6px);
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. DATA PROCESSING & ROBUSTNESS PIPELINE
# ---------------------------------------------------------
def detect_coordinate_columns(df):
    """
    Automatically detects latitude and longitude column names in a DataFrame.
    Checks common patterns: 'decimalLatitude', 'decimalLongitude', 'latitude',
    'longitude', 'lat', 'lon', 'lng', etc. (case-insensitive and trimmed).
    Returns (lat_col, lon_col) or (None, None).
    """
    if df is None or len(df.columns) == 0:
        return None, None

    lat_candidates = [
        "decimallatitude", "latitude", "decimal_latitude", "lat", 
        "coord_lat", "lat_deg", "y", "y_coord", "verbatimlatitude"
    ]
    lon_candidates = [
        "decimallongitude", "longitude", "decimal_longitude", "lon", 
        "lng", "long", "coord_lon", "lon_deg", "x", "x_coord", "verbatimlongitude"
    ]

    col_map = {str(c).strip().lower(): c for c in df.columns}

    detected_lat = None
    detected_lon = None

    # Priority 1: Exact matches
    for cand in lat_candidates:
        if cand in col_map:
            detected_lat = col_map[cand]
            break

    for cand in lon_candidates:
        if cand in col_map:
            detected_lon = col_map[cand]
            break

    # Priority 2: Fuzzy substring checks if not found
    if detected_lat is None:
        for low_c, orig_c in col_map.items():
            if "lat" in low_c and not any(neg in low_c for neg in ["relat", "plate", "slate", "latter", "collat"]):
                detected_lat = orig_c
                break

    if detected_lon is None:
        for low_c, orig_c in col_map.items():
            if any(term in low_c for term in ["lon", "lng"]) and not any(neg in low_c for neg in ["clone", "elong"]):
                detected_lon = orig_c
                break

    return detected_lat, detected_lon


@st.cache_data(show_spinner="🌿 Loading and processing biodiversity dataset...")
def load_and_preprocess_data():
    """
    Loads 'biodiversity.csv' from workspace root, automatically detects delimiters,
    identifies latitude and longitude columns, drops invalid coordinates,
    formats categorical fields, and extracts temporal/geographic attributes.
    """
    possible_files = ["biodiversity.csv", "India_biodiversity.csv", "data.csv"]
    file_path = None
    
    # Locate dataset file
    for fname in possible_files:
        if os.path.exists(fname):
            file_path = fname
            break
            
    if not file_path:
        csvs = [f for f in os.listdir(".") if f.endswith(".csv") or f.endswith(".tsv")]
        if csvs:
            file_path = csvs[0]
            
    if not file_path:
        return None, "No CSV or TSV file found in root project folder.", {}

    # Auto-detect delimiter
    sep = "\t"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        if "\t" in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        elif ";" in first_line:
            sep = ";"

    # Read dataset safely
    try:
        df = pd.read_csv(file_path, sep=sep, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(file_path, sep=sep, low_memory=False, encoding="latin1", on_bad_lines="skip")

    # Clean column headers
    df.columns = df.columns.str.strip()
    initial_row_count = len(df)

    # Automatically identify latitude and longitude columns
    lat_col, lon_col = detect_coordinate_columns(df)

    dropped_coords_count = 0
    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        # Drop rows with missing or out-of-bounds coordinate values
        valid_coords_mask = (
            df[lat_col].notna() & 
            df[lon_col].notna() & 
            df[lat_col].between(-90.0, 90.0) & 
            df[lon_col].between(-180.0, 180.0) & 
            ~((df[lat_col] == 0.0) & (df[lon_col] == 0.0))
        )
        dropped_coords_count = initial_row_count - int(valid_coords_mask.sum())
        df = df[valid_coords_mask].copy()

        # Standardize coordinate column names
        df["latitude"] = df[lat_col]
        df["longitude"] = df[lon_col]
        df["decimalLatitude"] = df[lat_col]
        df["decimalLongitude"] = df[lon_col]
        df["has_valid_coords"] = True
    else:
        df["has_valid_coords"] = False

    geo_meta = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "initial_rows": initial_row_count,
        "dropped_rows": dropped_coords_count,
        "valid_rows": len(df)
    }

    # Clean categorical taxonomic columns
    taxa_cols = ["kingdom", "phylum", "class", "order", "family", "genus", "species", "taxonRank"]
    for col in taxa_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unspecified").astype(str).str.strip()
            df[col] = df[col].replace("", "Unspecified")
        else:
            df[col] = "Unspecified"

    # Clean Scientific Names
    if "scientificName" in df.columns:
        df["scientificName"] = df["scientificName"].fillna("Unknown Taxon").astype(str).str.strip()
    elif "verbatimScientificName" in df.columns:
        df["scientificName"] = df["verbatimScientificName"].fillna("Unknown Taxon").astype(str).str.strip()
    else:
        df["scientificName"] = df["species"]

    # Parse Event Dates
    if "eventDate" in df.columns:
        df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")
        df["year_parsed"] = df["eventDate"].dt.year
        df["month_parsed"] = df["eventDate"].dt.month
        df["month_name"] = df["eventDate"].dt.strftime("%B")
    else:
        df["eventDate"] = pd.NaT
        df["year_parsed"] = np.nan
        df["month_parsed"] = np.nan
        df["month_name"] = "Unknown"

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["year_parsed"])
    else:
        df["year"] = df["year_parsed"]

    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(df["month_parsed"])
    else:
        df["month"] = df["month_parsed"]

    # Year validation (filter placeholder years)
    df["year_clean"] = df["year"].apply(lambda y: int(y) if pd.notnull(y) and 1950 <= y <= 2026 else np.nan)

    # Indian Meteorological Seasons
    month_to_season = {
        12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
        3: "Summer / Pre-Monsoon (Mar-May)", 4: "Summer / Pre-Monsoon (Mar-May)", 5: "Summer / Pre-Monsoon (Mar-May)",
        6: "Monsoon (Jun-Sep)", 7: "Monsoon (Jun-Sep)", 8: "Monsoon (Jun-Sep)", 9: "Monsoon (Jun-Sep)",
        10: "Post-Monsoon (Oct-Nov)", 11: "Post-Monsoon (Oct-Nov)"
    }
    df["season"] = df["month"].map(month_to_season).fillna("Unknown Season")

    # Clean Location & Observer attributes
    if "stateProvince" in df.columns:
        df["stateProvince"] = df["stateProvince"].fillna("Unspecified Location").astype(str).str.strip()
        df["stateProvince"] = df["stateProvince"].replace("", "Unspecified Location")
    else:
        df["stateProvince"] = "Unspecified Location"

    if "locality" in df.columns:
        df["locality"] = df["locality"].fillna("").astype(str).str.strip()
    else:
        df["locality"] = ""

    if "recordedBy" in df.columns:
        df["recordedBy"] = df["recordedBy"].fillna("Anonymous Naturalist").astype(str).str.strip()
        df["recordedBy"] = df["recordedBy"].replace("", "Anonymous Naturalist")
    else:
        df["recordedBy"] = "Unknown"

    return df, file_path, geo_meta


# Load data
df_raw, load_info, geo_meta = load_and_preprocess_data()

if df_raw is None:
    st.error(f"❌ Failed to load dataset: {load_info}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR FILTERS (Real-time dynamic filtering)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 **Filters & Controls**")
    st.caption("Refine biodiversity records across India in real time.")
    
    # 1. Text Search
    search_query = st.text_input(
        "🔍 **Search Taxon / Observer**",
        placeholder="e.g. Tiger, Junonia, Aves, Narzary...",
        help="Filters records by species, scientific name, family, genus, or observer."
    ).strip()

    st.markdown("---")

    # 2. Kingdom Filter
    all_kingdoms = sorted([k for k in df_raw["kingdom"].unique() if k != "Unspecified"])
    if "Unspecified" in df_raw["kingdom"].values:
        all_kingdoms.append("Unspecified")
        
    selected_kingdoms = st.multiselect(
        "👑 **Kingdom**",
        options=all_kingdoms,
        default=all_kingdoms,
        help="Select one or more biological kingdoms."
    )

    # 3. Dynamic Phylum Filter (Reactive to Kingdom)
    if selected_kingdoms:
        available_phyla = sorted(
            [p for p in df_raw[df_raw["kingdom"].isin(selected_kingdoms)]["phylum"].unique() if p != "Unspecified"]
        )
    else:
        available_phyla = sorted([p for p in df_raw["phylum"].unique() if p != "Unspecified"])

    if "Unspecified" in df_raw["phylum"].values:
        available_phyla.append("Unspecified")

    selected_phyla = st.multiselect(
        "🧬 **Category / Phylum**",
        options=available_phyla,
        default=available_phyla,
        help="Filter records by taxonomic phylum (e.g. Arthropoda, Chordata, Tracheophyta)."
    )

    # 4. Dynamic Taxon / Class Filter (Reactive to Kingdom & Phylum)
    class_pool = df_raw.copy()
    if selected_kingdoms:
        class_pool = class_pool[class_pool["kingdom"].isin(selected_kingdoms)]
    if selected_phyla:
        class_pool = class_pool[class_pool["phylum"].isin(selected_phyla)]

    available_classes = sorted(class_pool["class"].unique())
    top_classes = class_pool["class"].value_counts().head(10).index.tolist()

    class_filter_mode = st.radio(
        "🗂️ **Taxon / Class Mode**",
        ["All Classes", "Select Specific Classes"],
        horizontal=True
    )
    
    if class_filter_mode == "Select Specific Classes":
        selected_classes = st.multiselect(
            "Choose Classes:",
            options=available_classes,
            default=top_classes[:5] if top_classes else available_classes[:3]
        )
    else:
        selected_classes = available_classes

    st.markdown("---")

    # 5. Year Range Slider
    valid_years = df_raw["year_clean"].dropna().astype(int)
    if not valid_years.empty:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
        default_start = max(min_year, 2010)
        selected_year_range = st.slider(
            "📅 **Observation Year Range**",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, max_year),
            step=1
        )
    else:
        selected_year_range = (2000, 2024)

    # 6. Season Filter
    all_seasons = [
        "Monsoon (Jun-Sep)",
        "Summer / Pre-Monsoon (Mar-May)",
        "Winter (Dec-Feb)",
        "Post-Monsoon (Oct-Nov)"
    ]
    selected_seasons = st.multiselect(
        "⛅ **Season**",
        options=all_seasons,
        default=all_seasons
    )

    # Coordinate Detection Summary in Sidebar
    with st.expander("📍 **Auto-Detected Coordinates**", expanded=False):
        st.markdown(f"• **Latitude Column:** `{geo_meta.get('lat_col', 'N/A')}`")
        st.markdown(f"• **Longitude Column:** `{geo_meta.get('lon_col', 'N/A')}`")
        st.markdown(f"• **Raw Records:** `{geo_meta.get('initial_rows', 0):,}`")
        st.markdown(f"• **Missing GPS Dropped:** `{geo_meta.get('dropped_rows', 0):,}`")
        st.markdown(f"• **Valid Geo Points:** `{geo_meta.get('valid_rows', 0):,}`")

    # Reset Button
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------
# Apply Real-time Filters
# ---------------------------------------------------------
filtered_df = df_raw.copy()

if selected_kingdoms:
    filtered_df = filtered_df[filtered_df["kingdom"].isin(selected_kingdoms)]

if selected_phyla:
    filtered_df = filtered_df[filtered_df["phylum"].isin(selected_phyla)]

if selected_classes:
    filtered_df = filtered_df[filtered_df["class"].isin(selected_classes)]

if not valid_years.empty:
    filtered_df = filtered_df[
        (filtered_df["year_clean"].isna()) | 
        ((filtered_df["year_clean"] >= selected_year_range[0]) & (filtered_df["year_clean"] <= selected_year_range[1]))
    ]

if selected_seasons:
    filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]

if search_query:
    q = search_query.lower()
    mask = (
        filtered_df["species"].str.lower().str.contains(q, na=False) |
        filtered_df["scientificName"].str.lower().str.contains(q, na=False) |
        filtered_df["phylum"].str.lower().str.contains(q, na=False) |
        filtered_df["family"].str.lower().str.contains(q, na=False) |
        filtered_df["genus"].str.lower().str.contains(q, na=False) |
        filtered_df["recordedBy"].str.lower().str.contains(q, na=False)
    )
    filtered_df = filtered_df[mask]

total_raw_count = len(df_raw)
filtered_count = len(filtered_df)
pct_retained = (filtered_count / total_raw_count * 100) if total_raw_count > 0 else 0


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">
        <span>🌿</span> Indian Biodiversity Analytics Dashboard
    </div>
    <div class="hero-subtitle">
        High-performance spatial exploration of citizen science biodiversity observations across India.
        Investigate species taxonomy, OpenStreetMap distributions, seasonal trends, and occurrence records.
    </div>
    <div class="hero-badges">
        <div class="hero-badge">📁 File: {load_info}</div>
        <div class="hero-badge">📍 Coords: {geo_meta.get('lat_col', 'lat')} & {geo_meta.get('lon_col', 'lon')}</div>
        <div class="hero-badge">📊 {filtered_count:,} of {total_raw_count:,} Records Active ({pct_retained:.1f}%)</div>
        <div class="hero-badge">⏳ Year Range: {selected_year_range[0]} – {selected_year_range[1]}</div>
        <div class="hero-badge">🗺️ Free OpenStreetMap Tiles (No API Key Required)</div>
    </div>
</div>
""", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ **No records match your selected filter criteria.** Please adjust your sidebar filters.")
    st.stop()


# ---------------------------------------------------------
# 3. TOP METRICS (st.metric)
# ---------------------------------------------------------
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

unique_species_count = filtered_df[filtered_df["species"] != "Unspecified"]["species"].nunique()
unique_genera_count = filtered_df[filtered_df["genus"] != "Unspecified"]["genus"].nunique()

top_kingdom = filtered_df[filtered_df["kingdom"] != "Unspecified"]["kingdom"].value_counts()
top_k_name = top_kingdom.index[0] if not top_kingdom.empty else "N/A"
top_k_count = top_kingdom.iloc[0] if not top_kingdom.empty else 0

top_family = filtered_df[filtered_df["family"] != "Unspecified"]["family"].value_counts()
top_fam_name = top_family.index[0] if not top_family.empty else "N/A"
top_fam_count = top_family.iloc[0] if not top_family.empty else 0

valid_geo_count = filtered_df["has_valid_coords"].sum()
geo_pct = (valid_geo_count / filtered_count * 100) if filtered_count > 0 else 0

with m_col1:
    st.metric(
        label="Total Active Records",
        value=f"{filtered_count:,}",
        delta=f"{pct_retained:.1f}% of total"
    )

with m_col2:
    st.metric(
        label="Unique Species",
        value=f"{unique_species_count:,}",
        delta=f"{unique_genera_count:,} Genera"
    )

with m_col3:
    st.metric(
        label="Dominant Kingdom",
        value=top_k_name,
        delta=f"{top_k_count:,} records"
    )

with m_col4:
    st.metric(
        label="Top Family",
        value=top_fam_name[:14],
        delta=f"{top_fam_count:,} records"
    )

with m_col5:
    st.metric(
        label="Geo-referenced",
        value=f"{geo_pct:.1f}%",
        delta=f"{valid_geo_count:,} points"
    )

st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. FREE & WORKING GEOSPATIAL MAP (OpenStreetMap / scatter_map)
# ---------------------------------------------------------
st.markdown('<div class="section-title">📍 Interactive Geospatial Occurrence Map (OpenStreetMap)</div>', unsafe_allow_html=True)

lat_col = geo_meta.get("lat_col", "decimalLatitude")
lon_col = geo_meta.get("lon_col", "decimalLongitude")

# Prepare spatial subset with clean coordinates
geo_subset = filtered_df.dropna(subset=[lat_col, lon_col]).copy()
geo_subset[lat_col] = pd.to_numeric(geo_subset[lat_col], errors="coerce")
geo_subset[lon_col] = pd.to_numeric(geo_subset[lon_col], errors="coerce")
geo_subset = geo_subset.dropna(subset=[lat_col, lon_col])
geo_subset = geo_subset[
    (geo_subset[lat_col].between(-90.0, 90.0)) & 
    (geo_subset[lon_col].between(-180.0, 180.0)) & 
    ~((geo_subset[lat_col] == 0.0) & (geo_subset[lon_col] == 0.0))
]

if geo_subset.empty:
    st.info("No spatial coordinates available for the current filter selection.")
else:
    # Map controls
    map_ctrl1, map_ctrl2, map_ctrl3, map_ctrl4 = st.columns([2, 2, 2, 1])

    with map_ctrl1:
        map_engine = st.selectbox(
            "🗺️ **Tile Engine (100% Free, No API Key):**",
            ["OpenStreetMap (px.scatter_map)", "Carto Positron (px.scatter_map)", "Natural Earth Projection (px.scatter_geo)"],
            index=0
        )

    with map_ctrl2:
        color_by_options = [c for c in ["phylum", "kingdom", "class", "family", "season"] if c in geo_subset.columns]
        map_color_var = st.selectbox(
            "🎨 **Color Markers By:**",
            options=color_by_options,
            index=0
        )

    with map_ctrl3:
        # Intelligent sampling: maximum 3,000 to 5,000 points dynamically for smooth 60fps interaction
        max_possible_pts = min(len(geo_subset), 5000)
        default_sample = min(max_possible_pts, 3500)
        sample_pts = st.slider(
            "⚡ **Display Sample Limit (pts):**",
            min_value=min(500, max_possible_pts),
            max_value=max(max_possible_pts, 500),
            value=default_sample,
            step=500,
            help="Subsamples data dynamically to 3,000–5,000 points to ensure buttery-smooth panning and zooming without freezing the browser."
        )

    with map_ctrl4:
        marker_size = st.slider("Size:", min_value=2, max_value=12, value=5, step=1)

    # Apply intelligent dynamic sampling
    if len(geo_subset) > sample_pts:
        display_map_df = geo_subset.sample(n=sample_pts, random_state=42).copy()
        st.caption(f"ℹ️ Rendering an optimized representative sample of **{sample_pts:,}** of **{len(geo_subset):,}** geo-referenced points for lag-free exploration.")
    else:
        display_map_df = geo_subset.copy()

    # Prepare rich hover tooltip data
    display_map_df["tooltip_species"] = (
        display_map_df["species"].replace("Unspecified", "")
        .combine_first(display_map_df["scientificName"])
        .fillna("Unknown Taxon")
    )
    display_map_df["tooltip_phylum"] = display_map_df["phylum"].fillna("Unspecified")
    display_map_df["tooltip_kingdom"] = display_map_df["kingdom"].fillna("Unspecified")
    display_map_df["tooltip_class"] = display_map_df["class"].fillna("Unspecified")
    display_map_df["tooltip_observer"] = display_map_df["recordedBy"].fillna("Anonymous")
    display_map_df["tooltip_date"] = display_map_df["eventDate"].astype(str).str[:10].replace("NaT", "Undated")
    display_map_df["tooltip_state"] = display_map_df["stateProvince"].fillna("India")

    # Center coordinates
    center_lat = float(display_map_df[lat_col].median()) if not display_map_df.empty else 22.5
    center_lon = float(display_map_df[lon_col].median()) if not display_map_df.empty else 82.0

    # Custom Hovertemplate (Species, Kingdom/Phylum, Observer, Location Coordinates)
    hovertemplate = (
        "<b>%{hovertext}</b><br><br>"
        "🌿 <b>Phylum:</b> %{customdata[0]}<br>"
        "👑 <b>Kingdom:</b> %{customdata[1]} | <b>Class:</b> %{customdata[2]}<br>"
        "👤 <b>Observer:</b> %{customdata[3]}<br>"
        "📅 <b>Date:</b> %{customdata[4]}<br>"
        "📍 <b>State:</b> %{customdata[5]}<br>"
        "🌐 <b>Coordinates:</b> %{lat:.4f}°N, %{lon:.4f}°E<extra></extra>"
    )

    custom_cols = [
        "tooltip_phylum", "tooltip_kingdom", "tooltip_class",
        "tooltip_observer", "tooltip_date", "tooltip_state"
    ]

    # Render Selected Map
    if "scatter_geo" in map_engine:
        fig_map = px.scatter_geo(
            display_map_df,
            lat=lat_col,
            lon=lon_col,
            color=map_color_var,
            hover_name="tooltip_species",
            custom_data=custom_cols,
            scope="asia",
            projection="natural earth",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title=f"Occurrences across Asia/India (Colored by {map_color_var.title()})"
        )
        fig_map.update_traces(
            marker=dict(size=marker_size, opacity=0.8),
            hovertemplate=hovertemplate
        )
        fig_map.update_geos(
            center=dict(lat=center_lat, lon=center_lon),
            showcountries=True,
            countrycolor="#94a3b8",
            showcoastlines=True,
            showland=True,
            landcolor="#f8fafc",
            showocean=True,
            oceancolor="#e0f2fe"
        )
    else:
        # Determine open map style
        chosen_style = "open-street-map" if "OpenStreetMap" in map_engine else "carto-positron"
        
        # Use scatter_map if available (modern Plotly 5.24+ / 6.0+)
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(
                display_map_df,
                lat=lat_col,
                lon=lon_col,
                color=map_color_var,
                hover_name="tooltip_species",
                custom_data=custom_cols,
                zoom=4.2,
                center=dict(lat=center_lat, lon=center_lon),
                map_style=chosen_style,
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Occurrences across India — {chosen_style.replace('-', ' ').title()} (Colored by {map_color_var.title()})"
            )
        else:
            fig_map = px.scatter_mapbox(
                display_map_df,
                lat=lat_col,
                lon=lon_col,
                color=map_color_var,
                hover_name="tooltip_species",
                custom_data=custom_cols,
                zoom=4.2,
                center=dict(lat=center_lat, lon=center_lon),
                mapbox_style=chosen_style,
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Occurrences across India (Colored by {map_color_var.title()})"
            )

        fig_map.update_traces(
            marker=dict(size=marker_size, opacity=0.8),
            hovertemplate=hovertemplate
        )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=35, b=0),
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. INTERACTIVE CHARTS (Plotly st.plotly_chart)
# ---------------------------------------------------------
tab_charts_spatial, tab_charts_taxa, tab_charts_temporal = st.tabs([
    "📍 Spatial Lat/Lon Gradients",
    "🧬 Taxonomy & Species Richness",
    "📈 Temporal & Seasonal Trends"
])

# Sub-Tab 1: Latitudinal & Longitudinal Histograms
with tab_charts_spatial:
    st.markdown('<div class="section-title">🧭 Latitudinal & Longitudinal Biodiversity Gradients</div>', unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        fig_lat = px.histogram(
            geo_subset,
            x=lat_col,
            nbins=40,
            color="kingdom",
            title=f"Latitudinal Distribution ({lat_col}: South to North India)",
            labels={lat_col: f"Latitude ({lat_col} °N)", "count": "Observations"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_lat.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_lat, use_container_width=True)

    with g_col2:
        fig_lon = px.histogram(
            geo_subset,
            x=lon_col,
            nbins=40,
            color="kingdom",
            title=f"Longitudinal Distribution ({lon_col}: West to East India)",
            labels={lon_col: f"Longitude ({lon_col} °E)", "count": "Observations"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_lon.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_lon, use_container_width=True)

# Sub-Tab 2: Taxonomy Breakdown & Top Species
with tab_charts_taxa:
    st.markdown('<div class="section-title">🧬 Taxonomic Breakdown & Top Recorded Species</div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        top_s_df = (
            filtered_df[filtered_df["species"] != "Unspecified"]["species"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        top_s_df.columns = ["Species", "Observations"]
        
        fig_s = px.bar(
            top_s_df,
            x="Observations",
            y="Species",
            orientation="h",
            color="Observations",
            color_continuous_scale="Viridis",
            text="Observations",
            title="Top 15 Most Recorded Species"
        )
        fig_s.update_layout(
            height=460,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40, b=20, l=20, r=20),
            coloraxis_showscale=False
        )
        fig_s.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_s, use_container_width=True)

    with t_col2:
        phylum_df = filtered_df["phylum"].value_counts().head(8).reset_index()
        phylum_df.columns = ["Phylum", "Observations"]
        
        fig_phylum = px.pie(
            phylum_df,
            names="Phylum",
            values="Observations",
            hole=0.45,
            title="Observations by Category / Phylum",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_phylum.update_traces(textposition='inside', textinfo='percent+label')
        fig_phylum.update_layout(height=460, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_phylum, use_container_width=True)

# Sub-Tab 3: Temporal Trends
with tab_charts_temporal:
    st.markdown('<div class="section-title">📈 Temporal Observations Growth & Seasonality</div>', unsafe_allow_html=True)
    time_col1, time_col2 = st.columns([3, 2])

    with time_col1:
        year_df = (
            filtered_df["year_clean"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        year_df.columns = ["Year", "Observations"]
        
        if not year_df.empty:
            fig_year = px.area(
                year_df,
                x="Year",
                y="Observations",
                markers=True,
                title="Annual Biodiversity Observation Growth",
                color_discrete_sequence=["#059669"]
            )
            fig_year.update_traces(
                line=dict(width=3, color="#047857"),
                fillcolor="rgba(16, 185, 129, 0.2)"
            )
            fig_year.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("No temporal year data available.")

    with time_col2:
        season_order = [
            "Monsoon (Jun-Sep)",
            "Summer / Pre-Monsoon (Mar-May)",
            "Winter (Dec-Feb)",
            "Post-Monsoon (Oct-Nov)"
        ]
        season_df = (
            filtered_df["season"]
            .value_counts()
            .reindex(season_order)
            .dropna()
            .reset_index()
        )
        season_df.columns = ["Season", "Observations"]
        
        fig_season = px.bar(
            season_df,
            x="Season",
            y="Observations",
            color="Season",
            title="Seasonal Activity Peak",
            color_discrete_sequence=["#0284c7", "#f59e0b", "#6366f1", "#10b981"]
        )
        fig_season.update_layout(height=360, showlegend=False, margin=dict(t=40, b=20, l=20, r=20), xaxis_tickangle=-15)
        st.plotly_chart(fig_season, use_container_width=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. DATA EXPLORER & CSV EXPORT
# ---------------------------------------------------------
st.markdown('<div class="section-title">📋 Interactive Data Explorer & Export</div>', unsafe_allow_html=True)
st.caption("Inspect, filter, and export biodiversity records.")

preferred_cols = [
    "gbifID", "species", "scientificName", "kingdom", "phylum", "class", "order",
    "family", "genus", "eventDate", "year", "latitude", "longitude", "recordedBy", "stateProvince"
]
available_table_cols = [c for c in preferred_cols if c in filtered_df.columns]
extra_cols = [c for c in filtered_df.columns if c not in available_table_cols and not c.startswith("_")]

tbl_col1, tbl_col2 = st.columns([3, 1])
with tbl_col1:
    chosen_columns = st.multiselect(
        "Select Columns to Display:",
        options=available_table_cols + extra_cols,
        default=available_table_cols[:9]
    )
with tbl_col2:
    max_preview_rows = st.selectbox(
        "Preview Rows Limit:",
        options=[50, 100, 500, 1000, "All Records"],
        index=1
    )

# Slice records for viewing
if max_preview_rows == "All Records":
    view_df = filtered_df[chosen_columns] if chosen_columns else filtered_df
else:
    view_df = filtered_df[chosen_columns].head(int(max_preview_rows)) if chosen_columns else filtered_df.head(int(max_preview_rows))

st.dataframe(
    view_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

# Download filtered CSV Button
csv_col1, csv_col2 = st.columns([2, 3])
with csv_col1:
    export_slice = filtered_df[chosen_columns] if chosen_columns else filtered_df
    csv_bytes = export_slice.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Filtered CSV ({filtered_count:,} Rows)",
        data=csv_bytes,
        file_name="india_biodiversity_filtered.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )
with csv_col2:
    st.caption(f"Export will download all **{filtered_count:,}** filtered records with the **{len(chosen_columns)}** selected columns.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>🌿 <b>Indian Biodiversity Analytics</b> • Built with Python, Streamlit, and Plotly (OpenStreetMap).</div>", unsafe_allow_html=True)
