import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------------- Streamlit config ------------------
st.set_page_config(
    page_title="Karnataka Segmentation Analysis",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  body, .stApp { background: linear-gradient(135deg,#0f172a,#1e293b,#020617); color: #e2e8f0 !important; }
  section[data-testid="stSidebar"]{ background: rgba(15,23,42,0.7); backdrop-filter: blur(12px); border-right:1px solid #334155; }
  .stMetric{ background: linear-gradient(135deg,#1e3a8a,#2563eb); padding:12px; border-radius:12px; color:white !important }
  h1,h2,h3,h4{ color:#bfdbfe !important }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def human_int(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return v

# ---------------- Load data ----------------
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "karnataka_districts.csv")
    if not os.path.exists(path):
        st.error("⚠️ CSV file 'karnataka_districts.csv' not found next to app.py")
        st.stop()
    df = pd.read_csv(path)
    df.rename(columns={
        "Literacy Rate": "Literacy_Rate",
        "Per Capita Income": "Per_Capita_Income",
        "Urbanization (%)": "Urbanization_Percent"
    }, inplace=True)
    return df

try:
    df = load_data().copy()
except Exception:
    st.stop()

required = {"District","Latitude","Longitude","Population","Literacy_Rate","Per_Capita_Income","Urbanization_Percent"}
if not required.issubset(set(df.columns)):
    st.error(f"CSV missing columns. Required: {sorted(required)}")
    st.stop()

# ---------------- Feature engineering ----------------
df["Population_Density"] = df["Population"] / 1000.0
bengaluru = (12.9716, 77.5946)

def safe_geodesic(lat, lon, ref=bengaluru):
    try:
        return geodesic((lat, lon), ref).km
    except Exception:
        return np.nan

df["Distance_to_Bengaluru_km"] = df.apply(lambda r: round(safe_geodesic(r["Latitude"], r["Longitude"]), 2), axis=1)

coastal = {"Dakshina Kannada", "Udupi", "Uttara Kannada"}
forest = {"Shivamogga", "Kodagu", "Chikkamagaluru"}

df["District_clean"] = df["District"].astype(str).str.replace("_", " ").str.strip()
df["Is_Coastal"] = df["District_clean"].isin(coastal).astype(int)
df["Is_Forest_Zone"] = df["District_clean"].isin(forest).astype(int)
df["Heat_Risk_Index"] = (df["Latitude"] > 15).astype(int)

urb_max = df["Urbanization_Percent"].max() or 1
inc_max = df["Per_Capita_Income"].max() or 1

df["Infra_Index"] = 0.5 * (df["Urbanization_Percent"] / urb_max) + 0.5 * (df["Per_Capita_Income"] / inc_max)
df["Env_Vuln_Index"] = 0.6 * df["Heat_Risk_Index"] + 0.2 * df["Is_Coastal"] + 0.2 * df["Is_Forest_Zone"]

# ---------------- Sidebar controls ----------------
st.sidebar.title("🔧 Controls")
feature_pool = [
    "Population","Literacy_Rate","Per_Capita_Income",
    "Urbanization_Percent","Population_Density",
    "Distance_to_Bengaluru_km","Is_Coastal","Is_Forest_Zone",
    "Heat_Risk_Index","Infra_Index","Env_Vuln_Index"
]

preset = st.sidebar.selectbox("Feature Preset", ["Balanced","Socio-Economic","Geography + Access","Environment Focus"])
if preset == "Balanced":
    default_feats = ["Population","Literacy_Rate","Per_Capita_Income","Urbanization_Percent"]
elif preset == "Socio-Economic":
    default_feats = ["Population","Literacy_Rate","Per_Capita_Income","Infra_Index"]
elif preset == "Geography + Access":
    default_feats = ["Population_Density","Distance_to_Bengaluru_km","Is_Coastal","Is_Forest_Zone"]
else:
    default_feats = ["Env_Vuln_Index","Heat_Risk_Index","Is_Coastal","Is_Forest_Zone"]

features = st.sidebar.multiselect("Select Features for Clustering", feature_pool, default_feats)
if len(features) < 2:
    st.sidebar.error("Select at least two features for clustering.")
    st.stop()

show_coastal = st.sidebar.checkbox("Coastal Only", False)
show_forest = st.sidebar.checkbox("Forest Only", False)

filtered_df = df.copy()
if show_coastal:
    filtered_df = filtered_df[filtered_df["Is_Coastal"] == 1]
if show_forest:
    filtered_df = filtered_df[filtered_df["Is_Forest_Zone"] == 1]

if filtered_df.shape[0] == 0:
    st.error("No districts match filters.")
    st.stop()

max_k = max(1, min(10, len(filtered_df)))
if max_k <= 2:
    st.sidebar.info(f"Only {max_k} districts → Clusters fixed to {max_k}.")
    n_clusters = max_k
else:
    n_clusters = st.sidebar.slider("Clusters (K)", min_value=2, max_value=max_k, value=min(4, max_k))

# ---------------- KMeans + PCA ----------------
@st.cache_data(show_spinner=False)
def run_kmeans(df_in, feats, k):
    for f in feats:
        if f not in df_in.columns:
            raise ValueError(f"Feature '{f}' not found in dataframe")
    X = df_in[feats].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    pcs = PCA(n_components=2).fit_transform(Xs)
    sil = silhouette_score(Xs, labels) if len(df_in) > k else np.nan
    db = davies_bouldin_score(Xs, labels) if len(df_in) > k else np.nan
    out = df_in.copy()
    out["Cluster"] = labels
    out["PC1"], out["PC2"] = pcs[:,0], pcs[:,1]
    return out, sil, db, pcs.var(axis=0).sum()

clustered_df, sil, db, pca_var = run_kmeans(filtered_df, features, n_clusters)

# ---------- FIX: Merge ALL Bengaluru districts into ONE cluster ----------
bengaluru_mask = clustered_df["District"].str.contains("Bengaluru", case=False)

if bengaluru_mask.any():
    base_cluster = clustered_df.loc[bengaluru_mask, "Cluster"].iloc[0]
    clustered_df.loc[bengaluru_mask, "Cluster"] = base_cluster
    clustered_df.loc[bengaluru_mask, "District"] = "Bengaluru"

# ---------------- Tabs ----------------
tabs = st.tabs([
    "🏠 Overview", "🔍 Clustering", "📊 Analytics",
    "🎯 Profiles", "🧠 Action Plan", "📌 District Profile",
    "🔥 Movement Identity", "📥 Export"
])

# ---------------- TAB 0 — Overview ----------------
with tabs[0]:
    st.header("Karnataka Segmentation Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", human_int(len(clustered_df)))
    c2.metric("Population", human_int(int(clustered_df["Population"].sum())))
    c3.metric("Avg Literacy", f"{clustered_df['Literacy_Rate'].mean():.2f}%")
    c4.metric("Clusters", n_clusters)

    st.subheader("🗺 Cluster Map")
    fig_map = px.scatter_mapbox(
        clustered_df, lat="Latitude", lon="Longitude",
        color="Cluster", size="Population",
        hover_name="District", mapbox_style="open-street-map",
        zoom=6, height=520
    )
    fig_map.update_layout(margin=dict(t=10,l=0,r=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map, use_container_width=True)

st.subheader("🌳 Population Treemap (Cluster → District)")

clustered_df["Cluster_str"] = clustered_df["Cluster"].astype(str)

fig_tree = px.treemap(
    clustered_df,
    path=["Cluster_str", "District"],
    values="Population",
    color="Cluster_str",
    hover_data=features,
    color_discrete_sequence=[
        "#F8D3E0",  # Soft Pink
        "#D7EAF3",  # Soft Sky Blue
        "#E8F5C8",  # Soft Lime
        "#FCECC9",  # Soft Apricot
        "#DAD7FE",  # Soft Lavender
    ],
    title="Population Distribution by Cluster"
)

fig_tree.update_traces(
    textfont=dict(color="black", size=14),
    marker=dict(line=dict(width=2, color="black"))
)

fig_tree.update_layout(
    margin=dict(t=40, l=0, r=0, b=0),
    paper_bgcolor="rgba(15,23,42,0.9)",
    plot_bgcolor="rgba(15,23,42,0.9)"
)

st.plotly_chart(fig_tree, use_container_width=True)

# ---------------- TAB 1 — Clustering ----------------
with tabs[1]:
    st.header("Clustering Performance & PCA View")

    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette", f"{sil:.3f}" if not np.isnan(sil) else "N/A")
    m2.metric("Davies-Bouldin", f"{db:.3f}" if not np.isnan(db) else "N/A")
    m3.metric("PCA Var (2D)", f"{pca_var*100:.1f}%")

    st.subheader("PCA Projection (2D)")
    fig_pca = px.scatter(clustered_df, x="PC1", y="PC2", color="Cluster", hover_name="District", hover_data=features, height=520)
    fig_pca.update_layout(margin=dict(t=10,l=0,r=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pca, use_container_width=True)

# ---------------- TAB 2 — Analytics ----------------
with tabs[2]:
    st.header("Feature Correlations")
    corr = clustered_df[features].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", height=520)
    fig_corr.update_layout(margin=dict(t=10,l=0,r=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- TAB 3 — Profiles ----------------
with tabs[3]:
    st.header("Cluster Profiles")
    for cid in sorted(clustered_df["Cluster"].unique()):
        subset = clustered_df[clustered_df["Cluster"] == cid]
        with st.expander(f"Cluster {cid} — {len(subset)} Districts"):
            st.write("**Districts:**", ", ".join(subset["District"].values))
            st.table(subset[features].mean().round(2).to_frame().T)

# ---------------- TAB 4 — Action Plan (SAFE) ----------------
with tabs[4]:
    st.header("Action Plan")

    overall = clustered_df[features].mean()

    for cid in sorted(clustered_df["Cluster"].unique()):
        sub = clustered_df[clustered_df["Cluster"] == cid]
        avg = sub[features].mean()
        actions = []
        p = 0

        # Literacy
        if "Literacy_Rate" in avg.index:
            if avg["Literacy_Rate"] < overall["Literacy_Rate"]:
                actions.append("🔴 Literacy low → Improve learning & digital centers")
                p += 1
            else:
                actions.append("🟢 Literacy OK")
        else:
            actions.append("ℹ️ Literacy_Rate not included in selected features")

        # Income
        if "Per_Capita_Income" in avg.index:
            if avg["Per_Capita_Income"] < overall["Per_Capita_Income"]:
                actions.append("🔴 Low income → Support MSMEs & training")
                p += 1
            else:
                actions.append("🟢 Income OK")
        else:
            actions.append("ℹ️ Per_Capita_Income not included in selected features")

        # Urbanization
        if "Urbanization_Percent" in avg.index:
            if avg["Urbanization_Percent"] < 30:
                actions.append("🔴 Low urbanization → Improve infrastructure & connectivity")
                p += 1
            else:
                actions.append("🟢 Urbanization OK")
        else:
            actions.append("ℹ️ Urbanization_Percent not included in selected features")

        priority = "🚨 HIGH" if p >= 3 else "⚠️ MEDIUM" if p == 2 else "✅ LOW"

        with st.expander(f"Cluster {cid} — {priority} Priority"):
            for a in actions:
                st.write(a)

# ---------------- TAB 5 — District Drill-Down Profile (NEW) ----------------
with tabs[5]:
    st.header("📌 District Drill-Down Profile")

    # district selector
    selected_dist = st.selectbox("Select a District", sorted(clustered_df["District"].unique()))

    dist_row = clustered_df[clustered_df["District"] == selected_dist].iloc[0]

    st.subheader(f"📍 {selected_dist} — District Overview")

    a1, a2, a3 = st.columns(3)
    a1.metric("Population", human_int(dist_row["Population"]))
    a2.metric("Literacy Rate", f"{dist_row['Literacy_Rate']:.1f}%")
    # check urbanization safe access
    a3_metric = f"{dist_row['Urbanization_Percent']:.1f}%" if "Urbanization_Percent" in dist_row.index else "N/A"
    a3.metric("Urbanization", a3_metric)

    b1, b2, b3 = st.columns(3)
    b1.metric("Per Capita Income", human_int(dist_row["Per_Capita_Income"]))
    b2.metric("Infra Index", f"{dist_row['Infra_Index']:.2f}")
    b3.metric("Env Vulnerability", f"{dist_row['Env_Vuln_Index']:.2f}")

    # Map location
    st.subheader("🗺 District Location")
    loc_df = pd.DataFrame([dist_row])
    fig_loc = px.scatter_mapbox(loc_df, lat="Latitude", lon="Longitude", hover_name="District", zoom=7, size=[20], mapbox_style="open-street-map", height=300)
    fig_loc.update_layout(margin=dict(t=10,l=0,r=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_loc, use_container_width=True)

    # ---------------- SINGLE DISTRICT BAR CHART (Best for your case) ----------------
    st.subheader("📊 District Feature Profile (Selected District Only)")

    bar_features_raw = [
        "Literacy_Rate", "Per_Capita_Income", "Urbanization_Percent",
        "Population_Density", "Infra_Index", "Env_Vuln_Index"
    ]

    # Keep only available features
    bar_features = [f for f in bar_features_raw if f in dist_row.index]

    district_vals = [float(dist_row[f]) for f in bar_features]

    # Build DataFrame
    bar_df = pd.DataFrame({
        "Feature": bar_features,
        "Value": district_vals
    })

    # Normalize values 0–100 for good display (normalize by dataset max to keep cross-district comparability)
    # This normalizes each feature by the maximum value across the state for that feature.
    norm_vals = []
    for f, v in zip(bar_features, district_vals):
        max_val = max(clustered_df[f].max(), 1)
        norm_vals.append((v / max_val) * 100)
    bar_df["Normalized"] = norm_vals

    # Plot
    fig_bar = px.bar(
        bar_df,
        x="Feature",
        y="Normalized",
        text="Value",
        height=420,
        template="plotly_dark"
    )

    fig_bar.update_layout(
        xaxis_title="Indicators",
        yaxis_title="Normalized Score (0–100)",
        margin=dict(t=10, l=0, r=0, b=0)
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    

    # Summary insights
    st.subheader("📝 Summary Insights")
    st.write(f"- **{selected_dist}** belongs to **Cluster {int(dist_row['Cluster'])}**.")
    st.write(f"- Distance from Bengaluru: **{dist_row['Distance_to_Bengaluru_km']} km**.")
    st.write(f"- Environmental vulnerability index: **{dist_row['Env_Vuln_Index']:.2f}**.")
    st.write("- (Trends shown are simulated for demonstration. Replace with real time series if available.)")

# ---------------- TAB 6 — Movement Identity ----------------
with tabs[6]:
    st.header("🔥 Movement Heat-Identity Segmentation (Prototype)")

    np.random.seed(42)
    n_people = 100
    move_data = {
        "Person_ID": range(1, n_people+1),
        "Avg_Distance_per_Day_km": np.random.uniform(1, 50, n_people),
        "Places_Visited_per_Week": np.random.randint(1, 15, n_people),
        "Time_at_Main_Location_hr": np.random.uniform(2, 16, n_people),
        "Speed_Avg_kmph": np.random.uniform(2, 30, n_people),
        "Routine_Score": np.random.uniform(0, 1, n_people),
    }
    move_df = pd.DataFrame(move_data)
    move_features = list(move_data.keys())[1:]
    Xs = StandardScaler().fit_transform(move_df[move_features])
    move_df["Cluster"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(Xs)
    label_map = {
        0: "📍 Micro-Zone Anchored", 1: "✈ Explorers", 2: "🏋 Routine Movers",
        3: "☕ Remote Workers", 4: "🚙 Field Workers"
    }
    move_df["Segment"] = move_df["Cluster"].map(label_map)
    st.write("### Sample Movement Data")
    st.dataframe(move_df.head())
    st.subheader("Movement Segment Summary")
    st.dataframe(move_df.groupby("Segment")[move_features].mean().round(2))
    pcs = PCA(n_components=2).fit_transform(Xs)
    move_df["PC1"], move_df["PC2"] = pcs[:,0], pcs[:,1]
    fig_mov = px.scatter(move_df, x="PC1", y="PC2", color="Segment", hover_name="Person_ID", height=520)
    fig_mov.update_layout(margin=dict(t=10,l=0,r=0,b=0))
    st.plotly_chart(fig_mov, use_container_width=True)

# ---------------- PDF / Export utilities ----------------

def draw_border_header_footer(c: rl_canvas.Canvas, doc):
    """Draw thin border, header and footer, and watermark on each PDF page."""
    width, height = A4
    # thin border
    c.setLineWidth(2)
    c.setStrokeColor(colors.HexColor('#0b3d91'))
    c.rect(18, 18, width-36, height-36)
    # header
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width/2, height-30, "Department of Computer Science and Engineering (Data Science) - NHCE")
    # footer - page number
    page_num = c.getPageNumber()
    c.setFont("Helvetica", 9)
    c.drawRightString(width-36, 20, f"Page {page_num}")
    # watermark (light)
    c.saveState()
    c.setFont("Helvetica", 48)
    c.setFillColor(colors.Color(0.8, 0.8, 0.8, alpha=0.08))
    c.translate(width/2, height/2)
    c.rotate(45)
    c.drawCentredString(0, 0, "NHCE — Karnataka Segmentation")
    c.restoreState()


def fig_to_png_bytes(fig, width=900, height=450, scale=1):
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return io.BytesIO(img_bytes)
    except Exception as e:
        st.warning(f"Could not export figure to image: {e}")
        return None


def build_pdf_report(clustered_df, figs: dict, selected_district=None, logo_path=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    story = []

    # Title page styles
    story.append(Spacer(1, 70)) 
    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=1, spaceAfter=6)
    subtitle = ParagraphStyle('subtitle', parent=styles['Heading2'], alignment=1)
    center_norm = ParagraphStyle('center', parent=styles['Normal'], alignment=1)
    

    # --- Title page ---
    story.append(Paragraph("NEW HORIZON COLLEGE OF ENGINEERING", title_style))
    story.append(Paragraph("(Affiliated to VTU, Belagavi)", center_norm))
    story.append(Paragraph("Department of Computer Science and Engineering (Data Science)", subtitle))
    story.append(Spacer(1, 14))

    story.append(Paragraph("KARNATAKA SEGMENTATION ANALYSIS", ParagraphStyle('h2', parent=styles['Title'], alignment=1, fontSize=20)))
    story.append(Paragraph("A Data-Driven Clustering & District Profiling Project", subtitle))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Project Report", subtitle))
    story.append(Paragraph(f"Academic Year: 2025 – 2026", center_norm))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Submitted in partial fulfillment of the requirements for the course:", center_norm))
    story.append(Paragraph('"Data Science / Machine Learning Project"', center_norm))
    story.append(Spacer(1, 18))

    # Students & Guide (bold)
    story.append(Paragraph("<b>CREATED BY:</b>", center_norm))
    story.append(Paragraph("<b>Mohan S Banagar</b> — USN: 1NH24CD406", center_norm))
    story.append(Paragraph("<b>Prajwal Jadimath</b> — USN: 1NH24CD408", center_norm))
    story.append(Spacer(1, 18))
    story.append(Paragraph("<b>GUIDED BY:</b>", center_norm))
    story.append(Paragraph("<b>Mr. Chandan Raj</b>", center_norm))
    story.append(Spacer(1, 18))

    if logo_path and os.path.exists(logo_path):
        try:
            logo_buf = io.BytesIO(open(logo_path, 'rb').read())
            story.append(RLImage(logo_buf, width=160, height=60))
            story.append(Spacer(1, 20))
        except Exception:
            pass

    story.append(Paragraph("Department of CSE – Data Science", center_norm))
    story.append(Paragraph("New Horizon College of Engineering, Bengaluru – 560103", center_norm))
    story.append(PageBreak())

    # --- Overview metrics ---
    story.append(Paragraph("Overview", styles["Heading2"]))
    metrics = [
        ["Districts", f"{len(clustered_df)}"],
        ["Population (total)", f"{int(clustered_df['Population'].sum()):,}"],
        ["Avg Literacy (%)", f"{clustered_df['Literacy_Rate'].mean():.2f}"],
        ["Clusters", f"{clustered_df['Cluster'].nunique()}"]
    ]
    t = Table(metrics, colWidths=[180, 220])
    t.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    story.append(t)
    story.append(PageBreak())

    # --- Charts ---
    for name in ("map","treemap","pca"):
        fig = figs.get(name)
        if fig is None:
            continue
        img_buf = fig_to_png_bytes(fig, width=1000, height=500)
        if img_buf is None:
            continue
        story.append(Paragraph(name.upper(), styles["Heading2"]))
        story.append(Spacer(1,6))
        story.append(RLImage(img_buf, width=480, height=240))
        story.append(Spacer(1,12))
    story.append(PageBreak())

    # --- Cluster summary table ---
    story.append(Paragraph("Cluster Summary (Selected features mean)", styles["Heading2"]))
    features_for_table = [c for c in clustered_df.columns if c in features]
    header = ["Cluster"] + features_for_table
    rows = [header]
    cluster_means = clustered_df.groupby("Cluster")[features_for_table].mean().round(2)
    for cid, row in cluster_means.iterrows():
        rows.append([str(cid)] + [str(row[f]) for f in features_for_table])
    tbl = Table(rows, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("GRID",(0,0),(-1,-1),0.25,colors.grey)
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # --- District detail if requested ---
    if selected_district:
        story.append(Paragraph(f"District Profile — {selected_district}", styles["Heading2"]))
        dr = clustered_df[clustered_df["District"] == selected_district].iloc[0]
        key_rows = [
            ["District", selected_district],
            ["Cluster", int(dr["Cluster"])],
            ["Population", f"{int(dr['Population']):,}"],
            ["Literacy Rate (%)", f"{dr['Literacy_Rate']:.2f}"],
            ["Per Capita Income", f"{dr['Per_Capita_Income']:,}"],
            ["Urbanization (%)", f"{dr['Urbanization_Percent']:.2f}"],
            ["Distance to Bengaluru (km)", f"{dr['Distance_to_Bengaluru_km']}"]
        ]
        ktable = Table(key_rows, colWidths=[200, 240])
        ktable.setStyle(TableStyle([("BOX",(0,0),(-1,-1),0.25,colors.grey)]))
        story.append(ktable)
        story.append(Spacer(1,12))
        for n in ("radar","trend"):
            fig = figs.get(n)
            if fig:
                img_buf = fig_to_png_bytes(fig, width=800, height=400)
                if img_buf:
                    story.append(RLImage(img_buf, width=450, height=220))
                    story.append(Spacer(1,8))
        story.append(PageBreak())

    # --- Action Plan summary ---
    story.append(Paragraph("Action Plan Summary", styles["Heading2"]))
    for cid in sorted(clustered_df["Cluster"].unique()):
        sub = clustered_df[clustered_df["Cluster"] == cid]
        story.append(Paragraph(f"Cluster {cid} — {len(sub)} districts", styles["Heading3"]))
        story.append(Paragraph("- Review literacy & income; consider infrastructure investments where metrics are below average.", styles["Normal"]))
        story.append(Spacer(1,6))

    # --- Conclusion (half page) ---
    story.append(PageBreak())
    story.append(Paragraph("<para alignment='center'><b><font size=18>Conclusion</font></b></para>", styles["Title"]))
    story.append(Spacer(1, 12))
    conclusion_text = """

This project successfully carried out a comprehensive and data-driven segmentation analysis of all districts in Karnataka using modern clustering and analytical techniques. By applying K-Means clustering, PCA-based visualization, and a wide range of socio-economic, geographic, infrastructural, and environmental indicators, the study was able to group districts into meaningful clusters that accurately reflect the multi-dimensional diversity of the state.

The analysis highlights major patterns related to population distribution, literacy variations, income disparities, levels of urban development, geographical advantages, and proximity to Bengaluru—one of the state’s most influential economic hubs. Additional environmental metrics such as forest influence, coastal exposure, and heat-risk zones further support the identification of vulnerable districts requiring targeted interventions.

Through interactive cluster profiling, feature correlation analysis, and district-wise drill-down dashboards, the system enables deeper understanding of development gaps, priority needs, and growth opportunities across regions. The insights derived from this study can assist government departments, planners, and policy makers in designing evidence-based strategies related to infrastructure expansion, education initiatives, economic strengthening, environmental protection, and sustainable district-level development.

Overall, this segmentation framework demonstrates how data science can support real-world planning and improve governance outcomes. The combination of clustering, visualization, and district-specific profiling makes the approach scalable and adaptable for future datasets, ensuring that the model remains relevant for long-term state development planning.
"""
    story.append(Paragraph(conclusion_text, styles["Normal"]))

    # Build PDF with border/header/footer/watermark applied to all pages
    doc.build(story, onFirstPage=draw_border_header_footer, onLaterPages=draw_border_header_footer)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Export Tab Integration ----------------
with tabs[7]:
    st.header("📥 Export Clustered Dataset & PDF Report")

    # CSV download (unique key)
    st.download_button(
        label="Download CSV",
        data=clustered_df.to_csv(index=False).encode(),
        file_name="karnataka_clusters.csv",
        mime="text/csv",
        key="download_csv_main"
    )

    figs_for_pdf = {
        "map": locals().get('fig_map'),
        "treemap": locals().get('fig_tree'),
        "pca": locals().get('fig_pca'),
        "radar": locals().get('fig_radar'),
        "trend": locals().get('fig_line')
    }

    logo_upload = st.file_uploader("Upload college logo (PNG/JPG) — optional", type=["png","jpg","jpeg"])    
    logo_path = None
    if logo_upload:
        logo_bytes = logo_upload.read()
        logo_path = os.path.join("tmp_logo.png")
        with open(logo_path, 'wb') as f:
            f.write(logo_bytes)

    selected_for_pdf = st.selectbox("Add district detail to PDF (optional)", options=[""] + sorted(clustered_df["District"].unique()))
    if st.button("Generate PDF Report", key="gen_pdf"):
        try:
            pdf_bytes = build_pdf_report(clustered_df, figs_for_pdf, selected_district=(selected_for_pdf or None), logo_path=logo_path)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="karnataka_report.pdf",
                mime="application/pdf",
                key="download_pdf_report"
            )
            st.success("PDF ready — click Download PDF.")
        except Exception as e:
            st.error(f"Failed to build PDF: {e}")

st.markdown("---")
st.caption("💡 Built by Mohan and Prajwal— Karnataka Segmentation Analysis")


