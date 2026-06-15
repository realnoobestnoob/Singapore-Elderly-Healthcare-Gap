"""
Data Pipeline & Feature Engineering for SgHealth-Optimize
"""
import pandas as pd
import numpy as np
import json
import os
import re
from utils.geo_lookup import get_sz_centroid

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── 1. Population Data ────────────────────────────────────────────────────────

def load_population() -> pd.DataFrame:
    """Load and concatenate all SingStat population CSVs (2011–2025)."""
    files = {
        "2011-2020": "respopagesex2011to2020.csv",
        "2021": "respopagesex2021.csv",
        "2022": "respopagesex2022.csv",
        "2023": "respopagesex2023.csv",
        "2024": "respopagesex2024.csv",
        "2025": "respopagesex2025.csv",
    }
    dfs = []
    for label, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        dfs.append(df)

    pop = pd.concat(dfs, ignore_index=True)

    # Standardise column names
    pop.columns = [c.strip() for c in pop.columns]

    # Clean Age column — extract numeric, treat "90_and_over" as 90
    def parse_age(val):
        val = str(val).strip()
        if val.lower() in ("total", "nan", ""):
            return np.nan
        m = re.search(r"\d+", val)
        return int(m.group()) if m else np.nan

    pop["AgeNum"] = pop["Age"].apply(parse_age)
    pop["Pop"] = pd.to_numeric(pop["Pop"], errors="coerce").fillna(0).astype(int)
    pop = pop.dropna(subset=["AgeNum"])
    pop["AgeNum"] = pop["AgeNum"].astype(int)

    # Remove aggregate rows
    pop = pop[pop["PA"].str.strip() != ""]
    pop = pop[pop["SZ"].str.strip() != ""]

    return pop


def build_features(pop: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subzone features for clustering:
      - aging_index_2025  : % population aged 65+ in 2025
      - senior_growth_rate: annualised CAGR of 65+ pop from 2015→2025
      - total_pop_2025    : total subzone population in 2025
    """
    # ── Aging Index 2025 ──────────────────────────────────────────────────────
    pop25 = pop[pop["Time"] == 2025]
    senior25 = (
        pop25[pop25["AgeNum"] >= 65]
        .groupby(["PA", "SZ"])["Pop"].sum()
        .reset_index(name="senior_pop_2025")
    )
    total25 = (
        pop25.groupby(["PA", "SZ"])["Pop"].sum()
        .reset_index(name="total_pop_2025")
    )
    feat = senior25.merge(total25, on=["PA", "SZ"])
    feat["aging_index_2025"] = feat["senior_pop_2025"] / feat["total_pop_2025"].replace(0, np.nan)

    # ── Senior Growth Rate 2015→2025 ─────────────────────────────────────────
    for yr, col in [(2015, "senior_pop_2015"), (2020, "senior_pop_2020")]:
        tmp = (
            pop[(pop["Time"] == yr) & (pop["AgeNum"] >= 65)]
            .groupby(["PA", "SZ"])["Pop"].sum()
            .reset_index(name=col)
        )
        feat = feat.merge(tmp, on=["PA", "SZ"], how="left")

    feat["senior_pop_2015"] = feat["senior_pop_2015"].fillna(0)
    feat["senior_pop_2020"] = feat["senior_pop_2020"].fillna(0)

    # CAGR over 10 years (2015–2025)
    def cagr(end, start, years=10):
        if start <= 0:
            return 0.0
        return (end / start) ** (1 / years) - 1

    feat["senior_growth_rate"] = feat.apply(
        lambda r: cagr(r["senior_pop_2025"], r["senior_pop_2015"]), axis=1
    )
    feat["senior_growth_rate"] = feat["senior_growth_rate"].clip(-0.3, 0.5)

    return feat


# ── 2. Infrastructure Data ────────────────────────────────────────────────────

def _load_geojson(fname) -> pd.DataFrame:
    """Parse a GeoJSON file and return a DataFrame with name, lat, lon."""
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r") as f:
        gj = json.load(f)

    rows = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        lon, lat = coords[0], coords[1]

        # Extract name from Description HTML if not directly in props
        name = props.get("NAME") or props.get("name", "")
        if not name:
            desc = props.get("Description", "")
            m = re.search(r"<th>NAME</th>\s*<td>(.*?)</td>", desc)
            name = m.group(1) if m else "Unknown"

        rows.append({"name": name.strip(), "lat": lat, "lon": lon})

    return pd.DataFrame(rows)


def load_eldercare() -> pd.DataFrame:
    return _load_geojson("EldercareServices.geojson")


def load_dementia_gtp() -> pd.DataFrame:
    return _load_geojson("DementiaFriendlyGoToPointsGTPs.geojson")


def assign_infrastructure_to_subzones(
    feat: pd.DataFrame,
    eldercare: pd.DataFrame,
    dementia: pd.DataFrame,
    poly: pd.DataFrame = None,
    hospitals: pd.DataFrame = None,
) -> pd.DataFrame:
    """Nearest-centroid spatial join for infrastructure assignment.

    Weights per facility type (applied to infra_density):
      eldercare / dementia GTP : 1×
      polyclinic               : 3×
      hospital                 : 5×

    Also computes Euclidean-degree distances to nearest hospital and
    polyclinic, converted to km (1° ≈ 111 km at Singapore's latitude).
    Higher distance → higher risk.
    """
    from utils.geo_lookup import get_sz_centroid

    sz_df = feat[["PA", "SZ"]].copy().reset_index(drop=True)
    centroids = sz_df.apply(
        lambda r: pd.Series(get_sz_centroid(r["PA"], r["SZ"]), index=["sz_lat", "sz_lon"]),
        axis=1,
    )
    sz_lat = centroids["sz_lat"].values   # shape (n_subzones,)
    sz_lon = centroids["sz_lon"].values

    # ── Weighted infra count ──────────────────────────────────────────────
    WEIGHTS = {"eldercare": 1, "dementia": 1, "polyclinic": 3, "hospital": 5}

    layers = [
        eldercare.assign(infra_type="eldercare"),
        dementia.assign(infra_type="dementia"),
    ]
    if poly is not None and len(poly):
        layers.append(poly.assign(infra_type="polyclinic"))
    if hospitals is not None and len(hospitals):
        layers.append(hospitals.assign(infra_type="hospital"))

    all_nodes = pd.concat(layers, ignore_index=True).dropna(subset=["lat", "lon"])

    node_lat = all_nodes["lat"].values[:, np.newaxis]   # (n_nodes, 1)
    node_lon = all_nodes["lon"].values[:, np.newaxis]
    dist_sq  = (node_lat - sz_lat) ** 2 + (node_lon - sz_lon) ** 2  # (n_nodes, n_sz)
    nearest  = dist_sq.argmin(axis=1)   # which subzone each node is nearest to

    all_nodes = all_nodes.copy()
    all_nodes["PA"] = sz_df["PA"].values[nearest]
    all_nodes["SZ"] = sz_df["SZ"].values[nearest]
    all_nodes["weight"] = all_nodes["infra_type"].map(WEIGHTS).fillna(1)

    weighted_counts = (
        all_nodes.groupby(["PA", "SZ"])["weight"]
        .sum()
        .reset_index(name="infra_count_weighted")
    )
    feat_out = feat.merge(weighted_counts, on=["PA", "SZ"], how="left")
    feat_out["infra_count_weighted"] = feat_out["infra_count_weighted"].fillna(0)
    feat_out["infra_density"] = (
        feat_out["infra_count_weighted"] / (feat_out["total_pop_2025"] / 1_000 + 0.5)
    )

    # ── Distance to nearest hospital (km) ─────────────────────────────────
    KM_PER_DEG = 111.0
    if hospitals is not None and len(hospitals):
        hosp = hospitals.dropna(subset=["lat", "lon"])
        h_lat = hosp["lat"].values[:, np.newaxis]
        h_lon = hosp["lon"].values[:, np.newaxis]
        d_deg = np.sqrt((h_lat - sz_lat) ** 2 + (h_lon - sz_lon) ** 2)
        feat_out["dist_nearest_hospital_km"] = d_deg.min(axis=0) * KM_PER_DEG
    else:
        feat_out["dist_nearest_hospital_km"] = 0.0

    # ── Distance to nearest polyclinic (km) ───────────────────────────────
    if poly is not None and len(poly):
        pc = poly.dropna(subset=["lat", "lon"])
        p_lat = pc["lat"].values[:, np.newaxis]
        p_lon = pc["lon"].values[:, np.newaxis]
        d_deg = np.sqrt((p_lat - sz_lat) ** 2 + (p_lon - sz_lon) ** 2)
        feat_out["dist_nearest_poly_km"] = d_deg.min(axis=0) * KM_PER_DEG
    else:
        feat_out["dist_nearest_poly_km"] = 0.0

    return feat_out


# ── 3. Hospital Admissions ────────────────────────────────────────────────────

def load_clinics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load polyclinics and public hospitals from singapore_clinic_pois.csv.
    Returns (polyclinics_df, hospitals_df) — each with name, lat, lon.
    """
    path = os.path.join(DATA_DIR, "singapore_clinic_pois.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["lat", "lon"])

    # Polyclinics: names containing "Polyclinic"
    poly = df[df["name"].str.contains("Polyclinic", case=False, na=False)][
        ["name", "lat", "lon"]
    ].reset_index(drop=True)

    # Public hospitals: well-known names only (avoid TCM / private medical centres)
    public_hospital_keywords = [
        "Singapore General Hospital", "Tan Tock Seng", "National University Hospital",
        "Changi General", "Khoo Teck Puat", "Sengkang General", "Ng Teng Fong",
        "Alexandra Hospital", "KK Women", "Kandang Kerbau", "National Heart Centre",
        "Woodlands Health", "Institute of Mental Health", "Woodbridge Hospital",
    ]
    pattern = "|".join(public_hospital_keywords)
    hospitals = df[df["name"].str.contains(pattern, case=False, na=False)][
        ["name", "lat", "lon"]
    ].reset_index(drop=True)

    # If public hospitals aren't in the CSV, hardcode the major ones
    if len(hospitals) < 5:
        hospitals = pd.DataFrame([
            {"name": "Singapore General Hospital",       "lat": 1.2795, "lon": 103.8352},
            {"name": "Tan Tock Seng Hospital",           "lat": 1.3214, "lon": 103.8462},
            {"name": "National University Hospital",     "lat": 1.2942, "lon": 103.7834},
            {"name": "Changi General Hospital",          "lat": 1.3404, "lon": 103.9497},
            {"name": "Khoo Teck Puat Hospital",          "lat": 1.4244, "lon": 103.8385},
            {"name": "Sengkang General Hospital",        "lat": 1.3954, "lon": 103.8936},
            {"name": "Ng Teng Fong General Hospital",    "lat": 1.3330, "lon": 103.7472},
            {"name": "Alexandra Hospital",               "lat": 1.2894, "lon": 103.8009},
            {"name": "KK Women's & Children's Hospital", "lat": 1.3092, "lon": 103.8453},
            {"name": "Woodlands Health Campus",          "lat": 1.4401, "lon": 103.7863},
        ])

    return poly, hospitals


def load_hospital_admissions() -> pd.DataFrame:
    """
    Load the wide-format hospital admissions CSV and melt to long format.
    Returns columns: DataSeries, date (period), admissions (int)
    """
    path = os.path.join(DATA_DIR, "AdmissionsToPublicSectorHospitalsMonthly.csv")
    df = pd.read_csv(path)

    # Melt: columns like "2024Jan", "2023Dec" → long format
    id_vars = ["DataSeries"]
    value_vars = [c for c in df.columns if c != "DataSeries"]

    long = df.melt(id_vars=id_vars, value_vars=value_vars,
                   var_name="period", value_name="admissions")

    # Parse period "2024Jan" → datetime
    long["date"] = pd.to_datetime(long["period"], format="%Y%b", errors="coerce")
    long = long.dropna(subset=["date"])
    long["admissions"] = pd.to_numeric(long["admissions"], errors="coerce")
    long = long.sort_values("date")

    return long
