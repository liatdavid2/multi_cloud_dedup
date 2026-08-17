from __future__ import annotations

import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR.parent / "data"))
CSV_PATH = DATA_DIR / "Grid5000.csv"
REPO_DIR = DATA_DIR / "reference-repository-master"
REPO_ZIP = DATA_DIR / "reference-repository.zip"

# Public mirror of Grid'5000's official reference repository.
DOWNLOAD_URL = "https://github.com/grid5000/reference-repository/archive/refs/heads/master.zip"

app = FastAPI(title="Grid5000 Configuration Clustering")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def nested_get(d: dict, *paths, default=None):
    for path in paths:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def human_bytes_to_gb(v):
    try:
        n = float(v)
    except Exception:
        return np.nan
    # Grid'5000 sizes are commonly stored as bytes.
    if n > 1024**3:
        return round(n / (1024**3), 2)
    if n > 1024**2:
        return round(n / (1024**3), 2)
    return round(n, 2)


def rate_to_gbps(v):
    try:
        n = float(v)
    except Exception:
        return np.nan
    if n > 1_000_000:
        return round(n / 1_000_000_000, 2)
    return round(n, 2)


def first_gpu_model(d):
    gpus = nested_get(d, "gpu_devices", default=[])
    if isinstance(gpus, list) and gpus:
        g = gpus[0]
        if isinstance(g, dict):
            return g.get("model") or g.get("vendor") or "GPU"
    return "None"


def fastest_network_gbps(d):
    adapters = nested_get(d, "network_adapters", default=[])
    rates = []
    if isinstance(adapters, list):
        for a in adapters:
            if isinstance(a, dict):
                r = a.get("rate")
                if r is not None:
                    try:
                        rates.append(float(r))
                    except Exception:
                        pass
    return rate_to_gbps(max(rates)) if rates else np.nan


def count_storage_devices(d):
    x = nested_get(d, "storage_devices", default=[])
    return len(x) if isinstance(x, list) else 0


def download_reference_repository(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists() and not force:
        return

    if force:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        if REPO_ZIP.exists():
            REPO_ZIP.unlink()
        if CSV_PATH.exists():
            CSV_PATH.unlink()

    r = requests.get(DOWNLOAD_URL, timeout=120)
    r.raise_for_status()
    REPO_ZIP.write_bytes(r.content)

    with zipfile.ZipFile(REPO_ZIP, "r") as z:
        z.extractall(DATA_DIR)

    build_csv()


def build_csv():
    sites_root = REPO_DIR / "data" / "grid5000" / "sites"
    if not sites_root.exists():
        raise RuntimeError("Grid'5000 repository layout was not found after download.")

    rows = []
    # Expected official layout:
    # sites/<site>/clusters/<cluster>/nodes/<node>.json
    for node_file in sites_root.glob("*/clusters/*/nodes/*.json"):
        parts = node_file.parts
        try:
            site = parts[parts.index("sites") + 1]
            cluster = parts[parts.index("clusters") + 1]
        except Exception:
            continue

        try:
            d = json.loads(node_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        node = node_file.stem

        cpu_model = nested_get(
            d,
            "processor.model",
            "processor.model_name",
            default="Unknown"
        )
        cpu_micro = nested_get(
            d,
            "processor.microarchitecture",
            "processor.other_description",
            default="Unknown"
        )
        arch = nested_get(
            d,
            "architecture.platform_type",
            "architecture.platform",
            default="Unknown"
        )

        cores = nested_get(
            d,
            "architecture.nb_cores",
            "processor.nb_cores",
            default=np.nan
        )
        cpus = nested_get(
            d,
            "architecture.nb_procs",
            "processor.nb_procs",
            default=np.nan
        )
        ram = nested_get(
            d,
            "main_memory.ram_size",
            "main_memory.size",
            default=np.nan
        )

        rows.append({
            "site": site,
            "cluster": cluster,
            "node": node,
            "architecture": arch,
            "cpu_model": str(cpu_model),
            "cpu_microarchitecture": str(cpu_micro),
            "cpu_count": cpus,
            "cores": cores,
            "ram_gb": human_bytes_to_gb(ram),
            "gpu_model": first_gpu_model(d),
            "network_max_gbps": fastest_network_gbps(d),
            "storage_device_count": count_storage_devices(d),
        })

    if not rows:
        raise RuntimeError("No Grid'5000 node JSON files were parsed.")

    df = pd.DataFrame(rows)
    for c in ["cpu_count", "cores", "ram_gb", "network_max_gbps", "storage_device_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(CSV_PATH, index=False)


def load_df():
    if not CSV_PATH.exists():
        download_reference_repository(force=False)
    return pd.read_csv(CSV_PATH)


FEATURE_COLUMNS = [
    "site",
    "cluster",
    "architecture",
    "cpu_model",
    "cpu_microarchitecture",
    "cpu_count",
    "cores",
    "ram_gb",
    "gpu_model",
    "network_max_gbps",
    "storage_device_count",
]

NUMERIC_COLUMNS = {
    "cpu_count", "cores", "ram_gb", "network_max_gbps", "storage_device_count"
}


def unique_configurations(df: pd.DataFrame, cols: list[str]):
    x = df[cols].copy()
    for c in cols:
        if c in NUMERIC_COLUMNS:
            x[c] = pd.to_numeric(x[c], errors="coerce")
            med = x[c].median()
            if pd.isna(med):
                med = 0
            x[c] = x[c].fillna(med)
        else:
            x[c] = x[c].fillna("Unknown").astype(str)

    grouped = x.groupby(cols, dropna=False).size().reset_index(name="node_count")
    return grouped


def encode_configs(cfg: pd.DataFrame, cols: list[str]):
    cat = [c for c in cols if c not in NUMERIC_COLUMNS]
    num = [c for c in cols if c in NUMERIC_COLUMNS]

    transformers = []
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if num:
        transformers.append(("num", StandardScaler(), num))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X = pre.fit_transform(cfg[cols])
    return np.asarray(X, dtype=float)


def quality_curve(X, n_configs: int):
    if n_configs < 3:
        return []
    max_k = min(15, n_configs - 1)
    points = []
    for k in range(2, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(X, labels))
        # Normalize silhouette from [-1, 1] to [0, 1]
        sil01 = (sil + 1.0) / 2.0
        compression = (n_configs - k) / max(1, n_configs - 1)
        combined = 0.75 * sil01 + 0.25 * compression
        points.append({
            "k": k,
            "silhouette": round(sil, 4),
            "compression": round(compression, 4),
            "score": round(combined, 4),
        })
    return points


def cluster_details(cfg: pd.DataFrame, X: np.ndarray, cols: list[str], k: int):
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    temp = cfg.copy()
    temp["_cluster"] = labels

    groups = []
    for label in sorted(temp["_cluster"].unique()):
        idx = np.where(labels == label)[0]
        Xg = X[idx]
        if len(idx) == 1:
            medoid_global_idx = idx[0]
        else:
            D = pairwise_distances(Xg, metric="euclidean")
            medoid_local = int(np.argmin(D.sum(axis=1)))
            medoid_global_idx = idx[medoid_local]

        medoid = cfg.iloc[medoid_global_idx]
        subset = temp[temp["_cluster"] == label].copy()

        canonical = {c: (None if pd.isna(medoid[c]) else medoid[c].item() if hasattr(medoid[c], "item") else medoid[c]) for c in cols}

        members = []
        for _, r in subset.sort_values("node_count", ascending=False).iterrows():
            members.append({
                "values": {
                    c: (None if pd.isna(r[c]) else r[c].item() if hasattr(r[c], "item") else r[c])
                    for c in cols
                },
                "node_count": int(r["node_count"]),
            })

        groups.append({
            "group": int(label) + 1,
            "canonical": canonical,
            "configuration_count": int(len(subset)),
            "node_count": int(subset["node_count"].sum()),
            "members": members[:50],
        })

    groups.sort(key=lambda g: g["node_count"], reverse=True)
    return groups


@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    try:
        df = load_df()
        return {
            "ok": True,
            "rows": int(len(df)),
            "sites": int(df["site"].nunique()),
            "clusters": int(df["cluster"].nunique()),
            "columns": FEATURE_COLUMNS,
            "csv": str(CSV_PATH.name),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/refresh")
def refresh():
    try:
        download_reference_repository(force=True)
        df = load_df()
        return {"ok": True, "rows": int(len(df))}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/sample")
def sample():
    try:
        df = load_df()
        return {"rows": df.head(20).replace({np.nan: None}).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/analyze")
def analyze(
    columns: str = Query("site,cluster,cpu_model"),
    k: int | None = Query(None),
):
    try:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        if len(cols) != 3:
            raise HTTPException(400, "Select exactly 3 columns.")
        bad = [c for c in cols if c not in FEATURE_COLUMNS]
        if bad:
            raise HTTPException(400, f"Unsupported columns: {bad}")

        df = load_df()
        cfg = unique_configurations(df, cols)
        n = len(cfg)
        if n < 2:
            raise HTTPException(400, "The selected columns create fewer than 2 unique configurations.")

        X = encode_configs(cfg, cols)
        curve = quality_curve(X, n)
        if not curve:
            recommended_k = min(2, n)
        else:
            recommended_k = max(curve, key=lambda p: p["score"])["k"]

        selected_k = int(k or recommended_k)
        selected_k = max(2, min(selected_k, n - 1 if n > 2 else 2))
        groups = cluster_details(cfg, X, cols, selected_k)

        return {
            "columns": cols,
            "nodes": int(len(df)),
            "unique_configurations": int(n),
            "recommended_k": int(recommended_k),
            "selected_k": int(selected_k),
            "curve": curve,
            "groups": groups,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
