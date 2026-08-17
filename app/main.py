from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import AgglomerativeClustering, KMeans, HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR.parent / "data"))
CSV_PATH = DATA_DIR / "Grid5000.csv"


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
    if n > 1024**3:
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
            if isinstance(a, dict) and a.get("rate") is not None:
                try:
                    rates.append(float(a["rate"]))
                except Exception:
                    pass
    return rate_to_gbps(max(rates)) if rates else np.nan


def count_storage_devices(d):
    x = nested_get(d, "storage_devices", default=[])
    return len(x) if isinstance(x, list) else 0


def parse_node(site, cluster, d, fallback_uid):
    uid = d.get("uid") or d.get("name") or fallback_uid

    cpu_model = nested_get(
        d, "processor.model", "processor.model_name", default="Unknown"
    )
    cpu_micro = nested_get(
        d, "processor.microarchitecture", "processor.other_description", default="Unknown"
    )
    arch = nested_get(
        d, "architecture.platform_type", "architecture.platform", default="Unknown"
    )
    cores = nested_get(
        d, "architecture.nb_cores", "processor.nb_cores", default=np.nan
    )
    cpus = nested_get(
        d, "architecture.nb_procs", "processor.nb_procs", default=np.nan
    )
    ram = nested_get(
        d, "main_memory.ram_size", "main_memory.size", default=np.nan
    )

    return {
        "site": site,
        "cluster": cluster,
        "node": uid,
        "architecture": arch,
        "cpu_model": str(cpu_model),
        "cpu_microarchitecture": str(cpu_micro),
        "cpu_count": cpus,
        "cores": cores,
        "ram_gb": human_bytes_to_gb(ram),
        "gpu_model": first_gpu_model(d),
        "network_max_gbps": fastest_network_gbps(d),
        "storage_device_count": count_storage_devices(d),
    }


def read_node_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_csv_from_repository():
    sites_root = REPO_DIR / "data" / "grid5000" / "sites"
    if not sites_root.exists():
        raise RuntimeError(
            "The downloaded Grid'5000 repository does not contain "
            "data/grid5000/sites."
        )

    node_files = []
    for pattern in (
        "*/clusters/*/nodes/*.json",
        "*/clusters/*/nodes/*.yaml",
        "*/clusters/*/nodes/*.yml",
    ):
        node_files.extend(sites_root.glob(pattern))

    if not node_files:
        raise RuntimeError(
            "No node inventory files were found in the Grid'5000 repository."
        )

    rows = []
    for node_file in node_files:
        try:
            rel = node_file.relative_to(sites_root)
            site = rel.parts[0]
            cluster = rel.parts[2]
        except Exception:
            continue

        d = read_node_file(node_file)
        if not d:
            continue

        rows.append(parse_node(site, cluster, d, node_file.stem))

    if not rows:
        raise RuntimeError("Grid'5000 node files were found but none could be parsed.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["site", "cluster", "node"])
    for c in ["cpu_count", "cores", "ram_gb", "network_max_gbps", "storage_device_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["site", "cluster", "node"]).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)




def load_df():
    """
    Local-only mode.
    The application never downloads data or accesses the Internet.
    Put Grid5000.csv in /app/data (host: ./data/Grid5000.csv).
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Grid5000.csv was not found. Put the file here: {CSV_PATH}"
        )
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
    return x.groupby(cols, dropna=False).size().reset_index(name="node_count")


def encode_configs(cfg: pd.DataFrame, cols: list[str]):
    cat = [c for c in cols if c not in NUMERIC_COLUMNS]
    num = [c for c in cols if c in NUMERIC_COLUMNS]

    transformers = []
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
        )
    if num:
        transformers.append(("num", StandardScaler(), num))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    X = pre.fit_transform(cfg[cols])
    return np.asarray(X, dtype=float)



def score_clustering(labels, X, n_configs):
    labels = np.asarray(labels)
    non_noise = labels != -1
    valid_labels = labels[non_noise]
    unique = sorted(set(valid_labels.tolist()))
    outlier_rate = float((labels == -1).mean())

    if len(unique) < 2 or non_noise.sum() < 3:
        sil = -1.0
    else:
        sil = float(silhouette_score(X[non_noise], valid_labels))

    k = len(unique)
    compression = (n_configs - max(1, k)) / max(1, n_configs - 1)

    # Quality matters most; compression is useful but should not dominate.
    normalized_sil = (sil + 1.0) / 2.0
    score = 0.72 * normalized_sil + 0.23 * compression - 0.15 * outlier_rate

    return {
        "k": int(k),
        "silhouette": round(sil, 4),
        "compression": round(compression, 4),
        "outlier_rate": round(outlier_rate, 4),
        "score": round(float(score), 4),
    }


def evaluate_fixed_k_algorithm(X, n_configs, algorithm):
    if n_configs < 3:
        return []

    max_k = min(15, n_configs - 1)
    points = []

    for k in range(2, max_k + 1):
        if algorithm == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        elif algorithm == "kmeans":
            labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X)
        else:
            raise ValueError("Unsupported fixed-k algorithm")

        m = score_clustering(labels, X, n_configs)
        m["k"] = k
        points.append(m)

    return points


def evaluate_hdbscan(X, n_configs):
    if n_configs < 5:
        return []

    candidates = []
    sizes = sorted(set([
        2, 3, 4, 5, 6, 8, 10,
        max(2, int(round(n_configs * 0.05))),
        max(2, int(round(n_configs * 0.10))),
    ]))

    for mcs in sizes:
        if mcs >= n_configs:
            continue
        try:
            labels = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
            m = score_clustering(labels, X, n_configs)
            m["min_cluster_size"] = int(mcs)
            m["labels"] = labels.tolist()
            candidates.append(m)
        except Exception:
            pass

    return candidates


def fit_algorithm(X, n_configs, algorithm, k=None):
    if algorithm == "agglomerative":
        curve = evaluate_fixed_k_algorithm(X, n_configs, "agglomerative")
        best = max(curve, key=lambda x: x["score"]) if curve else {"k": 2}
        chosen_k = int(k or best["k"])
        chosen_k = max(2, min(chosen_k, n_configs - 1))
        labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(X)
        metrics = score_clustering(labels, X, n_configs)
        metrics["k"] = chosen_k
        return labels, metrics, curve

    if algorithm == "kmeans":
        curve = evaluate_fixed_k_algorithm(X, n_configs, "kmeans")
        best = max(curve, key=lambda x: x["score"]) if curve else {"k": 2}
        chosen_k = int(k or best["k"])
        chosen_k = max(2, min(chosen_k, n_configs - 1))
        labels = KMeans(n_clusters=chosen_k, n_init="auto", random_state=42).fit_predict(X)
        metrics = score_clustering(labels, X, n_configs)
        metrics["k"] = chosen_k
        return labels, metrics, curve

    if algorithm == "hdbscan":
        candidates = evaluate_hdbscan(X, n_configs)
        if not candidates:
            # fallback only for tiny/problematic selections
            labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
            metrics = score_clustering(labels, X, n_configs)
            metrics["fallback"] = True
            return labels, metrics, []

        best = max(candidates, key=lambda x: x["score"])
        labels = np.asarray(best.pop("labels"))
        metrics = dict(best)
        curve = [
            {k: v for k, v in c.items() if k != "labels"}
            for c in candidates
        ]
        return labels, metrics, curve

    raise ValueError("Unknown algorithm")


def consensus_summary(results):
    fixed_votes = [
        r["metrics"]["k"]
        for r in results
        if r["algorithm"] in ("agglomerative", "kmeans")
    ]
    hdb = next((r for r in results if r["algorithm"] == "hdbscan"), None)
    if hdb and hdb["metrics"]["k"] >= 2:
        fixed_votes.append(hdb["metrics"]["k"])

    if not fixed_votes:
        return {"votes": [], "consensus_k": None, "agreement": "No consensus"}

    counts = {}
    for v in fixed_votes:
        counts[v] = counts.get(v, 0) + 1

    # Exact majority if there is one. Otherwise use median as a stable consensus.
    best_k, best_count = max(counts.items(), key=lambda x: (x[1], -x[0]))
    if best_count >= 2:
        consensus_k = int(best_k)
        agreement = f"{best_count} of {len(fixed_votes)} algorithms selected {best_k} groups"
    else:
        consensus_k = int(round(float(np.median(fixed_votes))))
        agreement = f"No exact majority; median recommendation is {consensus_k} groups"

    return {
        "votes": fixed_votes,
        "consensus_k": consensus_k,
        "agreement": agreement,
    }


def json_safe(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def explain_cluster(subset, cols):
    parts = []
    for c in cols:
        s = subset[c]
        if c in NUMERIC_COLUMNS:
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if len(vals):
                lo, hi = float(vals.min()), float(vals.max())
                med = float(vals.median())
                if abs(hi-lo) < 1e-9:
                    parts.append(f"{c} is consistently {round(med,2)}")
                else:
                    parts.append(f"{c} ranges {round(lo,2)}–{round(hi,2)} (median {round(med,2)})")
        else:
            vc = s.astype(str).value_counts()
            if len(vc):
                top = vc.index[0]
                share = vc.iloc[0] / max(1, vc.sum())
                if share >= 0.7:
                    parts.append(f"mostly {c}={top} ({round(share*100)}%)")
                else:
                    top2 = ", ".join(vc.head(3).index.tolist())
                    parts.append(f"{c} is mixed; common values: {top2}")
    return "; ".join(parts[:5]) + "."


def cluster_details_from_labels(cfg, X, cols, labels):
    labels = np.asarray(labels)
    temp = cfg.copy()
    temp["_cluster"] = labels

    groups = []

    # Noise/outliers first-class for HDBSCAN
    cluster_values = sorted([x for x in temp["_cluster"].unique() if x != -1])
    if -1 in temp["_cluster"].unique():
        cluster_values.append(-1)

    for label in cluster_values:
        idx = np.where(labels == label)[0]
        subset = temp[temp["_cluster"] == label].copy()

        if label == -1:
            canonical = None
            explanation = "Outliers/noise: these configurations were not similar enough to a stable HDBSCAN group."
        else:
            Xg = X[idx]
            if len(idx) == 1:
                medoid_global_idx = idx[0]
            else:
                D = pairwise_distances(Xg, metric="euclidean")
                medoid_local = int(np.argmin(D.sum(axis=1)))
                medoid_global_idx = idx[medoid_local]
            medoid = cfg.iloc[medoid_global_idx]
            canonical = {c: json_safe(medoid[c]) for c in cols}
            explanation = explain_cluster(subset, cols)

        members = []
        for _, r in subset.sort_values("node_count", ascending=False).iterrows():
            members.append({
                "values": {c: json_safe(r[c]) for c in cols},
                "node_count": int(r["node_count"]),
            })

        groups.append({
            "group": "Outliers" if label == -1 else int(label) + 1,
            "canonical": canonical,
            "configuration_count": int(len(subset)),
            "node_count": int(subset["node_count"].sum()),
            "explanation": explanation,
            "members": members[:50],
        })

    groups.sort(key=lambda g: g["node_count"], reverse=True)
    return groups


def projection_points(X, labels):
    if len(X) < 2:
        return []

    if X.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        coords = np.column_stack([X[:, 0], np.zeros(len(X))])

    # Normalize for frontend SVG coordinates
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where((maxs - mins) == 0, 1, (maxs - mins))
    norm = (coords - mins) / span

    return [
        {
            "x": round(float(norm[i,0]), 5),
            "y": round(float(norm[i,1]), 5),
            "cluster": int(labels[i]),
        }
        for i in range(len(labels))
    ]
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
            "csv": CSV_PATH.name,
            "source": "Local data/Grid5000.csv",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}



@app.get("/api/sample")
def sample():
    try:
        df = load_df()
        safe = df.head(20).where(pd.notnull(df.head(20)), None)
        return {"rows": safe.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/analyze")
def analyze(
    columns: str = Query("site,cluster,cpu_model"),
    algorithm: str = Query("auto"),
    k: int | None = Query(None),
):
    try:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        if len(cols) < 2:
            raise HTTPException(400, "Select at least 2 columns.")

        bad = [c for c in cols if c not in FEATURE_COLUMNS]
        if bad:
            raise HTTPException(400, f"Unsupported columns: {bad}")

        if algorithm not in {"auto", "agglomerative", "kmeans", "hdbscan"}:
            raise HTTPException(400, "Unsupported algorithm.")

        df = load_df()
        cfg = unique_configurations(df, cols)
        n = len(cfg)

        if n < 3:
            raise HTTPException(400, "Need at least 3 unique configurations for clustering.")

        X = encode_configs(cfg, cols)

        comparison = []
        for name in ["agglomerative", "kmeans", "hdbscan"]:
            labels_i, metrics_i, curve_i = fit_algorithm(X, n, name, None)
            comparison.append({
                "algorithm": name,
                "metrics": metrics_i,
                "curve": curve_i,
            })

        consensus = consensus_summary(comparison)

        if algorithm == "auto":
            # Auto chooses the best-scoring algorithm, but consensus k is still shown separately.
            chosen = max(comparison, key=lambda r: r["metrics"]["score"])
            chosen_algorithm = chosen["algorithm"]
            labels, metrics, curve = fit_algorithm(
                X, n, chosen_algorithm,
                k if chosen_algorithm in ("agglomerative", "kmeans") else None
            )
        else:
            chosen_algorithm = algorithm
            labels, metrics, curve = fit_algorithm(
                X, n, algorithm,
                k if algorithm in ("agglomerative", "kmeans") else None
            )

        groups = cluster_details_from_labels(cfg, X, cols, labels)
        projection = projection_points(X, labels)

        return {
            "columns": cols,
            "nodes": int(len(df)),
            "unique_configurations": int(n),
            "selected_algorithm": chosen_algorithm,
            "selected_k": int(metrics.get("k", 0)),
            "metrics": metrics,
            "curve": curve,
            "comparison": comparison,
            "consensus": consensus,
            "groups": groups,
            "projection": projection,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

