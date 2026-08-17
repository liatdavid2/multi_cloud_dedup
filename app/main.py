from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import AgglomerativeClustering, KMeans, HDBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    pairwise_distances,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv('DATA_DIR', APP_DIR.parent / 'data'))
CSV_PATH = DATA_DIR / 'Grid5000.csv'

app = FastAPI(title='Grid5000 Configuration Clustering')
app.mount('/static', StaticFiles(directory=APP_DIR / 'static'), name='static')

FEATURE_COLUMNS = [
    'site','cluster','architecture','cpu_model','cpu_microarchitecture',
    'cpu_count','cores','ram_gb','gpu_model','network_max_gbps','storage_device_count',
]
NUMERIC_COLUMNS = {'cpu_count','cores','ram_gb','network_max_gbps','storage_device_count'}


def load_df():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'Grid5000.csv was not found. Put the file here: {CSV_PATH}')
    return pd.read_csv(CSV_PATH)


def unique_configurations(df: pd.DataFrame, cols: list[str]):
    """
    Build unique categorical configuration combinations.
    All selected columns are treated strictly as categorical values.
    """
    x = df[cols].copy()

    for c in cols:
        x[c] = x[c].fillna("Unknown").astype(str)

    return (
        x.groupby(cols, dropna=False)
         .size()
         .reset_index(name="node_count")
    )


def encode_configs(cfg: pd.DataFrame, cols: list[str]):
    """
    Categorical-only representation.
    Every selected dimension is one-hot encoded.
    No numeric scaling or numeric distance is used.
    """
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    X = encoder.fit_transform(cfg[cols].astype(str))
    return np.asarray(X, dtype=float)


def internal_metrics(labels, X, n):
    labels=np.asarray(labels)
    mask=labels!=-1
    used=labels[mask]
    k=len(set(used.tolist()))
    outlier=float((labels==-1).mean())
    sil=-1.0; db=None; ch=None
    if k>=2 and mask.sum()>k:
        Xm=X[mask]
        try: sil=float(silhouette_score(Xm,used))
        except Exception: pass
        try: db=float(davies_bouldin_score(Xm,used))
        except Exception: pass
        try: ch=float(calinski_harabasz_score(Xm,used))
        except Exception: pass
    compression=(n-max(1,k))/max(1,n-1)
    base=0.72*((sil+1)/2)+0.23*compression-0.15*outlier
    return {'k':int(k),'silhouette':round(sil,4),'davies_bouldin':None if db is None else round(db,4),
            'calinski_harabasz':None if ch is None else round(ch,2),'compression':round(compression,4),
            'outlier_rate':round(outlier,4),'score':round(float(base),4)}


def fixed_curve(X,n,alg):
    pts=[]
    for k in range(2,min(15,n-1)+1):
        if alg=='agglomerative': labels=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(X)
        else: labels=KMeans(n_clusters=k,n_init='auto',random_state=42).fit_predict(X)
        m=internal_metrics(labels,X,n); m['k']=k; pts.append(m)
    return pts


def hdb_candidates(X,n):
    vals=sorted(set([2,3,4,5,6,8,10,max(2,round(n*.05)),max(2,round(n*.10)),max(2,round(n*.20))]))
    out=[]
    for mcs in vals:
        if mcs>=n: continue
        try:
            labels=HDBSCAN(min_cluster_size=int(mcs)).fit_predict(X)
            m=internal_metrics(labels,X,n); m['min_cluster_size']=int(mcs); m['_labels']=labels
            out.append(m)
        except Exception: pass
    return out


def fit_algorithm(X,n,alg,k=None,params=None):
    if alg in ('agglomerative','kmeans'):
        curve=fixed_curve(X,n,alg)
        best=max(curve,key=lambda x:x['score']) if curve else {'k':2}
        kk=int(k or (params or {}).get('k') or best['k']); kk=max(2,min(kk,n-1))
        if alg=='agglomerative': labels=AgglomerativeClustering(n_clusters=kk,linkage='ward').fit_predict(X)
        else: labels=KMeans(n_clusters=kk,n_init='auto',random_state=42).fit_predict(X)
        m=internal_metrics(labels,X,n); m['k']=kk
        return labels,m,curve,{'k':kk}
    cand=hdb_candidates(X,n)
    if not cand:
        labels=AgglomerativeClustering(n_clusters=2,linkage='ward').fit_predict(X)
        m=internal_metrics(labels,X,n); m['fallback']=True
        return labels,m,[],{'fallback':True,'k':2}
    if params and params.get('min_cluster_size'):
        mcs=int(params['min_cluster_size'])
        labels=HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        m=internal_metrics(labels,X,n); m['min_cluster_size']=mcs
        curve=[{k:v for k,v in c.items() if k!='_labels'} for c in cand]
        return labels,m,curve,{'min_cluster_size':mcs}
    best=max(cand,key=lambda x:x['score'])
    labels=np.asarray(best['_labels']); m={k:v for k,v in best.items() if k!='_labels'}
    curve=[{k:v for k,v in c.items() if k!='_labels'} for c in cand]
    return labels,m,curve,{'min_cluster_size':best['min_cluster_size']}


def refit_on_matrix(X, alg, params, seed=42):
    n=len(X)
    if alg=='agglomerative': return AgglomerativeClustering(n_clusters=min(params['k'],n-1),linkage='ward').fit_predict(X)
    if alg=='kmeans': return KMeans(n_clusters=min(params['k'],n-1),n_init='auto',random_state=seed).fit_predict(X)
    return HDBSCAN(min_cluster_size=max(2,min(int(params['min_cluster_size']),n-1))).fit_predict(X)


def stability_score(X, full_labels, alg, params, runs=12, fraction=.8):
    n=len(X); rng=np.random.default_rng(1234); vals=[]
    size=max(5,int(round(n*fraction)))
    for i in range(runs):
        idx=np.sort(rng.choice(n,size=min(size,n),replace=False))
        try:
            sub=refit_on_matrix(X[idx],alg,params,42+i)
            vals.append(adjusted_rand_score(np.asarray(full_labels)[idx],sub))
        except Exception: pass
    return float(np.mean(vals)) if vals else 0.0


def perturbation_robustness(X, base_labels, algorithm, k=None, runs=15, sample_fraction=0.90):
    """
    Robustness via repeated 90% subsampling.

    Each run:
      1. sample 90% of configurations without replacement,
      2. refit the same clustering method on the subsample,
      3. compare the new labels against the original labels for those same
         configurations using Adjusted Rand Index (ARI).

    This avoids synthetic feature perturbations and measures whether the
    clustering structure survives modest data removal.
    """
    X = np.asarray(X, dtype=float)
    base_labels = np.asarray(base_labels)
    n = len(X)

    if n < 5:
        return 0.0

    scores = []

    for seed in range(runs):
        rng = np.random.default_rng(1000 + seed)
        sample_n = max(3, int(round(n * sample_fraction)))
        idx = np.sort(rng.choice(n, size=sample_n, replace=False))
        Xs = X[idx]

        try:
            if algorithm == "agglomerative":
                kk = int(k or len(set(base_labels.tolist())))
                kk = max(2, min(kk, len(Xs) - 1))
                new_labels = AgglomerativeClustering(
                    n_clusters=kk,
                    linkage="ward"
                ).fit_predict(Xs)

            elif algorithm == "kmeans":
                kk = int(k or len(set(base_labels.tolist())))
                kk = max(2, min(kk, len(Xs) - 1))
                new_labels = KMeans(
                    n_clusters=kk,
                    n_init="auto",
                    random_state=42 + seed
                ).fit_predict(Xs)

            elif algorithm == "hdbscan":
                # Reuse an approximate density scale from the full result.
                non_noise = base_labels[base_labels != -1]
                base_k = len(set(non_noise.tolist())) if len(non_noise) else 2
                mcs = max(2, min(12, int(round(len(Xs) / max(2, base_k * 4)))))
                new_labels = HDBSCAN(
                    min_cluster_size=mcs
                ).fit_predict(Xs)
            else:
                continue

            ref = base_labels[idx]
            new_labels = np.asarray(new_labels)

            # For HDBSCAN, compare only points assigned to a real cluster in
            # both runs. Noise is useful information but should not dominate
            # the robustness score.
            mask = (ref != -1) & (new_labels != -1)

            if mask.sum() < 3:
                continue
            if len(set(ref[mask].tolist())) < 2:
                continue
            if len(set(new_labels[mask].tolist())) < 2:
                continue

            scores.append(
                float(adjusted_rand_score(ref[mask], new_labels[mask]))
            )

        except Exception:
            continue

    if not scores:
        return 0.0

    return float(np.mean(scores))

def minmax(vals, reverse=False):
    arr=np.array([np.nan if v is None else float(v) for v in vals],dtype=float)
    valid=~np.isnan(arr)
    out=np.full(len(arr),.5)
    if valid.any():
        lo=np.nanmin(arr); hi=np.nanmax(arr)
        out[valid]=1.0 if abs(hi-lo)<1e-12 else (arr[valid]-lo)/(hi-lo)
    if reverse: out=1-out
    return out


def agreement_matrices(result_rows):
    names=[r['algorithm'] for r in result_rows]
    ari=[]; nmi=[]
    for a in result_rows:
        ar=[]; nr=[]
        for b in result_rows:
            ar.append(float(adjusted_rand_score(a['_labels'],b['_labels'])))
            nr.append(float(normalized_mutual_info_score(a['_labels'],b['_labels'])))
        ari.append(ar); nmi.append(nr)
    return names,ari,nmi


def evaluation_bundle(X,n,result_rows):
    names,ari,nmi=agreement_matrices(result_rows)
    for i,r in enumerate(result_rows):
        r['stability']=stability_score(X,r['_labels'],r['algorithm'],r['_params'])
        r['robustness']=perturbation_robustness(X,r['_labels'],r['algorithm'],r['_params'])
        others=[j for j in range(len(result_rows)) if j!=i]
        r['avg_ari']=float(np.mean([ari[i][j] for j in others])) if others else 1.0
        r['avg_nmi']=float(np.mean([nmi[i][j] for j in others])) if others else 1.0
        r['agreement']=(r['avg_ari']+r['avg_nmi'])/2

    sil=minmax([r['metrics']['silhouette'] for r in result_rows])
    db=minmax([r['metrics']['davies_bouldin'] for r in result_rows],reverse=True)
    ch=minmax([r['metrics']['calinski_harabasz'] for r in result_rows])
    for i,r in enumerate(result_rows):
        internal=(sil[i]+db[i]+ch[i])/3
        # Majority on k is intentionally not part of this score.
        final=(.30*r['stability']+.20*r['robustness']+.20*r['agreement']+.25*internal+.05*(1-r['metrics']['outlier_rate']))
        r['internal_composite']=float(internal); r['evaluation_score']=float(final)

    winner=max(result_rows,key=lambda r:r['evaluation_score'])
    explanation=(
        f"{winner['algorithm']} is recommended automatically. The decision does not use a majority vote on the number of clusters. "
        f"It combines stability ({winner['stability']:.2f}), 90% subsampling robustness ({winner['robustness']:.2f}), "
        f"agreement with other methods ({winner['agreement']:.2f}), internal cluster quality, and the outlier rate."
    )
    public=[]
    for r in result_rows:
        public.append({
            'algorithm':r['algorithm'],'metrics':r['metrics'],'stability':round(r['stability'],4),
            'robustness':round(r['robustness'],4),'avg_ari':round(r['avg_ari'],4),'avg_nmi':round(r['avg_nmi'],4),
            'agreement':round(r['agreement'],4),'internal_composite':round(r['internal_composite'],4),
            'evaluation_score':round(r['evaluation_score'],4),
        })
    return {
        'recommended_algorithm':winner['algorithm'],
        'recommended_k':int(winner['metrics']['k']),
        'explanation':explanation,
        'rows':public,
        'agreement':{'algorithms':names,'ari':[[round(x,3) for x in row] for row in ari],
                     'nmi':[[round(x,3) for x in row] for row in nmi]},
        'formula':'30% stability + 20% 90% subsampling robustness + 20% cross-algorithm agreement + 25% internal quality + 5% outlier retention',
    }



def derive_stable_range(curve, selected_k):
    """
    Find a practical cluster-count range rather than pretending one k is exact.

    Any candidate whose score is within 0.005 of the best score is treated as
    practically tied.

    Works for:
      - Agglomerative / K-Means curves: each row has k=<tested cluster count>
      - HDBSCAN curves: each row has k=<resulting cluster count> and
        min_cluster_size=<density parameter>
    """
    if not curve:
        return {
            "min_k": int(selected_k),
            "max_k": int(selected_k),
            "default_k": int(selected_k),
            "confidence": "Moderate",
            "note": "No cluster-count quality curve was available."
        }

    candidates = []
    for p in curve:
        if "score" not in p:
            continue

        # k is the actual resulting cluster count. For HDBSCAN this is not the
        # same as min_cluster_size.
        actual_k = p.get("k")
        if actual_k is None:
            actual_k = p.get("clusters")

        try:
            actual_k = int(actual_k)
        except Exception:
            continue

        if actual_k >= 2:
            candidates.append({
                "k": actual_k,
                "score": float(p["score"])
            })

    if not candidates:
        return {
            "min_k": int(selected_k),
            "max_k": int(selected_k),
            "default_k": int(selected_k),
            "confidence": "Moderate",
            "note": "No comparable cluster-count scores were available."
        }

    best = max(p["score"] for p in candidates)

    near = [
        p["k"]
        for p in candidates
        if best - p["score"] <= 0.005
    ]

    if not near:
        near = [int(selected_k)]

    # Deduplicate because HDBSCAN may produce the same cluster count for
    # several min_cluster_size values.
    near = sorted(set(near))
    min_k, max_k = min(near), max(near)

    if len(near) == 1:
        confidence = "High"
    elif max_k - min_k <= 2:
        confidence = "Moderate"
    else:
        confidence = "Low"

    if min_k == max_k:
        note = (
            f"{min_k} clusters is clearly preferred within the tested candidates."
        )
    else:
        note = (
            f"{min_k}–{max_k} clusters are practically tied "
            f"(within 0.005 of the best score)."
        )

    return {
        "min_k": int(min_k),
        "max_k": int(max_k),
        "default_k": int(selected_k),
        "confidence": confidence,
        "note": note,
    }

def json_safe(v):
    if pd.isna(v): return None
    return v.item() if hasattr(v,'item') else v


def explain_cluster(subset, cols):
    """
    Deterministic categorical explanation without an LLM.
    Reports dominant categorical values and their prevalence.
    """
    parts = []

    for c in cols:
        vc = subset[c].fillna("Unknown").astype(str).value_counts()
        if len(vc) == 0:
            continue

        top = str(vc.index[0])
        share = float(vc.iloc[0] / max(1, vc.sum()))

        if share >= 0.80:
            parts.append(f"{c} is predominantly {top} ({round(share * 100)}%)")
        elif share >= 0.60:
            parts.append(f"{c} is mostly {top} ({round(share * 100)}%)")
        else:
            common = ", ".join(map(str, vc.head(3).index.tolist()))
            parts.append(f"{c} is mixed; common values: {common}")

    return "; ".join(parts[:6]) + "."


def cluster_details(cfg,X,cols,labels):
    labels=np.asarray(labels); t=cfg.copy(); t['_cluster']=labels; groups=[]
    vals=sorted([x for x in t['_cluster'].unique() if x!=-1]);
    if -1 in t['_cluster'].unique(): vals.append(-1)
    for lab in vals:
        idx=np.where(labels==lab)[0]; sub=t[t['_cluster']==lab].copy()
        if lab==-1:
            canonical=None; explanation='Outliers/noise: these configurations were not similar enough to a stable HDBSCAN cluster.'
        else:
            Xg=X[idx]; medoid=idx[0] if len(idx)==1 else idx[int(np.argmin(pairwise_distances(Xg).sum(axis=1)))]
            canonical={c:json_safe(cfg.iloc[medoid][c]) for c in cols}; explanation=explain_cluster(sub,cols)
        groups.append({'group':'Outliers' if lab==-1 else int(lab)+1,'canonical':canonical,
                       'configuration_count':int(len(sub)),'node_count':int(sub['node_count'].sum()),'explanation':explanation,
                       'members':[{'values':{c:json_safe(r[c]) for c in cols},'node_count':int(r['node_count'])} for _,r in sub.sort_values('node_count',ascending=False).head(50).iterrows()]})
    return sorted(groups,key=lambda g:g['node_count'],reverse=True)


def projection_points(X,labels):
    if len(X)<2: return []
    coords=PCA(n_components=2,random_state=42).fit_transform(X) if X.shape[1]>=2 else np.column_stack([X[:,0],np.zeros(len(X))])
    mn,mx=coords.min(0),coords.max(0); span=np.where(mx-mn==0,1,mx-mn); norm=(coords-mn)/span
    return [{'x':round(float(norm[i,0]),5),'y':round(float(norm[i,1]),5),'cluster':int(labels[i])} for i in range(len(labels))]

@app.get('/',response_class=HTMLResponse)
def index(): return (APP_DIR/'static'/'index.html').read_text(encoding='utf-8')

@app.get('/api/status')
def status():
    try:
        df=load_df(); cols=[c for c in FEATURE_COLUMNS if c in df.columns]
        return {'ok':True,'rows':int(len(df)),'sites':int(df['site'].nunique()) if 'site' in df else 0,
                'clusters':int(df['cluster'].nunique()) if 'cluster' in df else 0,'columns':cols,'csv':CSV_PATH.name,'source':'Local data/Grid5000.csv'}
    except Exception as e: return {'ok':False,'error':str(e)}

@app.get('/api/analyze')
def analyze(columns:str=Query('site,cluster,cpu_model'),algorithm:str=Query('auto'),k:int|None=Query(None)):
    try:
        cols=[c.strip() for c in columns.split(',') if c.strip()]
        if len(cols)<2: raise HTTPException(400,'Select at least 2 columns.')
        df=load_df(); missing=[c for c in cols if c not in df.columns]
        if missing: raise HTTPException(400,f'Missing columns: {missing}')
        cfg=unique_configurations(df,cols); n=len(cfg)
        if n<3: raise HTTPException(400,'Need at least 3 unique configurations.')
        X=encode_configs(cfg,cols)
        raw=[]
        for name in ['agglomerative','kmeans','hdbscan']:
            lab,m,curve,params=fit_algorithm(X,n,name,None)
            raw.append({'algorithm':name,'metrics':m,'curve':curve,'_labels':lab,'_params':params})
        evaluation=evaluation_bundle(X,n,raw)

        if algorithm=='auto': chosen_alg=evaluation['recommended_algorithm']
        elif algorithm in {'agglomerative','kmeans','hdbscan'}: chosen_alg=algorithm
        else: raise HTTPException(400,'Unsupported algorithm.')
        selected=next(r for r in raw if r['algorithm']==chosen_alg)
        if k and chosen_alg in ('agglomerative','kmeans'):
            labels,metrics,curve,params=fit_algorithm(X,n,chosen_alg,k)
        else:
            labels,metrics,curve,params=selected['_labels'],selected['metrics'],selected['curve'],selected['_params']

        comparison=[{'algorithm':r['algorithm'],'metrics':r['metrics']} for r in raw]
        groups=cluster_details(cfg,X,cols,labels)
        return {'columns':cols,'nodes':int(len(df)),'unique_configurations':int(n),'selected_algorithm':chosen_alg,
                'selected_k':int(metrics['k']),'metrics':metrics,'curve':curve,'comparison':comparison,
                'evaluation':evaluation,'groups':groups,'projection':projection_points(X,labels)}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,str(e))
