# Grid5000 Configuration Clustering — Automatic Evaluation

Local-only. Put `Grid5000.csv` in `data/`.

Run `docker compose up --build` and open `http://localhost:7171`.

The same Clustering tab now automatically compares Agglomerative, K-Means and HDBSCAN using internal quality, subsample stability, perturbation robustness and ARI/NMI cross-algorithm agreement. Majority vote on k is not used to choose the recommended algorithm.


## Refined recommendation logic

The UI no longer treats one cluster count as uniquely correct when several values
perform almost the same.

It now reports:
- Recommended clustering method
- Stable cluster-count range
- Default cluster count
- Confidence level

Cluster counts within 0.005 of the best quality score are treated as practically tied.

Robustness evaluation was also strengthened with:
- stronger encoded-feature perturbations
- ~10% random feature dropout

The PCA plot is visualization only; overlapping configurations can appear as one point.


## Final evaluation refinement

Robustness now uses repeated **90% subsampling** rather than synthetic feature
noise.

For each algorithm the application:
- samples 90% of configurations without replacement,
- refits the same clustering method,
- compares assignments on the shared configurations with Adjusted Rand Index,
- repeats this 15 times and reports the mean ARI.

HDBSCAN stable-range logic now uses the **actual number of clusters produced**
for each tested `min_cluster_size`, so ranges such as `2–4 clusters` can be
shown correctly.
