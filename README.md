# Grid5000 Configuration Clustering — Automatic Evaluation

Local-only. Put `Grid5000.csv` in `data/`.

Run `docker compose up --build` and open `http://localhost:7171`.

The same Clustering tab now automatically compares Agglomerative, K-Means and HDBSCAN using internal quality, subsample stability, perturbation robustness and ARI/NMI cross-algorithm agreement. Majority vote on k is not used to choose the recommended algorithm.
