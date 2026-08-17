# Grid'5000 Configuration Clustering — Local Only

This version performs **zero downloads** and makes **no Internet requests**.

Put your existing CSV here:

```text
grid5000_config_clustering/
└── data/
    └── Grid5000.csv
```

The filename must be exactly:

```text
Grid5000.csv
```

## Run

```bash
docker compose down
docker compose up --build
```

Open:

```text
http://localhost:7171
```

## Behavior

- If `data/Grid5000.csv` exists, the app loads it immediately.
- If the file is missing, the UI shows an error telling you where to place it.
- There is no download code.
- There is no Refresh button.
- There is no Grid'5000 API access.
- There is no GitHub access.

## Clustering

- Select any 2 or more dimensions.
- All selected dimensions are clustered together.
- Multiple k values are evaluated.
- Recommended k balances Silhouette separation and compression.
- Compare 3, 5, recommended, or custom group counts.
