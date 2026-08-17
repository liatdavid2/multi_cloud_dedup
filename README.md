# Grid'5000 Configuration Clustering

Uses **real Grid'5000 reference-repository data**.

The application automatically downloads the public Grid'5000 reference repository from GitHub on first use, parses node metadata, creates:

```text
data/Grid5000.csv
```

and runs mixed categorical/numeric configuration clustering.

## Run

```bash
docker compose up --build
```

Open:

```text
http://localhost:7171
```

The first data download can take a little time.

## UI workflow

1. Download / refresh real Grid'5000 data.
2. Select exactly 3 infrastructure columns.
3. Click **Analyze clustering**.
4. The system tests multiple values of `k`.
5. It recommends a group count using:
   - Silhouette separation
   - Compression
6. Choose the recommended k, 3 groups, 5 groups, or another k.
7. See exactly which existing configurations belong to each proposed group.

No synthetic rows are generated.
