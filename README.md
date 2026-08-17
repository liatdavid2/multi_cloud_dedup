# Multi-Cloud Resource Deduplication & Consolidation

## Dataset location

Put your dataset here:

```text
multi_cloud_dedup/
└─ data/
   └─ Cloud_Dataset.csv
```

The filename must be exactly:

```text
Cloud_Dataset.csv
```

No ZIP and no extraction are needed.

## Run with Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:2300
```

## Run without Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 2300
```

Then open:

```text
http://localhost:2300
```

The application:
1. Reads `data/Cloud_Dataset.csv`
2. Detects relevant CPU / memory / configuration columns
3. Finds near-duplicate configuration values
4. Creates consolidation recommendations
