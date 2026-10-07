# Intelligent Branch Service Load and Customer Experience Optimizer

Hackathon project for optimizing bank branch staffing, service load, and customer experience using historical branch signals, machine learning, and a live operations dashboard.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  React + Vite   │────▶│  FastAPI backend │────▶│  Supabase Postgres  │
│  Tailwind       │     │  (API layer)     │     │  (live app data)    │
│  Recharts       │     └────────┬─────────┘     └─────────────────────┘
└─────────────────┘              │
                                 │ reads locally (training only)
                                 ▼
                        ┌──────────────────┐
                        │  CSV in /data    │
                        │  + ML scripts    │
                        │  (Pandas, sklearn│
                        │   XGBoost)       │
                        └──────────────────┘
```

### Data boundaries

| Layer | Role |
|-------|------|
| **`data/bank_branch_synthetic_dataset.csv`** | Synthetic training data kept in the repo. Used **only on the machine** that trains models (not uploaded to Supabase). |
| **Supabase PostgreSQL** | Live application data: branches, staff, predictions, recommendations, alerts, and dashboard aggregates. Schema will be added via `supabase/migrations/` in a later step. |
| **`backend/models/`** | Directory for serialized models produced by local training scripts (gitignored except `.gitkeep`). |

### Tech stack

- **Frontend:** React, Vite, Tailwind CSS, Recharts (scaffold pending)
- **Backend:** Python, FastAPI
- **ML:** Pandas, NumPy, scikit-learn, XGBoost (training scripts pending)
- **Database:** Supabase (migrations pending)

## Repository layout

```
intelligent-branch-optimizer/
├── data/                          # Synthetic CSV for local ML training
├── backend/
│   ├── app/                       # FastAPI application
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/
│   │   ├── services/
│   │   ├── schemas/
│   │   └── utils/
│   ├── models/                    # Trained model artifacts (local)
│   ├── scripts/                   # Training & batch jobs (future)
│   ├── tests/
│   └── requirements.txt
├── frontend/                      # React app (future)
├── supabase/migrations/           # SQL migrations (future)
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.11+ (recommended)
- Node.js 20+ (when the frontend is scaffolded)
- [Supabase CLI](https://supabase.com/docs/guides/cli) (when applying migrations)

## Backend (current)

From the `backend` directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

Run tests:

```bash
cd backend
pytest
```

Optional environment variables (create `backend/.env` when integrating Supabase):

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Do not commit real credentials.

## Frontend (planned)

The `frontend/` directory is reserved for a Vite + React + Tailwind + Recharts app. Setup commands will be documented once the UI is initialized.

## ML training (planned)

Training will read `data/bank_branch_synthetic_dataset.csv` locally and write artifacts to `backend/models/`. The training dataset will **not** be stored in Supabase.

## License

Hackathon project — add a license if you open-source the work.
