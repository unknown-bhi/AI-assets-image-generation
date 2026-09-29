# AI Asset Creator

In-house AI asset generation service — text-to-image, image-to-image, and variations using Stable Diffusion 1.5.

## 📋 Features

- **Text-to-Image** — prompt se image generate karo
- **Image-to-Image** — reference image se 80-90% similar assets
- **Variations** — ek prompt se multiple variations
- **Custom dimensions** — 512x512, 1024x1024 (auto-adjust to multiples of 8)
- **History tracking** — user generation history + daily quota
- **Bulk operations** — bulk upload + execute
- **NSFW safe** — Falconsai classifier se post-generation filter

## 🛠️ Tech Stack

- **Backend:** FastAPI
- **Database:** PostgreSQL (SQLAlchemy 2.0 async)
- **Migrations:** Alembic
- **ML:** Diffusers + PyTorch (CPU)
- **Model:** Stable Diffusion 1.5
- **Safety:** Falconsai NSFW classifier

## 🚀 Setup

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- 8 GB+ RAM
- ~5 GB disk space (for models)

### Installation

```bash
# Clone
git clone <your-repo-url>
cd AI_asseet_creator

# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Setup environment
cp backend/.env.example backend/.env
# Edit .env with your values

# Setup database
sudo -u postgres psql
# CREATE DATABASE ai_asset_creator;
# \q

# Run migrations
cd backend
alembic upgrade head

# Download models (one-time, ~4 GB)
python scripts/download_models.py

# Run server
uvicorn app.main:app --reload