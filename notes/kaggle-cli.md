# Kaggle CLI quickstart

The repo has an isolated `.venv` with the Level 1 client packages installed.

```bash
cd /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle
source .venv/bin/activate
kaggle --version
```

Authentication is intentionally not performed automatically. Use the official browser OAuth flow when ready:

```bash
kaggle auth login
```

Then verify account access without downloading or submitting:

```bash
kaggle competitions list --group entered
kaggle competitions pages pokemon-tcg-ai-battle
kaggle competitions topics list pokemon-tcg-ai-battle -s hot --page-size 10
```

Download only after joining/accepting the competition rules:

```bash
mkdir -p data/competition
kaggle competitions download pokemon-tcg-ai-battle -p data/competition
```

Do not put credentials in this repository. `.env`, `.kaggle/`, and raw competition data are ignored by Git.
