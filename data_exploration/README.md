# Recipe Engine - Data Collection

Collects and processes recipe data from the Food.com dataset on Kaggle.

## Quick Start

### 1. Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Kaggle API Setup
```bash
# Get API token from https://www.kaggle.com/account
# Download kaggle.json and place it in:
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run
```bash
# Test with 10 recipes
python main.py --limit 10

# Process all ~500K recipes
python main.py

# Skip download (if already downloaded)
python main.py --limit 10 --skip-download
```

## Output
Creates `data/processed_recipes.csv` and `data/processed_recipes.json` with:
- Recipe title, ingredients, instructions, tags
- Prep time, cook time, total time (in minutes)
- Ratings, categories, author info

## Options
- `--limit N`: Process only first N recipes
- `--skip-download`: Use existing data files
- `--output-format csv|json|both`: Choose output format
