set -e  # Exit on error

echo "=========================================="
echo "Recipe Engine Server Setup"
echo "=========================================="

# Check if processed_recipes.csv exists
DATA_FILE="data/processed_recipes.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found!"
    echo "Please ensure the processed recipes CSV file exists in the data/ directory."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the server
echo ""
echo "=========================================="
echo "Starting Recipe Engine Server..."
echo "=========================================="
cd backend
python app.py

