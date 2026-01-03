# Codebase Reorganization Summary

The codebase has been reorganized into a meaningful folder structure for better maintainability and clarity.

## New Structure

```
Mémoire_Eugenia/
├── src/                    # Main application code
│   ├── main.py            # Main training script
│   ├── dashboard.py        # Streamlit dashboard
│   ├── start.py            # Entry point script
│   └── paths.py            # Path helper module
│
├── scripts/                # Utility scripts
│   ├── employee_analyzer_simple.py
│   ├── check_importance.py
│   ├── calculate_turnover_percentage.py
│   ├── create_database.py
│   ├── database_reader.py
│   ├── import_csv_to_database.py
│   ├── continuous_learning.py
│   ├── monitoring.py
│   ├── privacy_preserving.py
│   └── paths.py            # Path helper module
│
├── models/                 # ML model files (.pkl)
│   ├── turnover_criteria_model.pkl
│   ├── criteria_scaler.pkl
│   ├── criteria_encoder_*.pkl
│   └── ...
│
├── data/                   # Data files
│   ├── turnover_data.db    # SQLite database
│   ├── *.json              # Analysis results
│   └── archive/            # Archived CSV files
│
├── config/                 # Configuration files
│   └── config.yaml
│
├── docs/                   # Documentation
│   ├── README.md
│   ├── EXPLICATION_IMPORTANCE.md
│   └── LLM_DATA_GENERATION_README.md
│
└── requirements.txt        # Python dependencies
```

## Changes Made

1. **Created folder structure**: `src/`, `scripts/`, `models/`, `data/`, `config/`, `docs/`

2. **Moved files**:
   - Main application files → `src/`
   - Utility scripts → `scripts/`
   - Model files (.pkl) → `models/`
   - Data files (JSON, DB, CSV) → `data/`
   - Configuration → `config/`
   - Documentation → `docs/`

3. **Updated import paths**: All Python files now use path helper modules (`paths.py`) to reference files in their new locations:
   - `src/paths.py` - for files in `src/`
   - `scripts/paths.py` - for files in `scripts/`

4. **Path helper functions**:
   - `get_model_path(filename)` - Get path to model files
   - `get_data_path(filename)` - Get path to data files
   - `get_config_path(filename)` - Get path to config files
   - `get_db_path(filename)` - Get path to database files

## Usage

### Running the application

From the project root:
```bash
python src/start.py
```

Or directly:
```bash
python src/main.py          # Train model
python src/dashboard.py     # Run dashboard (with streamlit run)
python scripts/employee_analyzer_simple.py  # Analyze employees
```

### Importing modules

When importing from scripts or src:
```python
# From scripts folder
from paths import get_model_path, get_data_path

# Use the helpers
model = joblib.load(get_model_path('turnover_criteria_model.pkl'))
data = json.load(open(get_data_path('criteria_analysis_results.json')))
```

## Benefits

- **Better organization**: Clear separation of concerns
- **Easier maintenance**: Related files are grouped together
- **Scalability**: Easy to add new files in appropriate folders
- **Path management**: Centralized path handling prevents errors
- **Professional structure**: Follows Python project best practices



