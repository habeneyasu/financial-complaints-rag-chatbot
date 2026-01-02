# Notebooks

This directory contains Jupyter notebooks for the Financial Complaints RAG Chatbot project.

## Notebook Organization

Notebooks are organized sequentially by task:

### Task 1: Data Loading and EDA

1. **`01_load_dataset.ipynb`**
   - Loads the full CFPB complaint dataset
   - Validates data structure
   - Provides initial data overview

2. **`02_eda_analysis.ipynb`**
   - Performs exploratory data analysis
   - Analyzes product distribution
   - Analyzes narrative lengths and presence
   - Creates visualizations

### Future Tasks

- Task 2: Vector Store Creation
- Task 3: RAG Pipeline Implementation
- Task 4: Application Development

## Usage

To run notebooks, ensure you have Jupyter installed:

```bash
pip install jupyter ipykernel
jupyter notebook
```

## Best Practices

1. **Run notebooks sequentially** - Each notebook builds on previous work
2. **Use the modular classes** - Import from `src/` for reusable functionality
3. **Save processed data** - Use `DataPreprocessor.save_processed_data()` to persist results
4. **Document findings** - Add markdown cells to document insights

## Dependencies

All required packages are listed in `requirements.txt`. Key dependencies:
- `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `jupyter` - Notebook environment
