# Humor Detection Pipeline

A comprehensive machine learning project for humor detection using both traditional and modern NLP approaches with rigorous statistical analysis.

## Overview

This project implements and compares two different approaches for humor detection:
1. **BERT-based classifier** using transformer models
2. **TF-IDF + MLP classifier** using traditional feature engineering

The project includes comprehensive statistical analysis with multiple random seeds for robust evaluation.

## Features

- **Dual Model Approach**: Compare modern transformer-based models with traditional ML approaches
- **Statistical Rigor**: Multi-seed experiments with confidence intervals, significance testing, and effect size analysis
- **Comprehensive Evaluation**: Bootstrap confidence intervals, paired t-tests, McNemar's tests
- **Rich Visualizations**: Multiple plots showing performance distributions, comparisons, and statistical significance
- **Reproducible Results**: Proper random seed management for reproducible experiments

## Installation

### Prerequisites
- Python 3.8+ 
- Virtual environment (recommended)

### Setup

1. **Create and activate virtual environment:**
```bash
python3 -m venv humor_detection_env
source humor_detection_env/bin/activate  # On Windows: humor_detection_env\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- transformers
- datasets
- scipy
- pingouin
- statsmodels
- accelerate>=0.26.0 (required for PyTorch Trainer)

## Data Requirements

The script expects a CSV file named `hahackathon_data.csv` with the following columns:
- `text`: The text content to classify for humor
- `is_humor`: Binary target variable (0 for non-humor, 1 for humor)
- `humor_rating`: Numerical humor rating score
- `humor_controversy`: Controversy score
- `offense_rating`: Offense rating score

## Usage

1. **Ensure your data file is in the project directory:**
   - Place `hahackathon_data.csv` in the same directory as `humor_detection_pipeline.py`

2. **Activate the virtual environment:**
```bash
source humor_detection_env/bin/activate
```

3. **Run the script:**
```bash
python humor_detection_pipeline.py
```

## Output

The script generates several types of output:

### Console Output
- Progress updates for each random seed experiment
- Model training progress
- Comprehensive statistical analysis results
- Performance summaries with confidence intervals

### Generated Files

**Statistical Results:**
- `statistical_analysis_results_YYYYMMDD_HHMMSS.json`: Detailed numerical results

**Visualizations:**
- `boxplot_accuracy_distribution.png`: Accuracy distribution across seeds
- `boxplot_f1_score_distribution.png`: F1-score distribution
- `boxplot_auc_distribution.png`: AUC distribution
- `seed_comparison_accuracy.png`: Seed-by-seed accuracy comparison
- `seed_comparison_f1_score.png`: Seed-by-seed F1 comparison
- `seed_comparison_auc.png`: Seed-by-seed AUC comparison
- `confidence_intervals_comparison.png`: Model comparison with confidence intervals
- `performance_heatmap.png`: Performance heatmap
- `effect_sizes_significance.png`: Effect sizes and statistical significance

## Configuration

Key parameters can be modified at the top of `humor_detection_pipeline.py`:

```python
RANDOM_SEEDS = [42, 123, 456, 789, 1337]  # Random seeds for experiments
MODEL_NAME = "bert-base-uncased"           # BERT model to use
MAX_LENGTH = 128                           # Maximum sequence length
BATCH_SIZE = 16                            # Training batch size
NUM_EPOCHS = 3                             # Training epochs
TFIDF_FEATURES = 5000                      # Number of TF-IDF features
CONFIDENCE_LEVEL = 0.95                    # Statistical confidence level
N_BOOTSTRAP = 1000                         # Bootstrap iterations
```

## Statistical Analysis

The project includes comprehensive statistical analysis:

1. **Descriptive Statistics**: Mean, standard deviation, and confidence intervals
2. **Significance Testing**: Paired t-tests with Bonferroni correction
3. **Effect Size Analysis**: Cohen's d for practical significance
4. **Classification Comparison**: McNemar's tests for paired classification results
5. **Bootstrap Analysis**: Confidence intervals for all metrics

## Project Structure

```
.
├── humor_detection_pipeline.py  # Main script
├── requirements.txt          # Python dependencies
├── humor_detection_env/      # Virtual environment
├── README.md                 # This file
├── .github/
│   └── copilot-instructions.md
└── hahackathon_data.csv     # Your data file (not included)
```

## Troubleshooting

### Common Issues

1. **Missing data file**: Ensure `hahackathon_data.csv` is in the project directory
2. **CUDA issues**: The script will automatically use CPU if CUDA is not available
3. **Memory issues**: Reduce `BATCH_SIZE` if you encounter out-of-memory errors
4. **Long training time**: Reduce `NUM_EPOCHS` or use a smaller model for faster training

### Virtual Environment Issues

If you have issues with the virtual environment:
```bash
# Deactivate current environment
deactivate

# Remove old environment
rm -rf humor_detection_env

# Create new environment
python3 -m venv humor_detection_env
source humor_detection_env/bin/activate
pip install -r requirements.txt
```

## Results Interpretation

### Statistical Significance
- **p < 0.05**: Statistically significant difference
- **Cohen's d**:
  - 0.2: Small effect size
  - 0.5: Medium effect size  
  - 0.8: Large effect size

### Model Performance
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve
- **Confidence Intervals**: Range of likely true performance values

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.
