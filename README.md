# Legal Document Importance Prediction

A machine learning system to predict investigative importance scores (0-100) for legal documents, enabling automated prioritization of critical evidence.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project was developed for a Kaggle competition focused on digital forensics and investigative journalism. The goal is to build an ML model that automatically ranks legal documents by their investigative importance, helping analysts prioritize which documents to review first.

**Key Results:**
- **CV RMSE:** 4.64
- **Documents Analyzed:** 20,624
- **Features Engineered:** 130+
- **Models Used:** LightGBM, XGBoost, CatBoost

## Approach

### Data Pipeline
```
Raw Documents → Text Preprocessing → Feature Engineering → TF-IDF/SVD → Ensemble Model → Predictions
```

### Feature Engineering
- **Text Features:** TF-IDF vectorization with bigrams, reduced to 150 dimensions using TruncatedSVD
- **Count Features:** Power mentions, agencies, tags (from semicolon-separated lists)
- **Keyword Features:** Domain-specific keyword detection (legal terms, key individuals)
- **Categorical Encoding:** CV-safe target encoding with smoothing

### Model Architecture
- **Base Models:** LightGBM, XGBoost, CatBoost with 5-fold cross-validation
- **Meta-Learner:** Ridge regression stacking
- **Ensemble:** Weighted averaging based on inverse CV RMSE

## Project Structure

```
├── notebooks/
│   ├── solution_v1.py       # Initial submission
│   └── solution_v2.py       # Improved submission with stacking
├── images/
│   └── pipeline.png         # ML pipeline visualization
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/legal-document-importance-prediction.git
cd legal-document-importance-prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Run the solution
python notebooks/solution_v2.py
```

For Kaggle notebooks:
1. Create a new notebook from the competition page
2. Add competition data using "+ Add Data"
3. Copy the solution code and run

## Results

| Model | CV RMSE | Std |
|-------|---------|-----|
| LightGBM | 4.729 | ±0.055 |
| XGBoost | 4.725 | ±0.049 |
| CatBoost | 4.667 | ±0.071 |
| **Ensemble** | **4.644** | - |

## Tech Stack

- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, LightGBM, XGBoost, CatBoost
- **NLP:** TF-IDF, TruncatedSVD (LSA)
- **Data Processing:** Pandas, NumPy

## Key Techniques

1. **CV-Safe Target Encoding** - Prevents data leakage by encoding within cross-validation folds
2. **TruncatedSVD (LSA)** - Reduces TF-IDF matrix to dense features while preserving semantic meaning
3. **Ridge Stacking** - Learns optimal weights for combining base model predictions
4. **Early Stopping** - Prevents overfitting in gradient boosting models

## Competition

- **Platform:** Kaggle
- **Competition:** Bash 8.0 Round 2
- **Task:** Regression (predict importance score 0-100)
- **Metric:** Root Mean Squared Error (RMSE)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Suyash Shukla
- LinkedIn: [Your LinkedIn](www.linkedin.com/in/suyash-shukla-research-analyst)

---

*Built with data science for investigative journalism*
