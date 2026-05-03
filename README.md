# MLDebugger

**Live App:**
https://ml-debug.streamlit.app/

---

MLDebugger is a lightweight web app that helps identify why a machine learning model underperforms.
It analyzes datasets, detects common issues, evaluates a baseline model, and provides actionable recommendations.

---

## Features

* Dataset overview (rows, columns, data types)
* Missing value detection with severity levels
* Class imbalance detection
* Correlation analysis to identify redundant features
* Baseline model training using Logistic Regression
* Confusion matrix and recall-based error analysis
* Insight generation explaining model failures
* Priority-based recommendations for improving performance

---

## How It Works

1. Upload a CSV dataset
2. The app analyzes data quality issues:

   * Missing values
   * Class imbalance
   * Feature correlation
3. A baseline model is trained
4. Model performance is evaluated
5. The system identifies failure patterns (e.g., low recall)
6. It connects data issues to model behavior
7. Final output: prioritized actions to improve the model

---

## Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn

---

## Installation (Local)

```bash id="y6j7kp"
git clone https://github.com/your-username/ml-debugger.git
cd ml-debugger
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Plans

* Support for regression problems
* Multi-class classification handling
* Automated preprocessing suggestions
* Feature importance analysis
* Model comparison

---

## Author

Geet Ajay Lunkad
