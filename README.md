# Employee Attrition Risk and Workforce Insights Using Data Analytics and Machine Learning

**Student:** Parth Lakhan | **Course:** TY B.Sc. IT
**Program:** IBM SkillsBuild Data Analytics with AI Academic Internship Program (BharatCares, in association with AICTE)
**Domain:** Human Resources Analytics

## Project overview

This project analyses employee data to understand workforce patterns, find groups with higher attrition, build and evaluate a machine-learning classification model, and give practical recommendations to HR decision-makers. Everything is in one Jupyter Notebook and follows the sequence **Data -> Information -> Insights -> Decisions -> Actions**.

## Business problem

Employee attrition can increase recruitment costs, disrupt teams, and cause organizations to lose experienced employees. Organizations need to understand workforce patterns, identify employee groups with higher attrition, and use data to guide retention strategies.

## Objectives

1. Inspect, check and clean the employee dataset.
2. Calculate attrition KPIs (overall, by department, job role, overtime, tenure, age group, satisfaction).
3. Visualize attrition patterns.
4. Train a baseline and machine-learning models and evaluate them with precision, recall, F1-score, ROC-AUC and a confusion matrix.
5. Provide a dashboard-style summary, insights, risks, opportunities and recommendations.

## Dataset

* **Name:** IBM HR Analytics Employee Attrition & Performance (fictional dataset created by IBM data scientists)
* **Source URL:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
* **Expected file:** `WA_Fn-UseC_-HR-Employee-Attrition.csv` (1,470 rows, 35 columns)

### Download and placement (no Kaggle API or credentials needed)

1. Open the Kaggle link above in your browser and sign in to Kaggle (free) if asked.
2. Click **Download** and extract the ZIP file.
3. Copy `WA_Fn-UseC_-HR-Employee-Attrition.csv` into the **same folder as the notebook** (a sub-folder named `data` also works).
4. If the file is missing, the notebook stops with a clear message telling you what to do.

> Note: the results in the delivered notebook were produced with IBM's public GitHub copy of this dataset (same shape and attrition counts), because Kaggle was not reachable from the environment that ran the notebook. Re-run the notebook with your Kaggle download; the numbers should match.

## Technologies used

Python 3, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, scikit-learn. No paid APIs or services.

## Project structure

```
Employee-Attrition-Analytics/
|-- Parth_EmployeeAttritionAnalytics.ipynb              # complete project (code + explanations + outputs)
|-- requirements.txt                                    # Python dependencies
|-- Parth_EmployeeAttritionAnalytics_ProjectReport.docx # Word project report
|-- README.md                                           # this file
|-- WA_Fn-UseC_-HR-Employee-Attrition.csv               # you add this (downloaded from Kaggle)
```

The CSV is not one of the four deliverables; you only add it locally to run the notebook.

## Setup on Windows

1. **Install Python 3.10 or newer** from https://www.python.org/downloads/ . During installation, tick **"Add Python to PATH"**.
2. Open **Command Prompt** (or PowerShell) and go to your project folder:
   ```
   cd C:\path\to\Employee-Attrition-Analytics
   ```
3. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
   Your browser opens; click `Parth_EmployeeAttritionAnalytics.ipynb`.

## How to execute the notebook

1. Make sure the CSV is in the same folder as the notebook.
2. In Jupyter choose **Kernel -> Restart & Run All** (or **Run -> Run All Cells**).
3. Wait about a minute. All cells should run without errors.
4. If a cell shows the "Dataset file ... was not found" message, place the CSV as described above and run again.

## Expected outputs

* Data inspection and data-quality tables.
* KPI table and attrition-rate tables by department, job role, overtime, age group, tenure group and income band.
* Charts: attrition distribution, department and job-role comparison, overtime / travel / marital status, age / tenure / income, satisfaction ratings, income and tenure box plots, correlation chart.
* Model comparison table, confusion matrix, ROC curves, threshold trade-off table, feature-importance chart.
* A dashboard-style summary figure and written findings, risks, opportunities and recommendations whose numbers are calculated from the data.

## Main KPIs and analyses (verified from the executed notebook)

* 1,470 employee records; 237 left; **overall attrition rate 16.1%**.
* By department: Sales 20.6%, Human Resources 19.0% (only 63 employees), Research & Development 13.8%.
* By job role: Sales Representative 39.8% (highest), Laboratory Technician 23.9%, Human Resources 23.1% (52 employees), Research Director 2.5% (lowest).
* Overtime 30.5% vs no overtime 10.4%; first-year employees 34.9%; ages 18-25 35.8%; lowest income quartile 29.3% vs highest 10.3%.
* Average job satisfaction 2.73 out of 4 (stayed 2.78, left 2.47).

## Machine-learning approach and evaluation

* **Target:** `Attrition` converted to 1 (left) / 0 (stayed). Identifiers, the target and derived group columns were excluded to prevent leakage.
* **Split:** stratified 80/20 train/test split (random_state = 42).
* **Pipeline:** StandardScaler for numeric columns and OneHotEncoder for categorical columns inside a scikit-learn `Pipeline`.
* **Imbalance:** handled with balanced class weights (no synthetic data, no extra library).
* **Models:** majority-class baseline, Logistic Regression, Random Forest. The model was selected using 5-fold stratified cross-validation on the training set only.
* **Test-set results (294 employees, 47 leavers):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.840 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression (selected) | 0.752 | 0.349 | 0.638 | 0.451 | 0.803 |
| Random Forest | 0.840 | 0.500 | 0.234 | 0.319 | 0.786 |

* Selected model confusion matrix: 191 true negatives, 56 false positives, 17 false negatives, 30 true positives.
* Most influential feature (permutation importance): OverTime, followed by NumCompaniesWorked, MaritalStatus, YearsSinceLastPromotion and BusinessTravel. Importance shows association, **not causation**.

## Main findings and recommendations (based on verified results)

1. **Review overtime and workload** - overtime employees left at about 2.9 times the rate of others, in every department.
2. **Strengthen first-year onboarding** - 34.9% attrition in the first year versus 6.7% at 11-20 years.
3. **Targeted retention reviews for high-attrition roles** - Sales Representative, Laboratory Technician, Sales Executive.
4. **Examine entry-level pay and progression** - the lowest income quartile and job level 1 have above-average attrition (income, level and age overlap).
5. **Manage frequent travel and use satisfaction pulse surveys** as early warnings.
6. **Use the model only as a supportive, aggregate planning aid** - never for automatic employment decisions.

No cost savings or percentage improvements are claimed, because the dataset has no cost data.

## Limitations

* Fictional, single-snapshot dataset; may not match a real organization.
* Patterns are associations, not proof of cause.
* Only 237 leavers (47 in the test set); small groups give unstable rates; metrics can change with a different split.
* Limited model tuning and no external validation.
* The model is exploratory and must not be used to make automatic employment decisions.

## References

* IBM HR Analytics Employee Attrition & Performance, Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
* pandas: https://pandas.pydata.org/docs/
* scikit-learn: https://scikit-learn.org/stable/
* Matplotlib: https://matplotlib.org/stable/ | Seaborn: https://seaborn.pydata.org/
* IBM SkillsBuild Data Analytics with AI Academic Internship Program, BharatCares with AICTE, 2026.
