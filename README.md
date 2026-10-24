# 🎓 Intelligent Student Dropout Risk Prediction System

### An Explainable Machine Learning Approach to Early Student Support

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple" alt="SHAP">
  <img src="https://img.shields.io/badge/Status-Development-success" alt="Project Status">
</p>

---

## 📌 Project Overview

Student dropout is a significant challenge for higher education institutions. Identifying students who may be at risk of discontinuing their studies can help academic advisors and student-support teams provide timely assistance.

This project develops an **intelligent, explainable machine learning system for predicting student dropout risk** using multidimensional student data collected through a structured questionnaire at **Ladoke Akintola University of Technology (LAUTECH), Nigeria**.

The system compares multiple machine learning algorithms, addresses class imbalance, optimizes the classification threshold, and uses SHAP-based explainability to interpret model predictions. The selected model is integrated into a Flask web application that presents risk predictions, contributing factors, and suggested interventions.

The overall goal is to move beyond prediction alone and demonstrate how machine learning can support practical, data-informed student retention efforts.

## 🎯 Project Objectives

- Develop a machine learning model for identifying students at risk of dropout.
- Examine demographic, academic, financial, social, and health-related student characteristics.
- Compare the predictive performance of nine supervised machine learning algorithms.
- Address class imbalance using the Synthetic Minority Over-sampling Technique (SMOTE).
- Optimize model hyperparameters and the classification probability threshold.
- Interpret model predictions using SHapley Additive exPlanations (SHAP).
- Deploy the selected model through a Flask-based web application.
- Provide information that may help guide timely academic and related support.

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Student Data Analysis | Explores multidimensional student characteristics. |
| 🤖 Model Comparison | Evaluates nine supervised classification algorithms. |
| ⚖️ Class Imbalance Handling | Applies SMOTE to the training data. |
| 🛠️ Model Optimization | Uses hyperparameter tuning and probability-threshold optimization. |
| 📈 Model Evaluation | Measures accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC. |
| 🔍 Explainable AI | Uses SHAP to interpret global and individual predictions. |
| 🌐 Web Deployment | Integrates the selected model into a Flask application. |
| 💡 Decision Support | Displays predictions, contributing factors, and suggested actions. |

## 🧠 Machine Learning Models

Nine supervised classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. CatBoost
7. AdaBoost
8. Support Vector Machine (SVM)
9. K-Nearest Neighbours (KNN)

The models were compared using a common evaluation framework. Random Forest, CatBoost, and XGBoost were selected for further optimization based on their comparative performance.

Random Forest was ultimately selected for the deployed system after hyperparameter tuning and probability-threshold optimization.

## 📂 Dataset Description

The study used primary data collected through a structured questionnaire administered to students at LAUTECH.

The dataset covers several categories of student characteristics:

- **Demographic:** gender and age range.
- **Academic:** faculty, level of study, CGPA range, class attendance, and weekly study hours.
- **Financial:** financial support and scholarship status.
- **Social and educational environment:** parental education, accommodation, internet access, participation in activities, and academic support.
- **Health and wellbeing:** health challenges and school-related stress.

The target variable, *Dropout Intention*, is formulated as a binary classification outcome:

- `0` — Not at risk
- `1` — At risk

The project uses questionnaire-based indicators of dropout intention. Therefore, the model predicts the defined risk category rather than confirming that a student will actually leave university.

> **Data privacy:** Student-level data should not be published in the repository without appropriate authorization, privacy safeguards, and removal of identifying information.

## 🔐 Data Privacy and Confidenctial
The student questionnaire was designed to protect respondents’ privacy. No personally identifying information, such as students’ names, email addresses, or matriculation numbers, was collected in the responses used for this study. Confidentiality was considered during the survey design and data collection process, and the collected responses were used for the academic research and machine learning analysis described in this project.

## ⚙️ Project Workflow

The project follows an end-to-end machine learning workflow:

1. Data collection
2. Data validation and cleaning
3. Exploratory data analysis (EDA)
4. Data preprocessing and encoding
5. Stratified train-test splitting
6. Class imbalance handling using SMOTE on training data
7. Model training and comparative evaluation
8. Hyperparameter tuning
9. Probability-threshold optimization
10. Final model selection
11. SHAP-based model interpretation
12. Flask web application integration

The test set is retained for final evaluation and is not oversampled during model training.

## 📊 Model Performance

After model comparison, hyperparameter tuning, and probability-threshold optimization, Random Forest was selected as the final classifier.

### Final Random Forest Results

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 88.46% |
| Precision | 38.89% |
| Recall | 63.64% |
| F1-score | 48.28% |
| ROC-AUC | 78.95% |
| PR-AUC | 51.52% |
| Optimized threshold | 0.40 |

These results were obtained on the held-out test set.

**Why recall matters:** In an early-warning setting, missing a student who belongs to the at-risk class can limit opportunities for timely support. The selected model identified approximately 63.64% of the at-risk students in the test set, while its precision indicates that some students flagged as at risk were not members of the observed at-risk class.

The results illustrate the trade-off between identifying more potentially at-risk students and generating false-positive alerts. The model should therefore support, rather than replace, professional assessment by academic advisors.

## 🔍 Explainable Artificial Intelligence (XAI)

Predictive performance alone does not explain why a model produces a particular result.

This project incorporates SHAP to examine how input features contribute to model predictions.

The explainability component includes:

- **Global explainability:** SHAP beeswarm analysis to examine feature importance and the distribution of feature contributions.
- **Local explainability:** SHAP force plots to inspect factors influencing an individual prediction.
- **Decision support:** Presenting contributing factors alongside the predicted risk category.

The SHAP analysis reported in this study identified class attendance, internet access, and parental level of education as the most influential predictors, with hours of study and CGPA also contributing.

These are model-based associations, not proof that the features cause student dropout.

## 🌐 Web Application

The trained model is integrated into a Flask-based web application.

The application accepts student characteristics and generates one of two outcomes:

- **At-Risk of Academic Attrition**
- **Stable Academic Progress**

The interface also presents a prediction probability, contributing factors, and a suggested action based on the predicted outcome.

The application is intended as a prototype decision-support tool for academic advisors and student-support personnel. Predictions should be reviewed alongside relevant student circumstances before any intervention is made.

### Application Screenshots

Add your actual application screenshots here after uploading them to the repository.

For example, if you store the images in `images/`:

```markdown
### At-Risk Prediction

![At-Risk Prediction]("at-risk-prediction.png")

### Stable Academic Progress

![Stable Academic Progress]("stable-academic-progress.png")
```

## 🏗️ Project Structure

The repository includes notebooks, data files, and modular Python components. The structure below is a suggested presentation of the project; update it to match the actual files in your repository.

```text
Final-Year-ML-Project/
│
├── notebook/
│   ├── data/
│   │   ├── preprocessed_data.csv
│   │   └── ...
│   ├── EDA&FE.ipynb
│   └── Model_Building.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│   │   ├── exception.py
│   │   ├── logger.py
│   │   └── utils.py
├── static/
│   ├── style.css
├── template/
│   ├── home.html
│   └── index.html
├── .gitignore
├── application.py
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

**Note:** Keep private datasets, virtual environments, logs, and generated model artifacts out of version control unless there is a clear reason to include them. Use a suitable `.gitignore` file.

## 💻 Installation and Setup

### Prerequisites

- Jupyter Notebook
- Python 3.12
- Git
- A code editor such as Visual Studio Code

### 1. Clone the Repository

```bash
git clone https://github.com/OLATECh001/Intelligent-ML-System-for-Student-dropout.git
cd Final Year ML Project
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Ensure the dependency versions match the environment used to train and run the application.

### 4. Run the Application


```bash
python application.py
```

> The application requires the model artifacts and preprocessing components expected by the code. Confirm that these files are available and that the required data paths are configured before running it.

## 🛡️ Limitations and Future Improvements

Although the results demonstrate the potential of machine learning for student dropout-risk prediction, several limitations remain:

- The data were collected from one university, limiting immediate generalization to other institutions.
- Questionnaire-based dropout intention is not equivalent to verified future dropout.
- The minority at-risk class remains difficult to identify reliably.
- Additional longitudinal data could help evaluate how well predictions correspond to actual student outcomes.
- Further validation is needed across faculties, student subgroups, and institutions.
- Probability calibration, fairness assessment, and monitoring after deployment could strengthen future versions of the system.

Future work will focus on broader validation, improved minority-class identification, and more comprehensive evaluation of the system's usefulness for student-support decisions.

## 🧰 Technology Stack

- **Programming language:** Python
- **Data processing:** Pandas, NumPy
- **Data visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Gradient boosting:** XGBoost, CatBoost
- **Class imbalance:** imbalanced-learn (SMOTE)
- **Explainable AI:** SHAP
- **Web framework:** Flask
- **Development environment:** Jupyter Notebook, Visual Studio Code
- **Version control:** Git and GitHub

Verify that each listed package is present in the actual project dependencies before treating this list as the final environment specification.

## 👨‍💻 Author

**Rasheed Sikiru Olamide**

Computer Science Graduate | Data Science & Machine Learning

Interested in building practical machine learning systems, explainable AI applications, and data-driven software solutions.

- GitHub: [OLATECh001](https://github.com/OLATECh001)
- LinkedIn: [Olamide Rasheed](https://www.linkedin.com/in/olamide-rasheed-4535a6384/)

Replace the profile placeholders with your actual links.

## 📄 Citation

The article for this project is under writing, the citation will be listed once the project get published

## 📜 License

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Building interpretable machine learning systems for better-informed decisions.</strong>
</p>
