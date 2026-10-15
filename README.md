Traffic Accident Severity Prediction Using Machine Learning
Machine Learning Coursework — Phase 2: Baseline Implementation (Option A)
Project overview
This project investigates binary classification of fatal and non-fatal road collisions using the UK Department for Transport STATS19 2024 collision dataset. It implements an interpretable Logistic Regression baseline, compares it with class-weighted Logistic Regression and Random Forest, and explores hyperparameter tuning and validation-based classification thresholds.
The experiment is informed by Amiri et al. (2025), Machine learning approaches to traffic accident severity prediction: Addressing class imbalance. The research paper uses Australian hospitalisation-related data, whereas this project uses UK collision records. The results are not a direct replication or a controlled numerical comparison.
Dataset
- Publisher: UK Department for Transport (DfT)
- Source: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- Dataset: 2024 collision-level CSV (STATS19)
- Filename expected by the notebook: dft-road-casualty-statistics-collision-2024.csv
- Records in the supplied notebook execution: 100,927
- Binary target: fatal_target = 1 when collision_severity == 1 (fatal); 0 when severity is 2 or 3 (non-fatal).
The dataset is not included in this repository. Download it from the official source before running the notebook. This is a collision-level severity label, not an individual casualty fatality label.
Input features
speed_limit, number_of_vehicles, day_of_week, road_type, light_conditions, weather_conditions, road_surface_conditions, and urban_or_rural_area.
Negative coded values are treated as missing in the notebook. Numerical features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Preprocessing is fitted within scikit-learn pipelines.
Models and experiments
1. Standard Logistic Regression baseline.
2. Class-weighted Logistic Regression (class_weight="balanced").
3. Random Forest comparison model.
4. Logistic Regression hyperparameter search using GridSearchCV and stratified cross-validation.
5. Validation-based threshold selection and a revised train/validation/test experiment.
6. Dataset-quality checks, model-input tests, and automated assertions.
The notebook reports accuracy, balanced accuracy where applicable, fatal-class precision, fatal-class recall, fatal-class F1, average precision, confusion matrices, and precision–recall curves.
Selected results from the saved notebook execution
Measure	Revised final evaluation
Selected model	Logistic Regression
Selected validation threshold	0.0346
Accuracy	88.79%
Balanced accuracy	60.33%
Fatal precision	4.33%
Fatal recall	31.00%
Fatal F1	7.60%
Average precision	0.0410


The revised test set contained 300 fatal collisions, of which 93 were identified and 207 were missed. These are experimental results, not evidence of a deployment-ready road-safety system. The original baseline achieved high overall accuracy while missing fatal collisions at its default decision threshold.
Evaluation caveat: The revised experiment was designed after examining earlier test results. It should therefore be interpreted as an exploratory follow-up, not as a fully untouched confirmatory test.
How to run in Google Colab
1. Download the 2024 collision CSV from the official DfT page above.
2. Open Google Colab.
3. Upload and open the notebook file Traffic_Accident_Severity_Phase2 (3)(1).ipynb (or its renamed equivalent in this repository).
4. Choose Runtime → Run all. When the notebook's upload cell appears, upload dft-road-casualty-statistics-collision-2024.csv.
5. Check the dataset shape and target distribution, then inspect the model comparison, validation, test metrics, plots and testing output.
Note: The notebook uses google.colab.files.upload() and is intended to run in Colab. It needs the CSV file uploaded to the runtime each time the runtime is reset.
Python libraries
- pandas
- numpy
- matplotlib
- scikit-learn
- google.colab (available in Google Colab)
Colab generally provides these packages. In another Python environment, install the standard dependencies with:
pip install pandas numpy matplotlib scikit-learn notebook
The notebook may require small changes to its file-upload cell outside Colab.
Repository contents
Traffic-Accident-Severity-Prediction-ML/
├── README.md
└── Traffic_Accident_Severity_Phase2 (3)(1).ipynb
The notebook filename may be simplified after uploading; update this README to match if you rename it. Generated CSV results are created during execution and are not required for reproducing the experiment.
Limitations
- Fatal collisions are rare, so accuracy alone is misleading.
- Fatal-class precision and F1 remain low; many false alarms and missed fatal collisions remain.
- Some recorded features may not be available before a collision occurs. The model is best understood as a retrospective collision-severity classification baseline.
- The revised evaluation is exploratory because earlier test feedback informed the subsequent experiment.
- The UK dataset and the selected Australian research paper differ in geography, observation unit and outcome construction. Their scores must not be treated as directly comparable.
Research reference
M. A. Amiri, S. Afshari, and A. Soltani, “Machine learning approaches to traffic accident severity prediction: Addressing class imbalance,” Machine Learning with Applications, vol. 22, art. 100792, 2025. https://doi.org/10.1016/j.mlwa.2025.100792
Academic use
This repository documents a coursework baseline implementation for research and learning purposes. It is not designed or validated for operational safety decisions.
