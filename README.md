We have developed ML model using XG Boost alhorithm that predicts logic depth of circuit.
We have included features like input, output Fan in , Fan out,No of gates and trained data of 400+samples.
We generated random verilog codes and plotted it in vivado, Then we extracted feature data from random circuits and its depth using Vivado.
Then we tried different models like linear regression, Random Forest, XG boost and calulated mean square error and variance.
XG boost was most accurate.
Project Overview
This project develops aalgorithm to predict combinational logic 
Installation & Setup
1. Prerequisites
Ensure you have the following installed:

Python 3.x
Required Python packages (install via pip)
2. Install Dependencies
Run the following command to install necessary libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib scikit-learn xgboost
3. Dataset Preparation
Place the dataset DATASET.csv
in the project directory.

4. Running the Code
Execute the script with:

bash
Copy
Edit
python trained_modelcode.py
Approach Used to Develop the Algorithm


#The complexity 

 Data Generation & Feature Extraction (O(n)) 
1. Extracted Fan-in, Fan-out, Gate Count, Gate Types from Vivado. 
2. Each circuit is processed independently, leading to O(n) complexity. 
Model Training (O(n⋅ d⋅m)) 
1. Random Forest builds multiple trees, making training O(m⋅ n⋅ log n). 
2. XGBoost applies boosting, resulting in O(n⋅ d⋅m) complexity. 
Prediction Complexity 
1. Random Forest Prediction (O(m ⋅ log n)) – Each sample passes through multiple trees. 
2. XGBoost Prediction (O(log n)) – More efficient due to optimized traversal. 
Overall Complexity 
1. Feature Extraction: O(n) 
2. Training: O(n⋅ d⋅m) 
3. Prediction: O(logn) 
Conclusion 
1. High accuracy with reasonable training complexity. 
2. Fast logic depth prediction for pre-synthesis timing analysis. 
3. Scalable for large circuit datasets, improving timing violation detection.
