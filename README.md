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
