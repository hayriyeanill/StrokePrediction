# Artificial Neural Networks Class
# StrokePrediction Project

Stroke is a clinical syndrome that is characterized by signs and symptoms that indicate sudden or rapidly
developing partial loss of brain function originating from cerebral vessels, can last for 24 hours or longer and
may result in death. Stroke is the second most common cause of death in Turkey. 
In this paper, firstly, stroke estimation was made using neural network architectures and support vector
classification, and according to the results, the simple perceptron model with an epoch number of 50 gave the
best result with an accuracy of 0.95. 

# Dataset
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters
like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about
the patient. Dataset was selected from Kaggle public resource [8]. Dataset contains 5110 observations (rows)
and 12 features (columns). The dataset contains 7 numerical features (id, age, hypertension, heart_disease,
avg_glucose_level, bmi, stroke) and 5 nominal features (gender, ever_married, work_type, Residence_type,
smoking_status) that were converted into factors with numerical value designated for each level. ID information
has been removed from the dataset.

Dataset Available on: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

# Methods
1) Simple Perceptron
2) Multilayer Perceptron
3) Extreme Learning Machine
4) Support Vector Classification
