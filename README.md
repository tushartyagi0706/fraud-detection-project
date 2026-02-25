Fraud Detection Project
Overview

In this project, I worked on detecting fraudulent financial transactions using machine learning. The dataset contains transaction details like type of transaction, amount, and balance changes.

The goal was to build a model that can identify fraud transactions and understand which factors contribute the most to fraud.

What I Did

Loaded and explored the dataset

Checked for missing values

Visualized transaction amount using a boxplot

Created a correlation heatmap to understand feature relationships

Dropped unnecessary columns (nameOrig, nameDest)

Converted transaction type into numerical format using encoding

Split the data into training and testing sets

Trained a Logistic Regression model

Evaluated performance using Confusion Matrix, Precision, Recall, and ROC-AUC

Why Logistic Regression?

This is a binary classification problem (fraud or not fraud).
Logistic Regression is simple, effective, and gives probability scores for each transaction.

I also used class_weight='balanced' because the dataset is imbalanced.

Key Observations

Transfer and Cash Out transactions were more likely to be fraud.

High transaction amounts increased fraud probability.

Balance-related features also had an impact.

These results make practical sense because fraud usually involves transferring and withdrawing money quickly.

Tools Used:

Python
Pandas
Matplotlib
Seaborn
Scikit-learn

Conclusion

This project helped me understand how fraud detection works in real-world scenarios and how important evaluation metrics like recall are when dealing with imbalanced data.
