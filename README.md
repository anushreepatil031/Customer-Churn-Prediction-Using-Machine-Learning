# Customer-Churn-Prediction-Using-Machine-Learning
This project aims to predict customer churn for a bank using machine learning techniques. Churn refers to the customers who are likely to leave the bank. Identifying such customers in advance allows the business to take necessary actions to retain them.

The dataset contains information about 10,000 customers, including demographic data (age, geography, gender), account information (balance, number of products), and behavioral indicators (credit card ownership, activity status).

I performed data preprocessing by removing irrelevant columns (like CustomerID and Name), encoding categorical features, and splitting the data into training and test sets. Then, I trained a Random Forest Classifier to predict whether a customer will leave the bank (Exited = 1) or stay (Exited = 0).

The project also includes model evaluation using metrics like accuracy, classification report, and confusion matrix.
