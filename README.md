# Revolut Fraud Detection

This project aims to build a model to detect fraudsters on a Revolut Dataset.

The [dataset](https://www.kaggle.com/datasets/andrejzuba/revolutassignment?resource=download) was downloaded from Kaggle, and it contains three different CSV files.

One called transactions.csv with information about each transaction, user_id, timestamp, etc. Another is called users.csv, which, as the name says, has information about the user: country, age, creation date, etc. And finally, the fraudsters.csv, which contains only the user_id of the fraudsters.

The project comprehends the following phases:

1) **Merging** and **cleaning** the CSV files;
2) Check if the data is **balanced**. In this case, it was not, so I applied **undersampling** of the majority class;
3) **Econding** using **Target Encoding** and **One Hot Encoding**;
4) **Feature selection**, for this I used **Pearson Correlation**;
5) Try different Regression models. In the end, I chose the **Random Forest Regressor Model**
6) Model Evaluation.

The full description of the project can be followed on this Medium post:
<p  align="center">
	<img  width="500" src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*FAfnNRfDN7Duf9XuV8794w.jpeg"  alt="Material Bread logo">
</p>
<h1  align="center"><a  href = "https://medium.com/@macrodrigues/is-this-transaction-legit-revolut-fraud-detection-facddbc0fc80"> Medium Blog Post</h1>

