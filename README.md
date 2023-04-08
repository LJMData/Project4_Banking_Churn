# Predicting Churn for Bank Customers
![Banking_image](https://github.com/LJMData/Project4_Banking_Churn/raw/main/ScreenShots/Banking_Image.png)


## Context
A bank is looking to reduce customer churn by implementing a churn predictor tool. The tool would allow customer service reps to input key details about a customer before speaking with them, and then use machine learning to predict the likelihood of that customer churning. The goal is to enable reps to proactively address any issues or concerns a customer may have, and hopefully retain the customer before they decide to leave the bank. The bank hopes that this tool will lead to increased customer retention and ultimately improved financial performance.


## Part 1 : Data Preprocessing
This Jupyter Notebook is an example of data preprocessing techniques that can be used to prepare data for predictive modeling. It uses a dataset obtained from Kaggle that contains information about bank customers and their churn rates. The dataset can be downloaded from the following link: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers

- Read in the CSV file using pandas
- Drop any missing values from the dataset
- Drop the 'RowNumber' and 'Surname' columns from the dataset
- Use one-hot encoding to convert object values into numerical values
- Select the columns that need to be normalized
- Create a scaler object
- Fit the scaler to the selected columns
- Transform the selected columns using the scaler

## Part 2: Model Selection 

### Logistic Regression

### Random Tree

### Descision Tree
The Data was loaded into a Jupyter Notebook and the variables were identified 

```ruby
X = df.drop(["Exited", "CustomerId","Surname","RowNumber","Geography","Gender"], axis=1)
y = df["Exited"]
```
This was then split into the training and testing sets and the features were defined

```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
The object was created and and fitted to the model

```ruby
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
```
This model was then used to make predictions and evaluted.

![DTC_Example](https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/Descision_tree_example.png)


The confusion matrix shows that the model predicted 1388 true negatives and 206 true positives, but misclassified 219 false negatives and 187 false positives. The classification report shows that the model has an accuracy of 79.70%, precision of 0.48, recall of 0.52, and f1-score of 0.50 for predicting churn customers. The weighted average precision, recall, and f1-score are all around 0.80, which indicates that the model is decent at predicting both churn and non-churn customers.

### KNN

### Final Model Selection 

## Part 3: Creating a Front End Interface 
This code includes a Flask web application that accepts input from a user to check if a customer is likely to churn. The user inputs customer information such as age, credit score, tenure, balance, number of products, has credit card, is active member, and estimated salary. The information is sent to the server as a JSON object where it is then processed by the predict_churn function. The predict_churn function uses a decision tree model that has been loaded from a saved file to predict whether the customer is likely to churn or not. The result is returned as a JSON object and displayed in the web application. If an error occurs, it is handled gracefully and returned to the user. The web application is created using Flask and uses the Flask-CORS extension to handle cross-origin resource sharing. 

The result is retured as either Churned or Not Churned 

#### Churn Example 
![Churn](https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/Churn.png)

#### Not Churned Example 
![Not_churn] (https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/No_Churn.png)


