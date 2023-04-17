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


Used a sample of 10,000 records
Dropped "Exited", "CustomerId","NumOfProducts" columns 

![Drop](https://user-images.githubusercontent.com/115945473/232009937-cc165907-3023-4d7a-a283-9d460f22018a.jpg) 

Split and train the data with sklearn.model_selection train_test_split, use logistic regression model from SKLearn and to fit the model using the training data.

![Train](https://user-images.githubusercontent.com/115945473/232016546-2d40d1cf-bcf2-4875-a875-b0c1264e1e8b.jpg)


![FitLM](https://user-images.githubusercontent.com/115945473/232016647-d2f3be39-11f0-4da8-b122-53c5909e95e0.jpg)


Traing Data Scores

![TrainingDataScore](https://user-images.githubusercontent.com/115945473/232017815-d9599f82-3b6e-42b4-a6d7-1578c2c323e1.jpg)


Predict outcome of dataset show Accuracy Score of the model

![ModelAccuracyScore](https://user-images.githubusercontent.com/115945473/232019001-7cc6936e-411b-4ecd-88f9-2e4e470318e5.jpg)

Confusion Matrix![ConfusionMatix1](https://user-images.githubusercontent.com/115945473/232024064-8f28be3c-0e15-4de2-b807-4f7568406c54.jpg)


Clasification Report

![ClassificationReport](https://user-images.githubusercontent.com/115945473/232024528-1d04e5ca-f7a1-4dc9-98f5-483620d10b10.jpg)


Use the LogisticRegression classifier and the resampled data to fit the model and make predictions

![Resample](https://user-images.githubusercontent.com/115945473/232026655-67e74836-8404-41bd-8b04-bd0b1c362031.jpg)


Confusion Matrix with oversample data

![ResampleDataConfusionMatrix](https://user-images.githubusercontent.com/115945473/232027366-713e44f2-d890-45c1-a285-30b30d6b36ea.jpg)


 Use modulRandomOverSampler from the imbalanced-learn library to resample the data 
 
![balanceingData](https://user-images.githubusercontent.com/115945473/232430271-c436d404-9536-444f-9e86-75fbec2a9812.jpg)

 
 
 Clasification Report with resampled data

![ClassificationReport2](https://user-images.githubusercontent.com/115945473/232028052-8810966f-0c16-4132-b93b-ae531b8f19a8.jpg)

This test is ran on unresampled data.

![image](https://user-images.githubusercontent.com/115945473/232029062-b61df275-24dc-4fb5-8f87-7746eb71d5cc.png)

![blob2](https://user-images.githubusercontent.com/115945473/232030551-31d856a4-1624-4131-b63a-503088539ebc.jpg)

Fit the model by using the grid search classifier and this will take the LogisticRegression model and try 
each combination of parameters. Score the hypertuned model on the test dataset


![image](https://user-images.githubusercontent.com/115945473/232031582-a656a364-60bb-4d37-b57b-c10eeef5f88d.png)


Create the RandomizedSearch estimator along with a parameter object containing the values to adjust

Fit the model by using the randomized search estimator and this will take the LogisticRegression model and 
try a random sample of combinations of parameters


![FinalResults2](https://user-images.githubusercontent.com/115945473/232034811-03c97596-f2cf-4138-862b-abe393ef60ab.jpg)




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

The Hyperparameters were then tuned with the help of GridSearchCV

``` ruby
from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {'max_depth': [3, 4, 5, 6, 7],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'max_features': [2, 3, 4, 5, 6]}

# Create the grid search object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)
```
After performing a grid search for the best hyperparameters, we found that the optimal values were:

max_depth: 5
max_features: 6
min_samples_leaf: 1
min_samples_split: 6

These hyperparameters improved the accuracy of our model to 85%. The confusion matrix showed The confusion matrix shows that there were 1559 true negatives, 48 false positives, 234 false negatives, and 159 true positives, which is also an improvement on the previous model, as were the following scores The accuracy of the model is 0.86 on a dataset of 2000 samples. The macro average precision is 0.82, recall is 0.69, f1-score is 0.72. 

### KNN

### Final Model Selection 

## Part 3: Creating a Front End Interface 
This code includes a Flask web application that accepts input from a user to check if a customer is likely to churn. The user inputs customer information such as age, credit score, tenure, balance, number of products, has credit card, is active member, and estimated salary. The information is sent to the server as a JSON object where it is then processed by the predict_churn function. The predict_churn function uses a decision tree model that has been loaded from a saved file to predict whether the customer is likely to churn or not. The result is returned as a JSON object and displayed in the web application. If an error occurs, it is handled gracefully and returned to the user. The web application is created using Flask and uses the Flask-CORS extension to handle cross-origin resource sharing. 

The result is retured as either Churned or Not Churned 

#### Churn Example 
![Churn](https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/Churn.png)

#### Not Churned Example 
![Not_churn](https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/No_Churn.png)


