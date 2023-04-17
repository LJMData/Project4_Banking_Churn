# Predicting Churn for Bank Customers
![Banking_image](https://github.com/LJMData/Project4_Banking_Churn/raw/main/ScreenShots/Banking_Image.png)


## Context
A bank is looking to reduce customer churn by implementing a churn predictor tool. The tool would allow customer service reps to input key details about a customer before speaking with them, and then use machine learning to predict the likelihood of that customer churning. The goal is to enable reps to proactively address any issues or concerns a customer may have, and hopefully retain the customer before they decide to leave the bank. The bank hopes that this tool will lead to increased customer retention and ultimately improved financial performance.


## Part 1 : Data Preprocessing
This Jupyter Notebook is an example of data preprocessing techniques that can be used to prepare data for predictive modeling. It uses a dataset obtained from Kaggle that contains information about bank customers and their churn rates. The dataset can be downloaded from the following link: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers. This Dataset used a sample of 10,000 records. 

- Read in the CSV file using pandas
- Drop any missing values from the dataset
- Drop the 'RowNumber' and 'Surname' columns from the dataset
- Use one-hot encoding to convert object values into numerical values
- Select the columns that need to be normalized
- Create a scaler object
- Fit the scaler to the selected columns
- Transform the selected columns using the scaler

An example of the final data is below 

![Scaled_Data](https://user-images.githubusercontent.com/115428292/232575849-1e599b98-e9e5-47f4-ae3f-b9e668f706a2.png)


## Part 2: Model Selection 

### Logistic Regression

Dropped "Exited" and "CustomerId" columns 

![Drop](https://user-images.githubusercontent.com/115945473/232009937-cc165907-3023-4d7a-a283-9d460f22018a.jpg) 

Split and train the data with sklearn.model_selection train_test_split, use logistic regression model from SKLearn and to fit the model using the training data.

![Train](https://user-images.githubusercontent.com/115945473/232016546-2d40d1cf-bcf2-4875-a875-b0c1264e1e8b.jpg)


![FitLM](https://user-images.githubusercontent.com/115945473/232016647-d2f3be39-11f0-4da8-b122-53c5909e95e0.jpg)


Traing Data Scores

![TrainingDataScore](https://user-images.githubusercontent.com/115945473/232017815-d9599f82-3b6e-42b4-a6d7-1578c2c323e1.jpg)


Predict outcome of dataset show Accuracy Score of the model

![ModelAccuracyScore](https://user-images.githubusercontent.com/115945473/232019001-7cc6936e-411b-4ecd-88f9-2e4e470318e5.jpg)


Confusion Matrix

![ConfusionMatix1](https://user-images.githubusercontent.com/115945473/232024064-8f28be3c-0e15-4de2-b807-4f7568406c54.jpg)



Clasification Report

![ClassificationReport](https://user-images.githubusercontent.com/115945473/232024528-1d04e5ca-f7a1-4dc9-98f5-483620d10b10.jpg)


Use the LogisticRegression classifier and the resampled data to fit the model and make predictions

![Resample](https://user-images.githubusercontent.com/115945473/232026655-67e74836-8404-41bd-8b04-bd0b1c362031.jpg)


Confusion Matrix with oversample data

![ResampleDataConfusionMatrix](https://user-images.githubusercontent.com/115945473/232027366-713e44f2-d890-45c1-a285-30b30d6b36ea.jpg)


 Use moduleRandomOverSampler from the imbalanced-learn library to resample the data 
 
![balanceingData](https://user-images.githubusercontent.com/115945473/232430271-c436d404-9536-444f-9e86-75fbec2a9812.jpg)

 
 
 Clasification Report with resampled data

![ClassificationReport2](https://user-images.githubusercontent.com/115945473/232028052-8810966f-0c16-4132-b93b-ae531b8f19a8.jpg)

Simply put, hyperparameters allows you to customize how algorithms behave to a specific 
dataset. 

- Different from parameters, hyperparameters are specified by you and not by an internally learning algorithm.

- Picking the best hyperparameters for a model can be difficult, therefore we use random or grid search strategies for optimal values. 

Used Grid search 
Grid search is a brute-force search paradigm approach where we specify a list of values for different 
hyperparameters, and the computer will evaluate the model performance for each 
combination of those to obtain the optimal set. Fit the model by using the grid search classifier and this 
will take the LogisticRegression model and try each combination of parameters. Score the hypertuned model on the test dataset

![image](https://user-images.githubusercontent.com/115945473/232031582-a656a364-60bb-4d37-b57b-c10eeef5f88d.png)

### Random Forest
This script uses the Random Forest Classifier and GridSearchCV to optimize hyperparameters to classify customer churn. 

It first imports the necessary libraries and loads the processed dataset. 

It then splits the data into training and testing sets and creates a Random Forest Classifier with 100 estimators. It then fits the model and generates predictions, creates a confusion matrix, and calculates the accuracy score. 

![RandomForrest](https://user-images.githubusercontent.com/115428292/232575926-e5a6fa6e-ea9c-4783-8044-388da9375e47.png)

It visualizes the confusion matrix using Seaborn heatmap and creates a bar chart of the top 10 most important features. 

It then uses GridSearchCV to find the optimal hyperparameters and extracts the best decision forest. 

It creates a confusion matrix and calculates the accuracy score for this model.

In this case, there are 2,284 instances that were actually in class 0 and were correctly predicted as such, while 70 instances were actually in class 0 but were predicted as class 1. Similarly, 295 instances were actually in class 1 and were correctly predicted as such, while 351 instances were actually in class 1 but were predicted as class 0.

The accuracy score of the model is 0.861, meaning that the model correctly predicted the class of 86.1% of instances.

The classification report provides further evaluation metrics for the model. Precision, recall, and F1-score are provided for both classes, as well as their support. Precision measures how often the model correctly predicts the positive class (class 1), while recall measures how often the model correctly identifies actual positive instances. F1-score is a combination of precision and recall, with higher values indicating better performance.

Overall, the model performed well at identifying instances in class 0, with high precision and recall values. However, the model struggled to identify instances in class 1, with lower precision and recall values, indicating potential areas for improvement.

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
To begin with, the first step is to split the data into training and testing sets using the train_test_split() function. 

Next, a list of predictor columns needs to be defined and stored in a variable called predictors. To select the two best features based on their f-test scores, the SelectKBest() method from scikit-learn can be used, and the result can be stored in a variable called ft. 

The log of the p-values of the f-scores for each feature can be plotted using a barplot. 
The next step is to transform the training and testing sets to include only the two best features using the transform() method of the SelectKBest object. 

Finally, a for loop can be used to train a KNN classifier on the transformed training set and predict the labels of the transformed testing set for k values ranging from 1 to 20, and the accuracy scores can be stored in an array called mean_acc2. 

Overall, these steps can help in analyzing the data and selecting the best features for training the model, which can lead to better accuracy and predictions.

### Final Model Selection 

## Part 3: Creating a Front End Interface 
This code includes a Flask web application that accepts input from a user to check if a customer is likely to churn. The user inputs customer information such as age, credit score, tenure, balance, number of products, has credit card, is active member, and estimated salary. The information is sent to the server as a JSON object where it is then processed by the predict_churn function. The predict_churn function uses a decision tree model that has been loaded from a saved file to predict whether the customer is likely to churn or not. The result is returned as a JSON object and displayed in the web application. If an error occurs, it is handled gracefully and returned to the user. The web application is created using Flask and uses the Flask-CORS extension to handle cross-origin resource sharing. 

The result is retured as either Churned or Not Churned 

#### Churn Example 
![Churn](https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/Churn.png)

#### Not Churned Example 
![Not_churn](https://github.com/LJMData/Project4_Banking_Churn/blob/main/ScreenShots/No_Churn.png)


