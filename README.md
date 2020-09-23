# Telco Churn Project

## About the Project

## Goal 
- Find drivers for customer churn.

- Construct a ML classification model that accurately predicts customer churn.

- Create modules that make your process repeateable.

## Acknowledgement
The datasets were provided by CodeUp from a Sequel Database. You can learn more about codeup at https://codeup.com

## Data Dictionary
- Churn Rate (Churn): The churn rate, also known as the rate of attrition or customer churn, is the rate at which customers stop doing business with an entity and leave for a competitor. 
- DSL: "Digital Subscriber Line" DSL is an Internet connection delivered via telephone lines.
- Fiber: Fiber optic Internet is an Internet connection that transfers data fully or partially via fiber optic cables. “Fiber” refers to the thin glass wires inside the larger protective cable. “Optic” refers to the way the type of data transferred – light signals.
- Dummy variable: One that takes only the value 0 or 1 to indicate the absence or presence of some categorical effect that may be expected to shift the outcome.
- Chi Squared Test for Independence: A statical test that compares two variables in a contingency table to see if they are related.
- Hypothesis Testing: An act in statistics where an analyst tests an assumption regarding a population parameter.
- Null Hypothesis: A type of hypothesis used in statistics that proposes that there is no difference between certain characteristics of a population.
- Alternative Hypothesis: A type of hypothesis used in hypothesis testing that is contrary to the null hypothesis
- P-value: A p value is used in hypothesis testing to help you support or reject the null hypothesis. The p value is the evidence against a null hypothesis. The smaller the p-value, the stronger the evidence that you should reject the null hypothesis.
- Logistic Regression Algorithm: A classification algorithm, used when the value of the target variable is categorical in nature.
- Decision Tree Algorithm: A tree-structured classification algorithm, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.
- Random Forrest Algorithm: A classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
- KNN: A simple, supervised machine learning algorithm that can be used to solve both classification and regression problems by using similar data that is in close proximity of the target
- Classification Report: Used to measure the quality of predictions from a classification algorithm. The report shows the main classification metrics precision, recall and f1-score on a per-class basis.
- Confusion Matrix: A table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.
- Accuracy: The ratio of number of correct predictions to the total number of input samples.
- Precision: The number of correct positive results divided by the number of positive results predicted by the classifier.
- Recall: The number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).
- f1-score: The mean between precision and recall.

## Initial thoughts and Hypothesis

### Thoughts
- Overall churn rate is 26.5%
- Are there certain services where customers are unhappy?
### Hypothesis
- Hypothesis 1
> $H_0$: Customers contract type and churn are __independent__
> $H_a$: Customers contract type and churn are __dependent__
- Hypothesis 2
> $H_0$: Customers internet type and churn are __independent__
> $H_a$: Customers internet type and churn are __dependent__

## Project Plan
#### Acquire
- Data was acquired thru mysql telco_churn database provided by codeup
#### Prep
- Drop 11 rows where there were null values in total charges
- Change 3 category variables to 0/1 (device_protection, tech_support, online security, online backup, streaming tv, streaming movies and multiple lines)
- Create a new feature that represents tenure in years.
- Create dummy variables for gender, partner, dependents, phone_service, churn, paperless_billing, internet service, payment type, and contract type
- Add dummy variables back into dataframe
- Drop duplicate columns(payment_type_id, gender, partner, dependents, phone_service, churn, paperless_billing, contract_type, internet_service_type, payment_type    
- Split your data into train/validate/test.
- Create a prep_telco_data function
#### Explore
- Hypothesis testing and evaluation
#### Modeling  and Evalution
- Establish a baseline accuracy 
- Identify any variables that seem to provide limited to no additional information
- Train (fit, transform, evaluate) different models, varying the model type and hyperparameters.
- Compare evaluation metrics across all the models, and select the ones you want to test using your validate dataframe.
- Based on how your evaluation of your models using the train and validate datasets, choose your best model that you will try with your test data.
- Test the final model (transform, evaluate) on your out-of-sample data (the testing data set). Summarize the performance. Interpret your results.
#### Summary Conclusion
- In exploration, we found out that contract type and internet type are dependent of churn prediction by using Chi Squared Test.
- In modeling, we used Logistic Regression to find out that of the 27 different variables created to predict churn, 12 seem to be more influential than others.
- 4 more models were produced, another logistic regression, decision tree, random forest and a knn model.
- These models were evaluated against a baseline model with an accuracy of .734
- The logistic regression model performed the best and increased accuracy by 5% with an f1-score for churning customers staying between .565 and .597.
#### Next Steps
- Model needs some work to optimize precision and recall for a 1 in the churn_yes column. This is the predictor for churn vs a 0 is a predictor for keeping a customer. 
- Re-evalate inputs and find other variables that can have stronger correlations to churn.
- If alloted more time, a correlation graph could be produced for price and/or tenure. These two did not seem to be a factor of churn in the logistic regression model. 
## How to Reproduce
- Get permission from codeup to aquire the data from the sequel database.
- Install acquire.py, prepare.py and model.py into your working directory.
- Run the jupyter notebook.
- Form your own hypothesis and explore the data
- Create models and draw conclusions