# PyLoan Machine Learning Report 

## Overview of the Analysis

In this project, we reviewed a dataset of 77,536 records with Loan and Borrower information.  From this, we devised two Machine Learning models to help predict whether a loan would be Healthy (Class 0) or High-Risk (Class 1). The dataset consisted of values for Loan Size, Interest Rate, Borrower Income, Debt to Income ratio, the borrower's total accounts, the borrower's derogatory marks (adverse credit), the borrower's total debts, and the known loan status.

In prepping the data, we separated the Loan Status column from the rest of the data.  We then used train_test_split to create the variables, then we used Fit to slot the variables into our models.  We then used those models to generate predictions with the test data.

Our two prediction models were based on Logistic Regression.  The first model used the base data, as it was presented to us.  The second model utilized Resampling for our dataset.  Because our original data was heavily weighted towards Healthy (Class 0) loans, resampling the data smoothed the data distribution, and attempted to reblance the values. To illustrate, see the two brief tables below:

    Original Data:
    loan_status
    0    75036
    1     2500

    Oversampled Data:
    loan_status
    0    56271
    1    56271

Our models used SKLEARN Logistics Regression.  We utilized Train_Test_Split to create our testing subsets.  We used balanced_accuracy to determine how well our models reflected our data.  We used confusion_matrix to determine the number of True Healthy, True High Risk, false negatives, and false positives were found in our predictions.  And we used the classification_report to display our results.

After reviewing the results, the model with the resampled data appeared to be more desirable for identifying high-risk loans.

## Results

Note:
Class 0 = Healthy Loan
Class 1 = High-Risk Loan

Model: Logistic Regression
  * Accuracy: 95.20%
  * Precision Scores:
      * Class 0: 1.00
      * Class 1: 0.85
  * Recall scores:
      * Class 0: 0.99
      * Class 1: 0.91
  * F1-scores:
      * Class 0: 1.00
      * Class 1: 0.88
----------------------------------------------
Model: Logistic Regression w/ Resampled Data
  * Accuracy: 99.36%
  * Precision Scores:
      * Class 0: 1.00
      * Class 1: 0.84
  * Recall scores:
      * Class 0: 0.99
      * Class 1: 0.99
  * F1-scores:
      * Class 0: 1.00
      * Class 1: 0.91
      
## Summary

Before resampling the data, the Logistic Regression model returned an accuracy score of 95%.  After resampling the data for the second model, the accuracy improved to 99%.  The majority of our errors in the second model were from false positives (the model predicting High-Risk, but the sample was actually Healthy).  While not ideal, it is probably better that the model was being more conservative in its assessment than allowing a large number of False Negatives to go through.  This tells us that our model may be a little too restrictive, but not devastatingly so. 

In looking at the classification report, we see that the prescision for identifying healthy loans (Class 0) is 100%, and the precision for high-risk (Class 1) loans is around 84%.  The precision is down a percentage point from the first test, but the Recall is much higher at 99%, resulting in a 91% for the new F1-Score.

Based on these metrics, the Logistic Regression w/ Resampled Data model appears to be more desirable for identifying high-risk loans (Class 1) due to its higher recall and F1-score. This suggests that the resampling technique has effectively improved the model's ability to identify High-Risk loan situations.

As for recommendations, either model would work well for identifying Healthy loans.  However, it seems imperative that we use the resampled data to improve our chances of accurately predicting High-Risk loans.  
