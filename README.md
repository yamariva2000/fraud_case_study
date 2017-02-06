#Ecommerce Fraud Detection Study


##Introduction

Starting with a dataset of about 151,000 unique customer transactions, I identified about 14,000 transactions that were classified as fraudulent which equates to an overall 9.4% fraud rate.  There was an additional dataset providing a mapping from the ip address to the customer country.  By joining these two datasets, I was able to match about 130,000 customers to their origin country.  About 22,000 remain unknown.  

Next, I looked at fraud rates for segments of the customer population.  One variable of interest was  the age of the customer account.  It would make sense that newer accounts without much history would tend to have a higher proportion of the fraudulent activities.  I calculated the difference between the sign up date and the purchase data, and binned the results into 4 quartiles.  I also binned customer ages into 5 quintiles and purchase value into 4 quartiles.  

![Fig 1](./figure_1.png)

Above is Fig. 1 which shows the fraud rate against various measures.  For visualization purposes, I did not calculate statistical significance. Referring to the first graph, referral source, the highest fraud rates were associated with direct targeting of the website.  Males tend to have a higher fraud rate than females.  Accounts with ages between 0 and 25 days have over a 20% fraud rate compared to all other accounts having less than a 16% rate.  There does not seem to be any pattern of fraud over customer age and purchase value.  For location, fraud was more prevalent from Oceana, South America, and Africa.  North America was slightly above the average, wheras Asia and  Europe were below the average rate.  The country vs fraud rate plot shows very high rates for various countries; however their contribution to overall fraud is small due to smaller numbers of transactions.  
  
 ![Fig 2](./figure_2.png) 
  
Looking at Fig. 2, one can see that as a percentage of the total fraudulent activity, the United States makes up the highest percent of fraud, followed by a category of 'unknown' countries, China, Japan, UK, Korea, etc.  Together, 10 countries make up 80% of all fraud. 
 
##The model and fraud detection strategy

 For the classifier model, I used 5 predictive variables including referring source, sex, continent, account_age, and customer age.  In order to guard against over-fitting I chose not to use each individual country in the model.  Next, I binarized each feature into category variables.  I ran various sklearn classifiers with train / test split of 70%, and found that random forest classifier was reasonally good and fast (using class_weights='balanced').  The F1-score which is a composite of recall and precision was 0.34 against the test data.
 
 Next, I made some assumptions about the financial impacts of various strategies.  For example, I assumed that the cost of an undetected fraudulent activity (false negative) would have a cost of the entire purchase value of that incident.  For cases in which the model predicted fraud, we would incur a $5 admin fee to process the inquiry into the transaction.  By spending this amount for predicting fraud, we would save the purchase value loss associated with a positive prediction that turned out to be a true case of fraud.  Figure 3 below shows that the average purchas value is about $36, which may not be high relative to the adminstrative costs.
 
 ![Fig 3](./figure_3.png) 
 
 
The following confusion matrix shows that the costs would be under various prediction scenarios. 
   
                     Prediction                               
|       |          |False  | True  |
|--------------|--------|------------|-------|
|Actual | False | True Negative      | False Positive (-5 admin)|
|  | True | False Negative (-Purchase Value)| True Positive (-5 admin)|


The model provides for each transaction, a probability that the transaction is fraudulent.  By using different threshold probabilities, one can estimate the average fraud costs of each strategy.  As seen in Figure 4 below, these costs vary with the threshold chosen.  A threshold of 0 would mean that all transactions are flagged, while a threshold of 1 would mean that no transactions are checked.  For each prediction, we calculate the cost based on the actual class and the predicted one, and then average the results to estimate the fraud cost per transaction.  

![Fig 4](./figure_4.png) 

Based on the structure of admin costs, there may be an incentive to check transactions if the administrative costs are fixed vs. variable.  At an admin cost of $5/transaction, the minimum fraud cost is $2.6 (admin + loss)at a threshold of 0.60.  At different administrative costs, this curve would vary substantially.

Random forest classifier has the ability to determine the most important features relevant to the classification. For this analysis, account age was important.  Other important features include customer age, continent, and sex.

 Fraud detection requires balancing the expected costs of fraud with the customer relationship.  If an inquiry is made, how would the customer perceive the experience?  Would the customer be required to contact its bank to verify charges?  On the other hand, if no fraud prevention is used, what is the potential exposure?  Since the transaction sizes tend to be under $40, the exposure is approximately $3.50 per transaction.  However, if transaction sizes increase and we do not actively monitor the situation, our business insurance rates would also rise.  
      
 For now, the biggest driver for fraud may be new accounts in high growth countries.  A follow-up step would be to monitor the fraud rates for this segment as these customer purchases become a larger percentage of revenues.     
 
 
