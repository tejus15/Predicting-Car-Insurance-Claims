# Predicting-Car-Insurance-Claims
Trained and deployed five machine learning models (Logistic Regression, Decision Tree, Random Forest, KNN, SVM). Trained the models on original as well as oversampled data (to handle class imbalance) and found performance metrics. 

### Introduction 
According to LexisNexis, claims severities have steadily increased since 2020, with bodily injury rising by 20% and material damage rose by 47%. In 2023, 27% of collision claims were considered total losses, which involves payouts and consumers having to replace a vehicle or use other transportation. Moreover, the combined loss ratio for auto insurers in 2023 was 105% (LexisNexis). Early detection of car claims can help insurance companies identify high-risk customers and accurately price premiums for these higher-risk customers. Insurance companies can proactively reduce risk by taking steps to encourage safer driving through: discounts for customers in good standing, enrollment in driving courses, or pricing based on 
telematics data collected on every car.  
Safer driving can lead to fewer claims, which will lower claims costs and improve profitability for these insurance companies. Additionally, predicting the number of estimated claims can help determine the number of resources required for managing claims processes as well as the finances required to cover these claims.  

### Goal 
Classify whether a policyholder will file a car insurance claim within the next six months based on data about the policyholder and their vehicle. 

### Dataset 
The dataset used in this analysis is sourced from Kaggle and consists of 44 columns and 58,592 rows. The features provide information about policyholders and their vehicles for each claim. The dataset has 16 quantitative and 28 qualitative variables.  
The dependent variable, is_claim, is highly imbalanced with a positive class that makes up only 6.4% of the observations (~3,700). This class imbalance in our dataset is consistent with 
the imbalance present in real-world data, where only a small portion of policyholders file claims. For example, in 2022, 4.54% of collision insurance policyholders had claims (III)

### Project Report
Kindly refer to the attached report for a deeper understanding of our project 
[Group1_FinalReport(12.06.22).pdf](https://github.com/user-attachments/files/18328429/Group1_FinalReport.12.06.22.pdf)
