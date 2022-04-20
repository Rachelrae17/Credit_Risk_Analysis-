# Credit_Risk_Analysis-
Module 17


Overview: The Purpose of this project was to use machine Learning Models tools to input or apply in doing this analysis a credit card dataset from Lending Club. 
Machine Learning can help identify Farudulemt charges that may occur. With Evaluating oversampling, undersampling, Combined approach of Machine Learning tools to apply to the dataset of the credit card company in doing this study.  


Results- There are six models for Machine Learning being discribe and solved. 

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

df.head()

Next Step was to change variables datatype object into a Numeric datatype. 

# convert to df
df_types = pd.DataFrame(df.dtypes)
df_types.rename(columns = {0: "Dtype"}, inplace=True)
df_types.index = df_types.index.set_names(['Feature'])
df_types.reset_index()

# get df with only features having dtype 'object'
feature_strings = df_types[df_types['Dtype']=='object']

# get list of features that are objects
string_cols = feature_strings.index.values.tolist()

# remove loan_status from cols list
string_cols.remove('loan_status')

# create training variables by converting string values into numerical values using get_dummies()
df = pd.get_dummies(df, columns=string_cols)
df.head()

Next Step was to clean the dataset and set one variable (x) and the target into a second Variable (Y). 

# Create our features
X = df.drop(columns='loan_status')

# Create our target
y = df['loan_status']
X.describe()

Next Step Balance is to make sure the balance of the values are checked. 

y.value_counts() 

Next Step was making sure the dataset is divided into a training set and testing set. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)
Counter(y_train)

The Final results in a majority set of the "Low Risk", Numbering 51,357 and the "High Risk" Numbering of 255. 

Model 1- Naive Random Oversampling 

This Model Balances the different Classification Categories By Oversampling the minority set, When Fraudulent classification are Random they may Be chosen to provide two classification that can be balanced with a equal Number of cases. The Reason for the Machine Learning Algorithm is to see a sufficient number of fraudulent cases so that the future data points may be accurately acessed and classified. 

Next Step For Random Over Sampler Model 

# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=12)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
Counter(y_resampled)

The Results were 51,357 data points In each classification Set. 


# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=12)
model.fit(X_resampled, y_resampled)

The Next Step Predictions were then created using a testing set and a Balanced Accuracy Score, With confusion Matrix and an Imbalanced Classification Report was then generated and produced. 

# Calculated the balanced accuracy score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
y_pred = model.predict(X_test)
bas = balanced_accuracy_score(y_test, y_pred)

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual_0', 'Actual_1'], columns=['Pred_0', 'Pred_1'])

# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
cri = classification_report_imbalanced(y_test, y_pred)

![oversampling_results](https://user-images.githubusercontent.com/95897182/164316763-60b57b71-6a3c-411c-b9fc-2f772593014c.png)


Model 2- Smote Oversampling 
This is the second model in which utilized oversampling. There are data points from the minority set that then leads to oversampled and new data points into existing data points. Then there are new values that are created and add to the sample sets. 

# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=12,
                                 sampling_strategy='auto').fit_resample(X_train, y_train)
Counter(y_resampled)

Smote Results of machine learning were: 


![smote_results](https://user-images.githubusercontent.com/95897182/164318777-5132e1ef-d862-4559-9df9-f6f4457a4348.png)

Model 3- Cluster Centroids Undersampling 

In this model works by undersampling with the majority set. This Algorithm identifies as a cluster in the majority class then generates synthetic, data points which are called centriods, that are representative of clusters. The majority class is called undersampled down to the size of minority class. 

# Resample the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=12)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)

Results were 255 data points in each classification set. 

![cc_results](https://user-images.githubusercontent.com/95897182/164323192-39e69d7b-859f-4478-9ff6-8fef6a6dbb60.png)


Model 4- Smoteen Combination Sampling 
In This Model the combines undersampling techniques are used. The Minority of class is oversampled by using the Smote tecnique algorithm. 

# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=12)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)

![smoteenn_results](https://user-images.githubusercontent.com/95897182/164325833-7dc45069-05db-4204-a36f-ee9cfef375ca.png)

Model 5- Balanced Random Forest Classifier 
This Model is different than previous. This focuses on ensemble learning model in which multiple simple decision are combined to build a strong learning model. 

# import StandardScaler
from sklearn.preprocessing import StandardScaler

# create Standard Scaler instance
scaler = StandardScaler()

# fit the scaler
X_scaler = scaler.fit(X_train)

# scale data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

This data is Resampled and Fit 

# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
rf_classifier = BalancedRandomForestClassifier(n_estimators=100, random_state=12)

# fit the model
rf_classifier = rf_classifier.fit(X_train_scaled, y_train)

![randfor_results](https://user-images.githubusercontent.com/95897182/164327520-ae225d6e-9d5c-42e5-86d9-9172c41039ab.png)

In Addition benefit to using Random Forest Model Target With This Algorithm. 

# List the features sorted in descending order by feature importance
importances = sorted(zip(rf_classifier.feature_importances_, X.columns), reverse=False)
featured_scores = pd.DataFrame(importances, columns=['Importance', 'Features']).sort_values('Importance', ascending=False)
featured_scores = featured_scores[featured_scores.columns[::-1]]
featured_scores

![randfor_featurelist](https://user-images.githubusercontent.com/95897182/164333740-a8bc9ab1-5909-414a-9b88-a09e68d2ae5b.png)


Model 6- Easy Ensemble Adaboost Classifier 

In this last model the algorithn that will create a model for the dataset that then will evaluate the errors, from the first model. 

# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier
eec_classifier = EasyEnsembleClassifier(n_estimators=100, random_state=12)
eec_classifier = eec_classifier.fit(X_train_scaled, y_train)

![eeac_results](https://user-images.githubusercontent.com/95897182/164334759-fa940865-94b8-486b-b031-c6ba13311d9c.png)

Summary: 
All six of the models that were solved using algorithm. One through four models were used with applying, oversampling and undersampling or both. When using these algorithms for these four samples the best score was at a 62%. 
The fifth model for balance random forest classifier model score was 77.1%, this is higher than the first four models. 
The sixth model the easy Ensemble AdaBoost Classififer Model was the Best performing Model. This score was at 90.2%. The recall for low-risk was 0.96,Recall for high risk fraudulent charges was at 0.85 with support of 92 cases in the test set that was applied. For this project the model six the easy Ensemble AdaBoost model was the most supported and best choice for the Machine Learning approach In the LendingClub Credit Cards analysis. 
 
















