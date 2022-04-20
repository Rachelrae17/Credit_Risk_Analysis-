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

