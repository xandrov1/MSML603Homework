import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import data from 'Data_Iris.csv'
table_df = pd.read_csv('Data_Iris.csv')
# Get rid of any space that may be in the column names just in case
table_df.columns = table_df.columns.str.strip()
# Print first rows 
print('\nOriginal data')
print(table_df.head(),'\n')

# Convert categorical data to int representations of unique categories
for col in table_df.columns:
    labels, uniques = pd.factorize(table_df[col])
    table_df[col] = labels

print('After conversion to integer values:')   
print(table_df.head())

# Labels
y = table_df['species_name']
# features after removing label column
X = table_df.drop(columns='species_name')

# Split dataset (150) into training (120) and testing datasets (30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('\n# of training samples: ', len(X_train))
print('# of testing samples: ', len(X_test), '\n')

# a) Design a softmax regression classifier. For this, choose the ’lbfgs’ solver, which is the default in LogisticRegression.
print('Starting Softmax Logistic Regression fit')
logmodel = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
logmodel.fit(X_train, y_train)

# Get predictions 
predictions = logmodel.predict(X_test)

# Compute the testing accuracy and print out the confusion matrix
print('Softmax Logistic Regression results:')
print('LR test accuracy: ', accuracy_score(y_test, predictions))
print('Confusion matrix:\n', confusion_matrix(y_test, predictions))

# b) Design a soft margin support vector machine.
print('\nStarting Linear SVM fit')
svm_linear_model = LinearSVC(C=1.0)
svm_linear_model.fit(X_train, y_train)

# Get predictions
svm_linear_preds = svm_linear_model.predict(X_test)

# Compute the testing accuracy and print out the confusion matrix
print('Linear SVM results:')
print("Linear test SVM accuracy", accuracy_score(y_test, svm_linear_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, svm_linear_preds))

# c) Design a soft margin support vector machine using the radial basis function kernel.
print('\nStarting SVM fit')
svm_model = SVC(C=1.0)  # uses radial basis function
svm_model.fit(X_train, y_train)

# Get predictions
svm_preds = svm_model.predict(X_test)

print('SVM results:')
print("SVM test accuracy", accuracy_score(y_test, svm_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, svm_preds))