import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import data from 'table1.csv'
table1_df = pd.read_csv('table1.csv')
# Print first rows 
print('\nOriginal data')
print(table1_df.head())

# Convert categorical data to int representation. NB: doesnt enumerate values consistently across columns not sure about that 
for col in table1_df.columns:
    labels, uniques = pd.factorize(table1_df[col])
    table1_df[col] = labels

print("\nAfter conversion:")
print(table1_df.head())

# Feature vector
X = table1_df.drop(columns='depart_on_time')
# Label vector
y = table1_df['depart_on_time']

# Split dataset into training dataset and testing dataset; NB: added stratify parameter for balance when splitting data for traininig cause too little data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4)
print('\n# of training samples: ', len(X_train))
print('# of testing samples: ', len(X_test), '\n')

#GINI IMPURITY
print("*************GINI IMPURITY*************\n")
# Create a decision tree classifier
dtreeGini = DecisionTreeClassifier(criterion='gini', random_state = 13)
# Train decision tree using training dataset (features and labels fed to model so it learns patterns)
dtreeGini.fit(X_train, y_train)

# Perform prediction on testing data (test how accurate the patterns it learnt are)
predictions = dtreeGini.predict(X_test)
print('Testing accuracy: ', accuracy_score(y_test, predictions))
print('Confusion matrix:\n', confusion_matrix(y_test, predictions))

# View the tree graphically. 
plt.figure(figsize=(15, 15))
# Creates a jpg file.
plot_tree(dtreeGini, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree - Gini - D-O-T", size = 20) 
plt.savefig("GiniDT-Departures_on_time.jpg")

#INFORMATION GAIN
print("*************INFORMATION GAIN*************\n")
# Create a decision tree classifier NB: Information Gain criterion is entropy
dtreeIG = DecisionTreeClassifier(criterion='entropy', random_state = 13)
# Train decision tree using training dataset (features and labels fed to model so it learns patterns)
dtreeIG.fit(X_train, y_train)

# Perform prediction on testing data (test how accurate the patterns it learnt are)
predictions = dtreeIG.predict(X_test)
print('Testing accuracy: ', accuracy_score(y_test, predictions))
print('Confusion matrix:\n', confusion_matrix(y_test, predictions))

# View the tree graphically. 
plt.figure(figsize=(15, 15))
# Creates a jpg file.
plot_tree(dtreeIG, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree - Info Gain - D-O-T", size = 20) 
plt.savefig("IG_DT-Departures_on_time.jpg")
