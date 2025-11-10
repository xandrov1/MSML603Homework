import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import data from 'table1.csv'
table_df = pd.read_csv('Data_Iris.csv')
# Get rid of any space that may be in the column names just in case
table_df.columns = table_df.columns.str.strip()
# Print first rows 
print('\nOriginal data')
print(table_df.head(),'\n')

# convert categorical data to int representations of unique categories
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

#KNN
print("*******************KNN*******************")
# Array to store accuracies and related k
resultsKNN = []

for k in range(3, 8, 2):
    classifierKNN = KNeighborsClassifier(n_neighbors=k)
    classifierKNN.fit(X_train, y_train)
    predictionsKNN = classifierKNN.predict(X_test)
    accuracyKNN = accuracy_score(y_test, predictionsKNN)
    print("\nk=",k," Testing accuracy:", accuracyKNN)
    print('Confusion matrix:\n ', confusion_matrix(y_test, predictionsKNN))
    resultsKNN.append([k, accuracyKNN]) 

#DT
print("\n*******************DT*******************")

# Create decision tree classifiers
classifierDT_gini = DecisionTreeClassifier(criterion='gini', min_samples_split=25, random_state = 31)
classifierDT_infoGain = DecisionTreeClassifier(criterion='entropy', min_samples_split=25, random_state = 31)

# Train decision trees using training dataset
classifierDT_gini.fit(X_train, y_train)
classifierDT_infoGain.fit(X_train, y_train)

# Perform prediction on testing data (test how accurate the patterns they learnt are)
predictionsDT_gini = classifierDT_gini.predict(X_test)
predictionsDT_infoGain = classifierDT_infoGain.predict(X_test)

#Classifiers accuracies and Confusion matrices
accuracyDT_gini = accuracy_score(y_test, predictionsDT_gini)
confusionDT_gini = confusion_matrix(y_test, predictionsDT_gini)
accuracyDT_infoGain = accuracy_score(y_test, predictionsDT_infoGain)
confusionDT_infoGain = confusion_matrix(y_test, predictionsDT_infoGain)

#Print stats
print('\nDecision Tree with Gini criterion\n')
print('Testing accuracy: ', accuracyDT_gini)
print('Confusion matrix:\n', confusionDT_gini)

print('\nDecision Tree with Information gain criterion\n')
print('Testing accuracy: ', accuracyDT_infoGain)
print('Confusion matrix:\n', confusionDT_infoGain)


#RF
print("\n*******************RF*******************")

# Create Random Forest classifiers
classifierRF_gini = RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_split=25)
classifierRF_infoGain = RandomForestClassifier(criterion='entropy', n_estimators=100, min_samples_split=25)

# Train the classifiers
classifierRF_gini.fit(X_train, y_train)
classifierRF_infoGain.fit(X_train, y_train)

# Perform prediction on testing data (test how accurate the patterns they learnt are)
predictionsRF_gini = classifierRF_gini.predict(X_test)
predictionsRF_infoGain = classifierRF_infoGain.predict(X_test)

#Classifiers accuracies and Confusion matrices
accuracyRF_gini = accuracy_score(y_test, predictionsRF_gini)
confusionRF_gini = confusion_matrix(y_test, predictionsRF_gini)
accuracyRF_infoGain = accuracy_score(y_test, predictionsRF_infoGain)
confusionRF_infoGain = confusion_matrix(y_test, predictionsRF_infoGain)

#Print stats
print('\nRandom Forest with Gini criterion\n')
print('Testing accuracy: ', accuracyRF_gini)
print('Confusion matrix:\n', confusionRF_gini)

print('\nRandom Forest with Information gain criterion\n')
print('Testing accuracy: ', accuracyRF_infoGain)
print('Confusion matrix:\n', confusionRF_infoGain)

# Comparing results 
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# KNN
k_values = [item[0] for item in resultsKNN]
knn_accuracies = [item[1] for item in resultsKNN]
axs[0].plot(k_values, knn_accuracies, marker='o', linestyle='-', color='blue')
axs[0].set_title('KNN Accuracy vs k')
axs[0].set_xlabel('k')
axs[0].set_ylabel('Accuracy')
axs[0].set_xticks(k_values)
axs[0].grid(True)
axs[0].set_ylim(0, 1.05)

# DT
dt_criteria = ['Gini', 'Information Gain']
dt_accuracies = [accuracyDT_gini, accuracyDT_infoGain]
axs[1].bar(dt_criteria, dt_accuracies, color=['green', 'orange'])
axs[1].set_title('DT: Gini vs Info Gain')
axs[1].set_ylabel('Accuracy')
axs[1].grid(axis='y')
axs[1].set_ylim(0, 1.05)

# RF
rf_criteria = ['Gini', 'Information Gain']
rf_accuracies = [accuracyRF_gini, accuracyRF_infoGain]
axs[2].bar(rf_criteria, rf_accuracies, color=['purple', 'red'])
axs[2].set_title('RF: Gini vs Info Gain')
axs[2].set_ylabel('Accuracy')
axs[2].grid(axis='y')
axs[2].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
