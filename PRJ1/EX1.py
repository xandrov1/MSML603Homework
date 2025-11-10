# Decision Tree and Random Forest Classifiers
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import data set from DiabetesData.csv
heartDisease_pd = pd.read_csv('heart-disease-classification.csv')

#TASK 1 Numerical to Categorical 

# Rename columns: Age, RestingBP, Cholesterol, and MaxHR as Age_cat, RestingBP_cat, Cholesterol_cat, and MaxHR_cat (categorical data)
heartDisease_pd.rename(
    columns=
        {'Age' : 'Age_cat', 
        'RestingBP' : 'RestingBP_cat', 
        'Cholesterol' : 'Cholesterol_cat', 
        'MaxHR' : 'MaxHR_cat'}, 
        inplace= True
)

# Change Age_cat to categorical values
heartDisease_pd['Age_cat'] = pd.cut(
    heartDisease_pd['Age_cat'],
    bins=[-float('inf'), 40, 55, float('inf')], # (negative infinity to 40], (41 to 55], (55 to positive infinity]
    labels=['Young', 'Mid', 'Older'],
    right=True # Right side inclusive bins ']'
)

# Change RestingBP_cat to categorical values
heartDisease_pd['RestingBP_cat'] = pd.cut(
    heartDisease_pd['RestingBP_cat'],
    bins=[-float('inf'), 120, 140, float('inf')], # same as above only numbers and labels changed
    labels=['Low', 'Normal', 'High'],
    right=True # Right side inclusive bins ']'
)

# Change Cholesterol_cat to categorical values
heartDisease_pd['Cholesterol_cat'] = pd.cut(
    heartDisease_pd['Cholesterol_cat'],
    bins=[-float('inf'), 200, float('inf')], # 2 intervals this time. Still right side inclusive
    labels=['Normal', 'High'],
    right=True # Right side inclusive bins ']'
)

# Change MaxHR_cat to categorical values
heartDisease_pd['MaxHR_cat'] = pd.cut(
    heartDisease_pd['MaxHR_cat'],
    bins=[-float('inf'), 140, 165, float('inf')], 
    labels=['Low', 'Normal', 'High'],
    right=True # Right side inclusive bins ']'
)

#Test this works; 
# print(heartDisease_pd.head(20))
# print(heartDisease_pd['Age_cat'].value_counts())
# print(heartDisease_pd['RestingBP_cat'].value_counts())

# Just in case set columns to Pandas categorical type
cat_cols = ['Age_cat', 'RestingBP_cat', 'Cholesterol_cat', 'MaxHR_cat']
heartDisease_pd[cat_cols] = heartDisease_pd[cat_cols].astype('category')

# Save to new csv
heartDisease_pd.to_csv('HeartDiseaseData.csv', index= False)

#TASK 2 DT and Forest

# Convert categorical data to int representation.
for col in heartDisease_pd.columns:
    labels, uniques = pd.factorize(heartDisease_pd[col])
    heartDisease_pd[col] = labels

print("\nAfter conversion:")
print(heartDisease_pd.head(), '\n')

# Feature vector
X = heartDisease_pd.drop(columns='HeartDisease')
# Label vector
y = heartDisease_pd['HeartDisease']

#Sum accuracy scores for DT and RF
dtAccuracySum200 = rfAccuracySum200 = dtAccuracySum10 = rfAccuracySum10 = 0

for i in range(20):

    # Split dataset into training dataset and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Decision tree min_samples_split = 200
    dtree200 = DecisionTreeClassifier(min_samples_split=200)
    dtree200.fit(X_train, y_train)
    # Decision tree min_samples_split = 10
    dtree10 = DecisionTreeClassifier(min_samples_split=10)
    dtree10.fit(X_train, y_train)

    # Perform prediction on testing data 
    predictionsTree200 = dtree200.predict(X_test) # Predictions Decision tree min_samples_split = 200
    predictionsTree10 = dtree10.predict(X_test) # Predictions Decision tree min_samples_split = 10
    #Sum accuracies per type of decision tree
    dtAccuracySum200 += accuracy_score(y_test, predictionsTree200) 
    dtAccuracySum10 += accuracy_score(y_test, predictionsTree10)


    # Random forest min_samples_split = 200
    rForest200 = RandomForestClassifier(n_estimators=250, min_samples_split=200)
    rForest200.fit(X_train, y_train)
    # Random forest min_samples_split = 10
    rForest10 = RandomForestClassifier(n_estimators=250, min_samples_split=10)
    rForest10.fit(X_train, y_train)

    # Perform prediction on testing data 
    predictionsForest200 = rForest200.predict(X_test) # Predictions Random forest min_samples_split = 200
    predictionsForest10 = rForest10.predict(X_test)# Predictions Random forest min_samples_split = 10
    # Sum accuracies per type of random forest
    rfAccuracySum200 += accuracy_score(y_test, predictionsForest200)
    rfAccuracySum10 += accuracy_score(y_test, predictionsForest10)

# Actual averages calculation
dt200AccuracyAvg = dtAccuracySum200/20
dt10AccuracyAvg = dtAccuracySum10/20
rf200AccuracyAvg = rfAccuracySum200/20
rf10AccuracyAvg = rfAccuracySum10/20

# #DT
# print("*************Average accuracy of 20 Decision Trees with min_samples_split = 200*************\n")
# print(f"Accuracy: {dt200AccuracyAvg:.4f}\n")
# print("*************Average accuracy of 20 Decision Trees with min_samples_split = 10*************\n")
# print(f"Accuracy: {dt10AccuracyAvg:.4f}\n")


# #RF
# print("*************Average accuracy of 20 Random Forests with min_samples_split = 200*************\n")
# print(f"Accuracy: {rf200AccuracyAvg:.4f}\n")
# print("*************Average accuracy of 20 Random Forests with min_samples_split = 10*************\n")
# print(f"Accuracy: {rf10AccuracyAvg:.4f}\n")


results = pd.DataFrame({
    'min_samples_split': [200, 10],
    '(20) Decision Trees Avg. Accuracy': [dt200AccuracyAvg, dt10AccuracyAvg],
    '(20) Random Forests Avg. Accuracy': [rf200AccuracyAvg, rf10AccuracyAvg]
})

print(results)