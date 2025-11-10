import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler

# Import data from 'DiabetesData.csv'
diabetes_df = pd.read_csv('DiabetesData.csv')

# convert categorical data to int representations of unique categories
for col in diabetes_df.columns[:-1]: # all columns but last cause last is already serialized. 
    labels, uniques = pd.factorize(diabetes_df[col])
    diabetes_df[col] = labels

# Labels
y = diabetes_df['diagnosed_diabetes']
# features after removing label column
X = diabetes_df.drop(columns='diagnosed_diabetes')

def attributesSelecta(df): # Helper that randomly selects 3 and 7 attributes from a dataframe. Returns 2 lists of attributes (l=3 and l=7)

    feature_names = list(df.columns)  # all available features
    features3 = random.sample(feature_names, 3)
    features7 = random.sample(feature_names, 7)
    return features3, features7

# Store max accuracy and best feature sets for each model
max_accuracies = {"kNN 3": 0, "kNN 7": 0, "MNB 3": 0, "MNB 7": 0, "GNB 3": 0, "GNB 7": 0, "LR 3": 0, "LR 7": 0, "SVM 3": 0, "SVM 7": 0}
best_features = {"kNN 3": None, "kNN 7": None, "MNB 3": None, "MNB 7": None, "GNB 3": None, "GNB 7": None, "LR 3": None, "LR 7": None, "SVM 3": None, "SVM 7": None}

# Store all accuracies for plotting histograms
accuracies_m3 = {"kNN": [], "MNB": [], "GNB": [], "LR": [], "SVM": []}
accuracies_m7 = {"kNN": [], "MNB": [], "GNB": [], "LR": [], "SVM": []}

for i in range(100):
    
    features3, features7 = attributesSelecta(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # Split dataset into training dataset and testing dataset (25% testing)

    # Subset for the feature sets
    X_train3 = X_train[features3]
    X_test3 = X_test[features3]
    X_train7 = X_train[features7]
    X_test7 = X_test[features7]

    # kNN with 3 features
    classifierKNN3 = KNeighborsClassifier(n_neighbors=11)            # Create 
    classifierKNN3.fit(X_train3, y_train)                            # Train
    predictionsKNN3 = classifierKNN3.predict(X_test3)                # Test
    acc_kNN3 = accuracy_score(y_test, predictionsKNN3)               # Predictions
    accuracies_m3["kNN"].append(acc_kNN3)

    if acc_kNN3 > max_accuracies["kNN 3"]:
        max_accuracies["kNN 3"] = acc_kNN3
        best_features["kNN 3"] = features3

    # kNN with 7 features
    classifierKNN7 = KNeighborsClassifier(n_neighbors=11)            # Create 
    classifierKNN7.fit(X_train7, y_train)                            # Train
    predictionsKNN7 = classifierKNN7.predict(X_test7)                # Test
    acc_kNN7 = accuracy_score(y_test, predictionsKNN7)               # Predictions
    accuracies_m7["kNN"].append(acc_kNN7)

    if acc_kNN7 > max_accuracies["kNN 7"]:
        max_accuracies["kNN 7"] = acc_kNN7
        best_features["kNN 7"] = features7

    # Multinomial NB with 3 features
    nb_multi3 = MultinomialNB()                                   # Create
    nb_multi3.fit(X_train3, y_train)                              # Train
    multi3_preds = nb_multi3.predict(X_test3)                     # Test
    acc_MNB3 = accuracy_score(y_test, multi3_preds)               # Predictions
    accuracies_m3["MNB"].append(acc_MNB3)

    if acc_MNB3 > max_accuracies["MNB 3"]:
        max_accuracies["MNB 3"] = acc_MNB3
        best_features["MNB 3"] = features3

    # Multinomial NB with 7 features
    nb_multi7 = MultinomialNB()                                   # Create
    nb_multi7.fit(X_train7, y_train)                              # Train
    multi7_preds = nb_multi7.predict(X_test7)                     # Test
    acc_MNB7 = accuracy_score(y_test, multi7_preds)               # Predictions
    accuracies_m7["MNB"].append(acc_MNB7)

    if acc_MNB7 > max_accuracies["MNB 7"]:
        max_accuracies["MNB 7"] = acc_MNB7
        best_features["MNB 7"] = features7

    # Gaussian NB with 3 features 
    nb_gauss3 = GaussianNB()                                      # Create
    nb_gauss3.fit(X_train3, y_train)                              # Train
    gauss3_preds = nb_gauss3.predict(X_test3)                     # Test
    acc_GNB3 = accuracy_score(y_test, gauss3_preds)               # Predictions
    accuracies_m3["GNB"].append(acc_GNB3)

    if acc_GNB3 > max_accuracies["GNB 3"]:
        max_accuracies["GNB 3"] = acc_GNB3
        best_features["GNB 3"] = features3

    # Gaussian NB with 7 features 
    nb_gauss7 = GaussianNB()                                      # Create
    nb_gauss7.fit(X_train7, y_train)                              # Train
    gauss7_preds = nb_gauss7.predict(X_test7)                     # Test
    acc_GNB7 = accuracy_score(y_test, gauss7_preds)               # Predictions
    accuracies_m7["GNB"].append(acc_GNB7)

    if acc_GNB7 > max_accuracies["GNB 7"]:
        max_accuracies["GNB 7"] = acc_GNB7
        best_features["GNB 7"] = features7

    # Logistic Regression classifier
    logmodel3 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000) # Create
    logmodel3.fit(X_train3, y_train)                                            # Train
    log3_preds = logmodel3.predict(X_test3)                                     # Test
    acc_LR3 = accuracy_score(y_test, log3_preds)                                # Predictions
    accuracies_m3["LR"].append(acc_LR3)

    if acc_LR3 > max_accuracies["LR 3"]:
        max_accuracies["LR 3"] = acc_LR3
        best_features["LR 3"] = features3

    # Logistic Regression classifier
    logmodel7 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000) # Create
    logmodel7.fit(X_train7, y_train)                                            # Train
    log7_preds = logmodel7.predict(X_test7)                                     # Test
    acc_LR7 = accuracy_score(y_test, log7_preds)                                # Predictions
    accuracies_m7["LR"].append(acc_LR7)

    if acc_LR7 > max_accuracies["LR 7"]:
        max_accuracies["LR 7"] = acc_LR7
        best_features["LR 7"] = features7

    # SVM with 3 features
    scaler3 = MinMaxScaler()
    X_train3_scaled = scaler3.fit_transform(X_train3)
    X_test3_scaled = scaler3.transform(X_test3)
    
    svm_model = LinearSVC(C=50, max_iter=2000)              # Create
    svm_model.fit(X_train3_scaled, y_train)                 # Train
    preds3 = svm_model.predict(X_test3_scaled)              # Test
    acc_SVM3 = accuracy_score(y_test, preds3)               # Predictions
    accuracies_m3["SVM"].append(acc_SVM3)

    if acc_SVM3 > max_accuracies["SVM 3"]:
        max_accuracies["SVM 3"] = acc_SVM3
        best_features["SVM 3"] = features3

    # SVM with 7 features
    scaler7 = MinMaxScaler()
    X_train7_scaled = scaler7.fit_transform(X_train7)
    X_test7_scaled = scaler7.transform(X_test7)
    
    svm_model = LinearSVC(C=50, max_iter=2000)              # Create
    svm_model.fit(X_train7_scaled, y_train)                 # Train
    preds7 = svm_model.predict(X_test7_scaled)              # Test
    acc_SVM7 = accuracy_score(y_test, preds7)               # Predictions
    accuracies_m7["SVM"].append(acc_SVM7)
    
    if acc_SVM7 > max_accuracies["SVM 7"]:
        max_accuracies["SVM 7"] = acc_SVM7
        best_features["SVM 7"] = features7

# Plot the histograms
plt.figure(figsize=(12, 6))

# m=3 accuracies
plt.subplot(1, 2, 1)
plt.hist(accuracies_m3["kNN"], bins=10, alpha=0.7, label="kNN")
plt.hist(accuracies_m3["MNB"], bins=10, alpha=0.7, label="MNB")
plt.hist(accuracies_m3["GNB"], bins=10, alpha=0.7, label="GNB")
plt.hist(accuracies_m3["LR"], bins=10, alpha=0.7, label="LR")
plt.hist(accuracies_m3["SVM"], bins=10, alpha=0.7, label="SVM")
plt.title("Accuracy Distribution for m=3")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()

# m=7 accuracies
plt.subplot(1, 2, 2)
plt.hist(accuracies_m7["kNN"], bins=10, alpha=0.7, label="kNN")
plt.hist(accuracies_m7["MNB"], bins=10, alpha=0.7, label="MNB")
plt.hist(accuracies_m7["GNB"], bins=10, alpha=0.7, label="GNB")
plt.hist(accuracies_m7["LR"], bins=10, alpha=0.7, label="LR")
plt.hist(accuracies_m7["SVM"], bins=10, alpha=0.7, label="SVM")
plt.title("Accuracy Distribution for m=7")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()
