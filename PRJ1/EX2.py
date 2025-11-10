# Naive Bayes, k-Nearest Neighbors, Logistic Regression, and Support Vector Machine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import data from 'table1.csv'
diabetes_df = pd.read_csv('DiabetesData.csv')

# Print first rows 
print('\nOriginal data')
print(diabetes_df.head(),'\n')

# convert categorical data to int representations of unique categories
for col in diabetes_df.columns[:-1]: # all columns but last cause last is already serialized. 
    labels, uniques = pd.factorize(diabetes_df[col])
    diabetes_df[col] = labels

print('After conversion to integer values:')   
print(diabetes_df.head(), '\n')

# Labels
y = diabetes_df['diagnosed_diabetes']
# features after removing label column
X = diabetes_df.drop(columns='diagnosed_diabetes')

# (i) kNN but vary the value of k from 3 to 20 with an increment of one and plot the average test accuracy
k_values = range(3,21)
knnAvgAccuracies = [] # contains average accuracy per each k

for k in k_values:
    accuracyKNNSum = 0 # Sum of all accuracies for this k
    for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # Split dataset into training dataset and testing dataset (25% testing)
        classifierKNN = KNeighborsClassifier(n_neighbors=k)                       # Create 
        classifierKNN.fit(X_train, y_train)                                       # Train
        predictionsKNN = classifierKNN.predict(X_test)                            # Test
        accuracyKNNSum += accuracy_score(y_test, predictionsKNN)                  # Accuracy added to sum

    print(f"Finished training 20 kNNs with k = {k}")
    knnAvgAccuracies.append(accuracyKNNSum/20) # Append average accuracy of this k

plt.plot(k_values, knnAvgAccuracies, marker = 'o')
plt.title("Average accuracy of 20 kNNs per k")
plt.xlabel("k (Number of neighbors used)")
plt.ylabel("Average accuracy over 20 kNN")
plt.grid("True")
plt.show()
        
# (ii) Naive bayes classifiers (Multinomial and Gaussian)
accuracyMultiSum = accuracyGaussSum = 0
for i in range(20):

    # Split dataset into training dataset and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Multinomial Naive Bayes assumes categories follow multinomial distribution
    nb_multi = MultinomialNB()                              # Create
    nb_multi.fit(X_train, y_train)                          # Train
    multi_preds = nb_multi.predict(X_test)                  # Test
    accuracyMultiSum += accuracy_score(y_test, multi_preds) # Accuracy added to sum

    # Gaussian Naive Bayes assumes categories follow normal distribution 
    nb_gauss = GaussianNB()                                 # Create
    nb_gauss.fit(X_train, y_train)                          # Train
    gauss_preds = nb_gauss.predict(X_test)                  # Test
    accuracyGaussSum += accuracy_score(y_test, gauss_preds) # Accuracy added to sum

# Actual averages calculation
accuracyMulti = accuracyMultiSum/20
accuracyGauss = accuracyGaussSum/20
print(f'\nAverage Accuracy Multinomial Naive Bayes classifier = {accuracyMulti:.4f}\n')
print(f'Average Accuracy Gaussian Naive Bayes classifier = {accuracyGauss:.4f}\n')


# (iii) Logistic Regression
accuracyLRSum = 0
for i in range(20):

    # Split dataset into training dataset and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Logistic Regression classifier
    logmodel = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000) # Create
    logmodel.fit(X_train, y_train)                                             # Train
    log_preds = logmodel.predict(X_test)                                       # Test
    accuracyLRSum += accuracy_score(y_test, log_preds)                         # Accuracy added to sum
    
# Actual averages calculation
accuracyLR = accuracyLRSum/20
print(f'Average Accuracy with Logistic regression classifier = {accuracyLR:.4f}\n')

# (iv) SVM classifier: NB. Data features have different scales.
# Standardizing is mandatory when features use different units of measurement. 
# Added scaling too cause maybe it'd work better than standardizing. No significant changes
# Initially used svc with linear kernel. Because it was the fastest way to classify IMO, on the same dataset, linear hyperplane should be faster to compute than other types. Had a 0.5 accuracy score
# I changed to linearSVC which is basically an SVC with a linear kernel, get a better accuracy: 0.6
# I see the accuracies are similar so maybe linear decision boundary is just not it for this data set
# I tried SVC with kernel: rbf, poly, sigmoid, and linear. The all get worse than 0.6
# I switched back to LinearSVC because it's faster, it got no warnings ever, and had the best average accuracy, now I'm playing around with C and max_iter. 
# C: 0: How much misclassification is allowed in order to get a larger margin. 1 means find the greatest possible margin keeping misclassification low. 
# Higher Cs mean better classifications but margin may be smaller. May overfit data too 
# I tried higher Cs like 10000 and lower Cs like 0.1 and found no significant changes, average accuracy is always around 0.6 
# max_iter: the maximum number of iterations the solver will have to find the optimal separating hyperplane. I left it at 2000 but i did try numbers like 200000 and still got 0.6 average accuracy
accuracySVMSum = 0
for i in range(20):

    # Split dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Standardize scales on both training and testing features
    #scaler = StandardScaler() 
    scaler = MinMaxScaler()     
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = LinearSVC(C=10000, max_iter=2000)   # Create
    svm_model.fit(X_train_scaled, y_train)          # Train
    preds = svm_model.predict(X_test_scaled)        # Test
    accuracySVMSum += accuracy_score(y_test, preds) # Add test accuracy to sum

# Compute average test accuracy
accuracySVM = accuracySVMSum / 20
print(f'Average Accuracy with linear vector support machine= {accuracySVM:.4f}')