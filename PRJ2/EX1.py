import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

# pollution data frame import
pollutionDF = pd.read_csv('pollution_dataset.csv')
# Print first rows 
print('\nOriginal data: \n', pollutionDF.head())

# Convert to numpy arrays for sklearn
feature_names = pollutionDF.columns.drop('Air Quality')
X = pollutionDF[feature_names].values
y = pollutionDF['Air Quality'].values

# Replace negatives with 0
X = np.clip(X, 0, None)

print("\nFeature names:\n", feature_names)
print("\nFeatures values (first 5 rows):\n", X[:5])
print("\nLabels (first 5 rows):\n", y[:5])

# Creating two dictionaries to save accuracies and plotting
svm_results = {
    'Feature Selection': [],
    'Feature Importance': [],
    'PCA': [],
    'LDA': [],
    'Baseline': 0  # single value
}

nb_results = {
    'Feature Selection': [],
    'Feature Importance': [],
    'PCA': [],
    'LDA': [],
    'Baseline': 0  # single value
}

# Initialize baseline accuracies
svm_accuracy = nb_accuracy = 0

# Baseline: train models with all 9 features first. Using no dimensionality technique
for i in range(50):

    # Split and train on same split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i) # Use same random state (same rows/cities)

    # SVM
    svm_model = LinearSVC(C=1.0)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test) # Test SVM model
    svm_accuracy += accuracy_score(y_test, svm_preds) # Sum accuracy scores

    #NB
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test) # Test NB model
    nb_accuracy += accuracy_score(y_test, nb_preds) # Sum accuracy scores
    
svm_results['Baseline'] = svm_accuracy/50 # Store basline average accuracy for SVM
nb_results['Baseline'] = nb_accuracy/50 # Store baseline average accuracy for NB
    

# Task 1:
# Let Kf be the number of features to be selected by feature selection methods. Perform the feature
# selection using (i) univariate feature selection method and (ii) feature importance scores for Kf = 1, 2, 3.
# For computing feature importance scores, use a random forest classifier. List the features selected by these methods for Kf = 1, 2, 3.

print('\n Task 1 (i)')
X_featureSelection = [] # To store k top features
for i in range(3):

    # Univariate feature selection
    selector = SelectKBest(chi2, k = i+1) # Initialize selector with chi2 function to give scores to attributes and select k of them
    # X_new = selector.fit_transform(X, y) # Apply selector to vector and make new vector of k attributes with best scores; dont really need the transform so
    X_new = selector.fit(X, y) # Just fit the selector (calculate chi2 scores for each feature), don't need the returned vector 

    X_new_features_mask = selector.get_support() # Create a vector tagging (true/false) k attributes with best scores as true
    top_features = feature_names[X_new_features_mask] # Temporarily save k top features
    X_featureSelection.append(top_features) # Save k top features for later
    top_scores = selector.scores_[X_new_features_mask] # Temporarily save k top scores for selected features

    # To sort scores in descending so print keeps best as top score
    sorted_idx = np.argsort(top_scores)[::-1]

    print(f'\nKf = {i+1} features using univariate feature selection:')
    for feat, score in zip(top_features[sorted_idx], top_scores[sorted_idx]): # print attribute and score as tuple for each k
        print(f'  {feat}: {score:.4f}')

print('\n Task 1 (ii)')
clf = RandomForestClassifier(n_estimators=100, random_state=42) # Initialize random forest with 100 trees
clf = clf.fit(X, y) # Train the forest on data
tree_importance_sorted_idx = np.argsort(clf.feature_importances_) # Get importance scores of each attribute and sort it in ascending order


X_featureImportance = [] # To store k top features
for i in range(3):
    # Feature Importance
    top_indices = tree_importance_sorted_idx[-(i+1):][::-1] # Get indices of attributes with top scores. (was in ascending order so flip it ([::-1]) for print statement to keep it in descending order)
    top_features = feature_names[top_indices] # Save feature/s that correspond to the index/es
    X_featureImportance.append(top_features) # Save k top features for later
    top_scores = clf.feature_importances_[top_indices] # Save best scores
    
    print(f'\nKf = {i+1} features with highest scores:')
    for feat, score in zip(top_features, top_scores): # print attribute and score as tuple for each k
        print(f'  {feat}: {score:.4f}')


# Task 2: Let Ks be the dimension of the subspace onto which you will project the feature vectors using PCA (i) and LDA (ii). 
# Perform the PCA and find the new dataset comprising the new feature values for Ks = 1, 2, 3. Repeat it using the LDA. 

# Standardize features in X
X = StandardScaler().fit_transform(X)
X_pca = []
X_lda = []

# Task 2 (i & ii)
for i in range(3):

    # (i)
    pca = PCA(n_components=i+1) # Create new PCA for Ks
    pca.fit(X)
    X_p = pca.transform(X)
    X_pca.append(X_p) # Save feature derived with PCA

    # (ii)
    lda = LinearDiscriminantAnalysis(n_components=i+1) # Create new LDA for Ks
    lda.fit(X, y)
    X_l = lda.transform(X)
    X_lda.append(X_l) # Save feature derived with LDA


# Task 3: For Kf = 1, 2, 3, use the datasets from Task 1, which consist of a subset of the features selected via
# the two feature selection methods and perform multi-class classification using (i) SVM and (ii) naive Bayes.
# Assume Gaussian distribution for the naive Bayes classifier. For training and testing the classifiers, set aside
# 25 percent of the datasets for testing (and 75 percent for training). Compute the average test accuracy from 50 randomly split training and testing datasets  

print('\n Task 3 (i)\n')
for i in range(3):
    # Feature Selection 
    selected_features_fs = X_featureSelection[i]
    X_fs = pollutionDF[selected_features_fs].values  # Extract columns
    svm_accuracy_fs = 0

    # Feature Importance 
    selected_features_fi = X_featureImportance[i]
    X_fi = pollutionDF[selected_features_fi].values  # Extract columns
    svm_accuracy_fi = 0

    for j in range(50):
        # Feature Selection
        X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size=0.25, random_state=j)
        svm_model_fs = LinearSVC(C=1.0) # SVM model using feature selection for attributes
        svm_model_fs.fit(X_train, y_train)
        svm_preds_fs = svm_model_fs.predict(X_test) # Test SVM model
        svm_accuracy_fs += accuracy_score(y_test, svm_preds_fs) # Sum accuracy scores

        # Feature Importance
        X_train, X_test, y_train, y_test = train_test_split(X_fi, y, test_size=0.25, random_state=j)
        svm_model_fi = LinearSVC(C=1.0) # SVM model using feature importance for attributes
        svm_model_fi.fit(X_train, y_train)
        svm_preds_fi = svm_model_fi.predict(X_test) # Test SVM model
        svm_accuracy_fi += accuracy_score(y_test, svm_preds_fi) # Sum accuracy scores
    
    svm_results['Feature Selection'].append(svm_accuracy_fs/50) # Average accuracy of SVM models with feature selection attributes
    svm_results['Feature Importance'].append(svm_accuracy_fi/50) # Average accuracy of SVM models with feature importance attributes
    print(f'Kf={i+1}:\nFeature Selection, Avg SVM Accuracy: {svm_results['Feature Selection'][i]:.4f}\nFeature Importance, Avg SVM Accuracy: {svm_results['Feature Importance'][i]:.4f}')


print('\n Task 3 (ii)\n')
for i in range(3):
    # Feature Selection 
    selected_features_fs = X_featureSelection[i]
    X_fs = pollutionDF[selected_features_fs].values  # Extract columns
    nb_accuracy_fs = 0

    # Feature Importance 
    selected_features_fi = X_featureImportance[i]
    X_fi = pollutionDF[selected_features_fi].values  # Extract columns
    nb_accuracy_fi = 0

    for j in range(50):
        # Feature Selection
        X_train, X_test, y_train, y_test = train_test_split(X_fs, y, test_size=0.25, random_state=j)
        nb_model_fs = GaussianNB() # NB model using feature selection for attributes
        nb_model_fs.fit(X_train, y_train)
        nb_preds_fs = nb_model_fs.predict(X_test) # Test NB model
        nb_accuracy_fs += accuracy_score(y_test, nb_preds_fs) # Sum accuracy scores

        # Feature Importance
        X_train, X_test, y_train, y_test = train_test_split(X_fi, y, test_size=0.25, random_state=j)
        nb_model_fi = GaussianNB() # NB model using feature importance for attributes
        nb_model_fi.fit(X_train, y_train)
        nb_preds_fi = nb_model_fi.predict(X_test) # Test NB model
        nb_accuracy_fi += accuracy_score(y_test, nb_preds_fi) # Sum accuracy scores

    nb_results['Feature Selection'].append(nb_accuracy_fs/50) # Average accuracy of SVM models with feature selection attributes
    nb_results['Feature Importance'].append(nb_accuracy_fi/50) # Average accuracy of SVM models with feature importance attributes
    print(f'Kf={i+1}:\nFeature Selection, Avg NB Accuracy: {nb_results['Feature Selection'][i]:.4f}\nFeature Importance, Avg NB Accuracy: {nb_results['Feature Importance'][i]:.4f}')


# Task 4: For Ks = 1, 2, 3, use the new datasets generated via PCA and LDA from Task 2 and perform multi-class 
# classification using (i) SVM and (ii) naive Bayes with Gaussian distribution. 
# Use 25 percent of the datasets for testing (and 75 percent for training). 
# Compute the average test accuracy from 50 randomly split training and testing datasets

print('\n Task 4 (i)\n')
for i in range(3):
    # PCA
    X_p = X_pca[i]  # Get PCA-transformed data for Ks = i+1
    svm_accuracy_pca = 0
    
    # LDA
    X_l = X_lda[i]  # Get LDA-transformed data for Ks = i+1
    svm_accuracy_lda = 0
    
    for j in range(50):
        # PCA
        X_train, X_test, y_train, y_test = train_test_split(X_p, y, test_size=0.25, random_state=j)
        svm_model_pca = LinearSVC(C=1.0) # SVM model using PCA for attributes
        svm_model_pca.fit(X_train, y_train)
        svm_preds_pca = svm_model_pca.predict(X_test) # Test SVM model
        svm_accuracy_pca += accuracy_score(y_test, svm_preds_pca) # Sum accuracies
        
        # LDA
        X_train, X_test, y_train, y_test = train_test_split(X_l, y, test_size=0.25, random_state=j)
        svm_model_lda = LinearSVC(C=1.0) # SVM model using LDA for attributes
        svm_model_lda.fit(X_train, y_train)
        svm_preds_lda = svm_model_lda.predict(X_test) # Test SVM model
        svm_accuracy_lda += accuracy_score(y_test, svm_preds_lda) # Sum accuracies

    svm_results['PCA'].append(svm_accuracy_pca/50) # Average accuracy of SVM models with PCA attributes
    svm_results['LDA'].append(svm_accuracy_lda/50) # Average accuracy of SVM models with LDA attributes
    print(f'Ks={i+1}:\nPCA, Avg SVM Accuracy: {svm_results['PCA'][i]:.4f}\nLDA, Avg SVM Accuracy: {svm_results['LDA'][i]:.4f}')


print('\n Task 4 (ii)\n')
for i in range(3):
    # PCA
    X_p = X_pca[i]
    nb_accuracy_pca = 0
    
    # LDA
    X_l = X_lda[i]
    nb_accuracy_lda = 0
    
    for j in range(50):
        # PCA
        X_train, X_test, y_train, y_test = train_test_split(X_p, y, test_size=0.25, random_state=j)
        nb_model_pca = GaussianNB() # NB model using PCA for attributes
        nb_model_pca.fit(X_train, y_train)
        nb_preds_pca = nb_model_pca.predict(X_test) # Test NB model
        nb_accuracy_pca += accuracy_score(y_test, nb_preds_pca) # Sum accuracies
        
        # LDA
        X_train, X_test, y_train, y_test = train_test_split(X_l, y, test_size=0.25, random_state=j)
        nb_model_lda = GaussianNB() # NB model using LDA for attributes
        nb_model_lda.fit(X_train, y_train)
        nb_preds_lda = nb_model_lda.predict(X_test) # Test NB model
        nb_accuracy_lda += accuracy_score(y_test, nb_preds_lda)  # Sum accuracies
    
    nb_results['PCA'].append(nb_accuracy_pca/50) # Average accuracy of NB models with PCA attributes
    nb_results['LDA'].append(nb_accuracy_lda/50) # Average accuracy of NB models with LDA attributes
    print(f'Ks={i+1}:\n PCA, Avg NB Accuracy: {nb_results['PCA'][i]:.4f}\n LDA, Avg NB Accuracy: {nb_results['LDA'][i]:.4f}')

# Plotting
# A comparison across all methods for each classifier
# so one plot showing SVM accuracy for all 4 methods (Feature Selection, Feature Importance, PCA, LDA) on the same graph, and another for NB.
# Add one more for baseline (Using all 9 features) for SVM and NB

K_vals = [1,2,3]
plt.figure(figsize=(14, 6))  # wider for side-by-side, not as tall

# First subplot - SVM
plt.subplot(1, 2, 1)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.plot(K_vals, svm_results['Feature Selection'], label='Feature Selection', marker='o', color='blue')
plt.plot(K_vals, svm_results['Feature Importance'], label='Feature Importance', marker='o', color='green')
plt.plot(K_vals, svm_results['PCA'], label='PCA', marker='o', color='red')
plt.plot(K_vals, svm_results['LDA'], label='LDA', marker='o', color='purple')
plt.axhline(y=svm_results['Baseline'], label='Baseline (9 features)', linestyle='--', color='black')
plt.yticks(np.arange(0.5, 1.01, 0.05))
plt.xlabel("K values")
plt.ylabel("Average accuracy")
plt.title("SVM Performance") 
plt.grid()
plt.legend() 

# Second subplot - NB
plt.subplot(1, 2, 2)
plt.plot(K_vals, nb_results['Feature Selection'], label='Feature Selection', marker='o', color='blue')
plt.plot(K_vals, nb_results['Feature Importance'], label='Feature Importance', marker='o', color='green')
plt.plot(K_vals, nb_results['PCA'], label='PCA', marker='o', color='red')
plt.plot(K_vals, nb_results['LDA'], label='LDA', marker='o', color='purple')
plt.axhline(y=nb_results['Baseline'], label='Baseline (9 features)', linestyle='--', color='black')
plt.yticks(np.arange(0.5, 1.01, 0.05)) 
plt.xlabel("K values")
plt.ylabel("Average accuracy")
plt.title("Naive Bayes Performance") 
plt.grid()
plt.legend()

plt.tight_layout()  # prevents labels from overlapping
plt.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()
