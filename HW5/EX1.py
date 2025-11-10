import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data set from my_data.csv
percep_df = pd.read_csv('my_data.csv')
print('\nOriginal data\n', percep_df.head())

# a) Determine if the perceptron given by the pair ([1 2]^T , −6.9) correctly classifies the given samples
# Hyperplane equation is: w^T * x + b = 0 
# Basically plug in x1 and x2 from my_data.csv into the equation:
# 1 * x1 + 2 * x2 -6.9 = 0
# take the sign of each result (make predictions) to classify it as either of the labels (-1, +1 or 0)
# Compare predictions with correct labels

# Vector of Xs to plug into equation
X_vec = percep_df.drop(columns='Label')
# Vector of correct labels
y_vec = percep_df["Label"]
# Vector of predictions to store the sign of X_vec xs plugged in the perceptron equation
predictions = []

# Use vectorized operations to calculate the result for all rows at once; result contains the raw perceptron output per row of dataset
results = 1 * X_vec['X1'] + 2 * X_vec['X2'] - 6.9

# Apply the sign function and store the predictions; predictions contains class predictions
for result in results: 
    #prediction = np.sign(result)
    if(result > 0):
        predictions.append('Red')
    elif(result < 0):
        predictions.append('Blue')
    else:
        predictions.append("Hyperplane was wrong about me.")

# Comparing
print('\nCorrect predctions:\n', y_vec)
print('\nPredictions given the pair ([1 2]^T , −6.9):', predictions)

# Actual comparision 
correct = (y_vec==predictions)

# Final verdict
if(correct.all()):
    print("\nThe perceptron given by the pair ([1 2]^T , −6.9) correctly classifies the given samples\n")
else:
    print("\nThe perceptron given by the pair ([1 2]^T , −6.9) incorrectly classifies the given samples\n")

# b) Find a margin perceptron and specify the hyperplane given by (w, b). Verify that it is a margin perceptron. 
# [Hint: You can guess a margin perceptron and verify that it is a margin perceptron.]
# Verify it's a good margin perceptron by:
# Checking that max(0, 1 - y^k(w^T·x^k + b)) = 0 for all k where:
# y^k is the label value (Red = +1, Blue = -1) 
# k is used to specify the row index of dataset. Each row has a pair and a label associated with that pair. 
# X is a pair (x1, x2)
# w^T * x^k is the inner product <w,x>
# Example: in my_data.csv row one is: X = (x1 = 4, x2 = 3) and Label is red = +1
# Plug it into formula:
# max(0, 1 - (+1)((w1 * 4 + w2 * 3) + b)) = 0
# if the max(...) really equals 0 then w1,w2 and b are it for that X. 
# The right perceptron satisifies this condition for all Xs (rows) of the dataset

# Function to solve max(....) for each X in dataset, after we feed w1,w2 and b to function
def check_margin_perceptron(w, b, X_vec, y_vec):

    # Convert labels to +1/-1 (make a vector where label = Red is stored as +1 and Blue is -1)
    y_binary = np.where(y_vec == 'Red', 1, -1)

    # for all X pairs plug them in max(0, 1 - (+1)((w1 * 4 + w2 * 3) + b)) = 0
    for i in range(len(X_vec)):
        temp = 1 - y_binary[i] * (np.dot(w, X_vec.iloc[i]) + b)
        margin_loss = max(0, temp)

        # max wasn't 0 in at least one of the rows
        if(margin_loss != 0):
            print('Invalid margin perceptron')
            # Interrupt and return
            return False
        
        # Found margin perceptron
        print('Valid margin perceptron')
        return True

# Test ws and bs
w = [1,2]
b = -6
check_margin_perceptron(w, b, X_vec, y_vec)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Generate x1 values for the plot (range for x1 based on the data's min and max)
x1_range = np.linspace(X_vec['X1'].min() - 1, X_vec['X1'].max() + 1, 100)

# Solve for x2 using the hyperplane equation: x2 = (-w1*x1 - b) / w2
x2_hyperplane = (-w[0] * x1_range - b) / w[1]

# Plot Red points
ax.scatter(X_vec[y_vec == 'Red']['X1'], X_vec[y_vec == 'Red']['X2'], color='red', label='Red', s=100, edgecolors='k')
# Plot Blue points
ax.scatter(X_vec[y_vec == 'Blue']['X1'], X_vec[y_vec == 'Blue']['X2'], color='blue', label='Blue', s=100, edgecolors='k')
# Plot the hyperplane
ax.plot(x1_range, x2_hyperplane, color='green', linestyle='--', label=f'Hyperplane (w = {w}, b = {b})')

# Add labels and title
ax.set_title('Scatter Plot of Data (Red and Blue Classes), with Hyperplane from a)')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# Show the legend
ax.legend()
# Show the grid
ax.grid(True)
# Show the plot
plt.show()

# c) Compute the margin for the margin perceptron. I'll be using 2/||w||

print(f'\nThe margin, using the formula (i) 2/||w|| is: {2/np.linalg.norm(w)}')


