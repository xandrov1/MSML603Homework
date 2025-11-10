import pandas as pd

# Create a sample DataFrame
data = {'X1': [4, 4, 5, 5, 3, 3, 3, 4],
        'X2': [3, 2, 2, 1, 1, 0, -1, 0],
        'Label': ['Red', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue', 'Blue']}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('my_data.csv', index=False)