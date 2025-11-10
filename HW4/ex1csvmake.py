import csv

# Data to write to the CSV file
data = [
    ["foggy", "windy", "depart_on_time"],
    ["Yes", "No", "Yes"],
    ["No", "No", "Yes"],
    ["No", "Yes", "No"],
    ["Yes", "No", "Yes"],
    ["Yes", "Yes", "No"],
    ["No", "No", "Yes"],
    ["No", "Yes", "Yes"],
]

# Open (or create) a CSV file in write mode
with open('table1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the rows to the CSV file
    writer.writerows(data)

print("CSV file created successfully!")