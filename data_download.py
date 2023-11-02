import os
import requests
import pandas as pd
from zipfile import ZipFile

# Define the URLs for the accepted loan data
urls_accepted = [
    "https://resources.lendingclub.com/LoanStats3a.csv.zip",
    "https://resources.lendingclub.com/LoanStats3b.csv.zip",
    "https://resources.lendingclub.com/LoanStats3c.csv.zip",
    "https://resources.lendingclub.com/LoanStats3d.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2016Q1.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2016Q2.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2016Q3.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2016Q4.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2017Q1.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2017Q2.csv.zip",
    "https://resources.lendingclub.com/LoanStats_2017Q3.csv.zip"
]

# Create a directory to store data
data_dir = "Data"
os.makedirs(data_dir, exist_ok=True)

accepted_loan_df_list = []

for url in urls_accepted:
    zipfilename = url.split("/")[-1]
    datafilename = zipfilename.replace(".zip", "")

    # Download the ZIP file
    response = requests.get(url)
    with open(os.path.join(data_dir, zipfilename), "wb") as f:
        f.write(response.content)

    # Extract the CSV file from the ZIP
    with ZipFile(os.path.join(data_dir, zipfilename), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(data_dir, datafilename), skiprows=1)
    accepted_loan_df_list.append(df)

# Combine all accepted loan data into one DataFrame
accepted_loan_df = pd.concat(accepted_loan_df_list, axis=0, ignore_index=True)

# Save the combined DataFrame as an RDS file (alternative to RDS in Python)
accepted_loan_df.to_pickle(os.path.join(data_dir, "accepted_loan_data.pkl"))

# Clear the list of DataFrames to save memory
accepted_loan_df_list.clear()

# Define the URLs for the declined loan data
urls_declined = [
    "https://resources.lendingclub.com/RejectStatsA.csv.zip",
    "https://resources.lendingclub.com/RejectStatsB.csv.zip",
    "https://resources.lendingclub.com/RejectStatsD.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2016Q1.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2016Q2.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2016Q3.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2016Q4.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2017Q1.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2017Q2.csv.zip",
    "https://resources.lendingclub.com/RejectStats_2017Q3.csv.zip"
]

declined_loan_df_list = []

for url in urls_declined:
    zipfilename = url.split("/")[-1]
    datafilename = zipfilename.replace(".zip", "")

    # Download the ZIP file
    response = requests.get(url)
    with open(os.path.join(data_dir, zipfilename), "wb") as f:
        f.write(response.content)

    # Extract the CSV file from the ZIP
    with ZipFile(os.path.join(data_dir, zipfilename), "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(data_dir, datafilename))
    declined_loan_df_list.append(df)

# Combine all declined loan data into one DataFrame
declined_loan_df = pd.concat(declined_loan_df_list, axis=0, ignore_index=True)

# Save the combined DataFrame as a pickle file
declined_loan_df.to_pickle(os.path.join(data_dir, "declined_loan_data.pkl"))

# Clean up - Remove ZIP and CSV files
for filename in os.listdir(data_dir):
    if filename.endswith(".zip") or filename.endswith(".csv"):
        os.remove(os.path.join(data_dir, filename))

# Well, we now have combined accepted and declined loan data in DataFrames
