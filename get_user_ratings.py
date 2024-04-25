# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:23:17 2024

@author: Mingk
"""

import os
import pandas as pd
from tqdm import tqdm

# Define the directory containing the CSV files, "/path/to/csv_files" 
directory = "C:/Users/Mingk/Desktop/recsys/Epinions/dataset/reviews"

# Define the directory where the new CSV file will be saved
output_directory = "C:/Users/Mingk/Desktop/recsys/Epinions"

# Initialize an empty list to store data frames
dfs = []

# Iterate over the CSV files in the directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".csv"):
        # Extract username from the filename
        username, _ = filename.rsplit("_", 1)
        
        #product_category = product_category.replace(".csv", "")
        
        try:
        # Read the CSV file into a data frame
            df = pd.read_csv(os.path.join(directory, filename))
        except pd.errors.EmptyDataError:
            continue
        
        # Add header, rename columns
        df.columns = [0,1,2,3,4,5,6,7,8,9]
        df.rename(columns={0:"time" , 3:"item_name", 5: "sub_category", 7: "category", 8: "rating"}, inplace=True)
        df["user_name"] = username
        df.drop(columns=[1, 2, 4, 6, 9])
        
        # Append the modified data frame to the list
        dfs.append(df)
        
# concatenate data frames
final_df = pd.concat(dfs, ignore_index=True)

# Reorder columns
final_df = final_df[["user_name", "time" , "item_name", "sub_category", "category", "rating"]]
#print(set(final_df["user_name"]))

# Save the final data frame to a new CSV file
output_file = os.path.join(output_directory, "combined_user_reviews.csv")
user_name_output = os.path.join(output_directory, "username_from_ratings.txt")
user_name_ = set(final_df["user_name"])
with open(user_name_output, "w") as f:
    for item in user_name_:
        f.write(item + "\n")
final_df.to_csv(output_file, index=False)

print("CSV file with renamed columns saved to:", output_file)
