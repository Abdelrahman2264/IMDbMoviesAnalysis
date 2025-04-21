import pandas as pd
from pymongo import MongoClient
import json
from pathlib import Path

# 1. Read CSV File
csv_path = "D:\\Project Data Science Tools\\cleaned_movies_data.csv"
excel_data = pd.read_csv(csv_path)

# 2. Convert to Dictionary
data = excel_data.to_dict(orient="records")

# 3. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["MoviesDb"]
collection = db["mycollection"]

# 4. Insert Data (and clear existing data if needed)
collection.delete_many({})  # Optional: clears collection before insert
collection.insert_many(data)

# 5. Export to JSON
output_dir = "D:\\Project Data Science Tools\\mongodb_exports"
Path(output_dir).mkdir(exist_ok=True)  # Create directory if doesn't exist

output_path = f"{output_dir}\\movies_export.json"

# Fetch all documents and save as pretty-printed JSON
with open(output_path, 'w', encoding='utf-8') as f:
    cursor = collection.find({}, {'_id': 0})  # Exclude MongoDB _id field
    json.dump(list(cursor), f, indent=2, ensure_ascii=False)

print(f"Successfully:")
print(f"- Inserted {len(data)} records into MongoDB")
print(f"- Exported data to {output_path}")