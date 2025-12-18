#!/usr/bin/env python3
"""
Import clinical trials data from SQLite to MongoDB
"""

import sqlite3
import json
import os
from dotenv import load_dotenv

from db_utils import get_mongo_client

load_dotenv()

print("🔄 Starting data import from SQLite to MongoDB...")
print("=" * 60)

# Connect to SQLite
print("📂 Connecting to SQLite database...")
sqlite_conn = sqlite3.connect('clinical_trials.db')
cursor = sqlite_conn.cursor()

# Get total count
cursor.execute("SELECT COUNT(*) FROM trials")
total_count = cursor.fetchone()[0]
print(f"✓ Found {total_count:,} trials in SQLite database")

# Connect to MongoDB
print("\n📂 Connecting to MongoDB Atlas...")
DB_NAME = os.getenv("MONGO_DB_NAME", "clinical_trials")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "studies")
mongo_client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# Clear existing data (if any)
existing_count = collection.count_documents({})
if existing_count > 0:
    print(f"⚠️  Found {existing_count:,} existing documents in MongoDB")
    print("🗑️  Clearing existing data...")
    collection.delete_many({})
    print("✓ Cleared")

# Fetch and import data
print(f"\n📊 Importing {total_count:,} trials to MongoDB...")
cursor.execute("SELECT nct_id, raw_json FROM trials")

batch_size = 1000
batch = []
imported = 0
errors = 0

for row in cursor:
    nct_id, raw_json = row
    
    # Parse the JSON data
    try:
        if raw_json and raw_json.strip():
            json_data = json.loads(raw_json)
            batch.append(json_data)
            
            if len(batch) >= batch_size:
                collection.insert_many(batch)
                imported += len(batch)
                print(f"  ✓ Imported {imported:,} / {total_count:,} trials...")
                batch = []
        else:
            errors += 1
    except json.JSONDecodeError as e:
        errors += 1
        if errors <= 5:  # Only show first 5 errors
            print(f"  ⚠️  Error parsing JSON for {nct_id}: {str(e)[:50]}...")
        continue

# Import remaining batch
if batch:
    collection.insert_many(batch)
    imported += len(batch)
    print(f"  ✓ Imported {imported:,} / {total_count:,} trials...")

# Verify import
final_count = collection.count_documents({})
print(f"\n{'='*60}")
print(f"✅ Import Complete!")
print(f"{'='*60}")
print(f"SQLite records: {total_count:,}")
print(f"MongoDB documents: {final_count:,}")
print(f"Errors encountered: {errors}")

if final_count == imported:
    print(f"✓ Successfully imported {imported:,} records!")
else:
    print(f"⚠️  Warning: Count mismatch! Expected {imported:,} but got {final_count:,}")

# Show sample document
print(f"\n📄 Sample document structure:")
sample = collection.find_one()
if sample:
    sample.pop('_id', None)  # Remove MongoDB ID for cleaner display
    sample_str = json.dumps(sample, indent=2)
    # Show first 500 characters
    print(sample_str[:500] + "..." if len(sample_str) > 500 else sample_str)

# Close connections
sqlite_conn.close()
mongo_client.close()

print(f"\n{'='*60}")
print("🎉 MongoDB is ready! You can now run: python app.py")
print(f"{'='*60}")
