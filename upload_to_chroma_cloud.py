#!/usr/bin/env python3
"""
Upload embeddings to ChromaDB Cloud
Run this once to populate your production vector database
"""

import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from pymongo import MongoClient
from tqdm import tqdm

from services.embeddings import EMBEDDING_MODEL, build_embedding_text

# Load environment variables
load_dotenv()

# Configuration
BATCH_SIZE = 100  # Process trials in batches

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'clinical_trials')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'studies')

# ChromaDB Cloud credentials
CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
CHROMA_TENANT = os.getenv('CHROMA_TENANT')
CHROMA_DATABASE = os.getenv('CHROMA_DATABASE', 'clinicalchat')

# OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def main():
    print("=" * 70)
    print("ChromaDB Cloud Upload Script")
    print("=" * 70)
    
    # Validate environment variables
    print("\n[1/7] Validating configuration...")
    
    if not MONGO_URI:
        print("❌ MONGO_URI not found in environment variables")
        return
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    if not CHROMA_API_KEY or not CHROMA_TENANT:
        print("❌ CHROMA_API_KEY and CHROMA_TENANT required for ChromaDB Cloud")
        print("   Sign up at: https://www.trychroma.com/")
        return
    
    print(f"✓ MongoDB URI: {MONGO_URI[:30]}...")
    print(f"✓ OpenAI API Key: {OPENAI_API_KEY[:10]}...")
    print(f"✓ ChromaDB Tenant: {CHROMA_TENANT}")
    print(f"✓ ChromaDB Database: {CHROMA_DATABASE}")
    
    # Connect to MongoDB
    print("\n[2/7] Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    
    total_trials = collection.count_documents({})
    print(f"✓ Connected to MongoDB: {total_trials:,} trials found")
    
    if total_trials == 0:
        print("❌ No trials found in database!")
        return
    
    # Initialize OpenAI
    print("\n[3/7] Initializing OpenAI...")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"✓ OpenAI client initialized (model: {EMBEDDING_MODEL})")
    
    # Connect to ChromaDB Cloud
    print("\n[4/7] Connecting to ChromaDB Cloud...")
    chroma_client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    print("✓ Connected to ChromaDB Cloud")
    
    # Create or get collection
    print("\n[5/7] Setting up collection...")
    collection_name = 'clinical_trials_embeddings'
    
    try:
        # Try to get existing collection
        chroma_collection = chroma_client.get_collection(name=collection_name)
        existing_count = chroma_collection.count()
        print(f"✓ Found existing collection: {existing_count} embeddings")
        
        # Ask user if they want to reset
        print(f"\n⚠️  Collection already exists with {existing_count} embeddings")
        response = input("Do you want to DELETE and recreate? (yes/no): ").strip().lower()
        
        if response == 'yes':
            chroma_client.delete_collection(name=collection_name)
            print("✓ Deleted existing collection")
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Clinical trials semantic search embeddings"}
            )
            print("✓ Created new collection")
        else:
            print("✓ Using existing collection (will skip duplicates)")
            
    except Exception:
        # Collection doesn't exist, create it
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Clinical trials semantic search embeddings"}
        )
        print(f"✓ Created new collection: {collection_name}")
    
    # Process trials in batches
    print(f"\n[6/7] Processing {total_trials:,} trials in batches of {BATCH_SIZE}...")
    print("This will take some time and cost ~$0.10-1.00 in OpenAI credits")
    
    response = input("\nProceed with upload? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Upload cancelled")
        return
    
    processed = 0
    skipped = 0
    errors = 0
    
    # Get existing IDs to avoid duplicates
    try:
        existing_ids = set(chroma_collection.get()['ids'])
        print(f"Found {len(existing_ids)} existing embeddings (will skip)")
    except Exception:
        existing_ids = set()
    
    cursor = collection.find().batch_size(BATCH_SIZE)
    
    with tqdm(total=total_trials, desc="Processing trials") as pbar:
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        
        for trial in cursor:
            nct_id = trial.get('nct_id')
            
            if not nct_id:
                skipped += 1
                pbar.update(1)
                continue
            
            # Skip if already exists
            if nct_id in existing_ids:
                skipped += 1
                pbar.update(1)
                continue
            
            # Create text for embedding
            text = build_embedding_text(trial)
            
            # Prepare metadata (ChromaDB has field length limits)
            metadata = {
                'nct_id': nct_id,
                'title': trial.get('title', '')[:500],  # Limit length
                'status': trial.get('status', '')[:100],
            }
            
            batch_ids.append(nct_id)
            batch_texts.append(text)
            batch_metadatas.append(metadata)
            
            # Process batch when full
            if len(batch_ids) >= BATCH_SIZE:
                try:
                    # Generate embeddings using OpenAI
                    response = openai_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch_texts
                    )
                    embeddings = [item.embedding for item in response.data]
                    
                    # Store in ChromaDB Cloud
                    chroma_collection.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_texts
                    )
                    
                    processed += len(batch_ids)
                    
                except Exception as e:
                    print(f"\n❌ Error processing batch: {str(e)}")
                    errors += len(batch_ids)
                
                # Clear batch
                batch_ids = []
                batch_texts = []
                batch_metadatas = []
                
                pbar.update(BATCH_SIZE)
        
        # Process remaining batch
        if batch_ids:
            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts
                )
                embeddings = [item.embedding for item in response.data]
                
                chroma_collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
                
                processed += len(batch_ids)
                
            except Exception as e:
                print(f"\n❌ Error processing final batch: {str(e)}")
                errors += len(batch_ids)
            
            pbar.update(len(batch_ids))
    
    # Final verification
    print("\n[7/7] Verifying upload...")
    final_count = chroma_collection.count()
    
    print("\n" + "=" * 70)
    print("UPLOAD COMPLETE")
    print("=" * 70)
    print(f"✓ Total trials in MongoDB: {total_trials:,}")
    print(f"✓ Embeddings uploaded: {processed:,}")
    print(f"⚠️  Skipped (already exist): {skipped:,}")
    if errors > 0:
        print(f"❌ Errors: {errors:,}")
    print(f"✓ Total in ChromaDB Cloud: {final_count:,}")
    print("\n✅ Your production vector database is ready!")
    print("\nAdd these to your production environment:")
    print(f"  CHROMA_API_KEY={CHROMA_API_KEY[:10]}...")
    print(f"  CHROMA_TENANT={CHROMA_TENANT}")
    print(f"  CHROMA_DATABASE={CHROMA_DATABASE}")


if __name__ == '__main__':
    main()
