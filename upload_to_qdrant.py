#!/usr/bin/env python3
"""
Upload embeddings to Qdrant Cloud
Run this once to populate your production vector database with clinical trial embeddings
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from pymongo import MongoClient
from tqdm import tqdm
import uuid

from services.embeddings import EMBEDDING_MODEL, build_embedding_text

# Load environment variables
load_dotenv()

# Configuration
BATCH_SIZE = 100  # Process trials in batches
EMBEDDING_DIMENSION = 1536  # OpenAI ada-002 dimension
COLLECTION_NAME = "clinical_trials"

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI') or os.getenv('MONGO_URL')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'clinical_trials')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'studies')

# Qdrant Cloud credentials
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def main():
    print("=" * 70)
    print("Qdrant Cloud Upload Script - Clinical Trials Embeddings")
    print("=" * 70)
    
    # Validate environment variables
    print("\n[1/8] Validating configuration...")
    
    if not MONGO_URI:
        print("❌ MONGO_URI or MONGO_URL not found in environment variables")
        print("   Add MONGO_URI=your-mongodb-connection-string to .env")
        return
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Add OPENAI_API_KEY=your-openai-key to .env")
        return
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ QDRANT_URL and QDRANT_API_KEY required for Qdrant Cloud")
        print("   Sign up at: https://cloud.qdrant.io/")
        print("   Create a cluster and get your URL and API key")
        print("   Add to .env:")
        print("     QDRANT_URL=https://your-cluster.qdrant.io")
        print("     QDRANT_API_KEY=your-api-key")
        return
    
    print(f"✓ MongoDB URI: {MONGO_URI[:30]}...")
    print(f"✓ OpenAI API Key: {OPENAI_API_KEY[:10]}...")
    print(f"✓ Qdrant URL: {QDRANT_URL}")
    print(f"✓ Collection name: {COLLECTION_NAME}")
    
    # Connect to MongoDB
    print("\n[2/8] Connecting to MongoDB...")
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        
        total_trials = collection.count_documents({})
        print(f"✓ Connected to MongoDB: {total_trials:,} trials found")
        
        if total_trials == 0:
            print("❌ No trials found in database!")
            return
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
        return
    
    # Initialize OpenAI
    print("\n[3/8] Initializing OpenAI...")
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"✓ OpenAI client initialized (model: {EMBEDDING_MODEL})")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {str(e)}")
        return
    
    # Connect to Qdrant Cloud
    print("\n[4/8] Connecting to Qdrant Cloud...")
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("✓ Connected to Qdrant Cloud")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {str(e)}")
        return
    
    # Create or recreate collection
    print("\n[5/8] Setting up Qdrant collection...")
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if collection_exists:
            # Get collection info
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            existing_count = collection_info.points_count  # Changed from vectors_count
            print(f"✓ Found existing collection: {existing_count:,} vectors")
            
            # Ask user if they want to reset
            print(f"\n⚠️  Collection already exists with {existing_count:,} vectors")
            response = input("Do you want to DELETE and recreate? (yes/no): ").strip().lower()
            
            if response == 'yes':
                qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                print("✓ Deleted existing collection")
                
                # Create new collection
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Created new collection: {COLLECTION_NAME}")
            else:
                print("✓ Using existing collection")
        else:
            # Create new collection
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created new collection: {COLLECTION_NAME}")
    
    except Exception as e:
        print(f"❌ Collection setup failed: {str(e)}")
        return
    
    # Calculate cost estimate
    print("\n[6/8] Cost estimate...")
    estimated_cost = (total_trials / 1000) * 0.10  # $0.10 per 1000 embeddings
    print(f"Estimated OpenAI cost: ${estimated_cost:.2f}")
    print(f"Processing time: ~{(total_trials / 10):.0f} seconds")
    
    response = input("\nProceed with upload? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Upload cancelled")
        return
    
    # Process trials in batches
    print(f"\n[7/8] Processing {total_trials:,} trials in batches of {BATCH_SIZE}...")
    
    processed = 0
    skipped = 0
    errors = 0
    
    # Get existing IDs to check for duplicates
    try:
        existing_ids = set()
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        existing_ids = {point.payload.get('nct_id') for point in scroll_result[0] if point.payload.get('nct_id')}
        if existing_ids:
            print(f"Found {len(existing_ids)} existing vectors (will skip duplicates)")
    except Exception:
        existing_ids = set()
    
    cursor = collection.find().batch_size(BATCH_SIZE)
    
    with tqdm(total=total_trials, desc="Processing trials") as pbar:
        batch_points = []
        batch_texts = []
        batch_nct_ids = []
        
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
            
            batch_texts.append(text)
            batch_nct_ids.append(nct_id)
            
            # Store trial data for payload
            batch_points.append({
                'nct_id': nct_id,
                'title': trial.get('title', '')[:500],  # Limit length
                'status': trial.get('status', ''),
                'conditions': trial.get('conditions', [])[:10],  # Limit array size
                'interventions': trial.get('interventions', [])[:10],
            })
            
            # Process batch when full
            if len(batch_texts) >= BATCH_SIZE:
                try:
                    # Generate embeddings using OpenAI
                    response = openai_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch_texts
                    )
                    embeddings = [item.embedding for item in response.data]
                    
                    # Create Qdrant points
                    points = []
                    for i, (nct_id, embedding, payload) in enumerate(zip(batch_nct_ids, embeddings, batch_points)):
                        points.append(
                            PointStruct(
                                id=str(uuid.uuid4()),  # Generate unique ID
                                vector=embedding,
                                payload=payload
                            )
                        )
                    
                    # Upload to Qdrant
                    qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    
                    processed += len(points)
                    
                except Exception as e:
                    print(f"\n❌ Error processing batch: {str(e)}")
                    errors += len(batch_texts)
                
                # Clear batches
                batch_points = []
                batch_texts = []
                batch_nct_ids = []
                
                pbar.update(BATCH_SIZE)
        
        # Process remaining batch
        if batch_texts:
            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts
                )
                embeddings = [item.embedding for item in response.data]
                
                points = []
                for i, (nct_id, embedding, payload) in enumerate(zip(batch_nct_ids, embeddings, batch_points)):
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                
                processed += len(points)
                
            except Exception as e:
                print(f"\n❌ Error processing final batch: {str(e)}")
                errors += len(batch_texts)
            
            pbar.update(len(batch_texts))
    
    # Final verification
    print("\n[8/8] Verifying upload...")
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        final_count = collection_info.points_count  # Changed from vectors_count
        
        print("\n" + "=" * 70)
        print("UPLOAD COMPLETE")
        print("=" * 70)
        print(f"✓ Total trials in MongoDB: {total_trials:,}")
        print(f"✓ Vectors uploaded: {processed:,}")
        print(f"⚠️  Skipped (already exist): {skipped:,}")
        if errors > 0:
            print(f"❌ Errors: {errors:,}")
        print(f"✓ Total in Qdrant Cloud: {final_count:,}")
        print("\n✅ Your Qdrant production vector database is ready!")
        print("\nAdd these to your production environment:")
        print(f"  QDRANT_URL={QDRANT_URL}")
        print(f"  QDRANT_API_KEY={QDRANT_API_KEY[:10]}...")
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")


if __name__ == '__main__':
    main()
