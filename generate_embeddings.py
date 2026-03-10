"""
Generate embeddings for clinical trials and store in ChromaDB
This script processes all trials in MongoDB and creates vector embeddings for semantic search
Uses OpenAI embeddings for high-quality embeddings
"""

import os
from openai import OpenAI
import chromadb
from tqdm import tqdm
import json
from dotenv import load_dotenv
import time

from db_utils import get_mongo_client
from services.embeddings import EMBEDDING_MODEL, build_embedding_text

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_NAME = os.getenv("MONGO_DB_NAME", "clinical_trials")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "studies")

# ChromaDB settings
CHROMADB_PATH = './chromadb_data'
COLLECTION_NAME_CHROMA = 'clinical_trials_embeddings'

# OpenAI client
openai_client = OpenAI()

# =============================================================================
# MAIN PROCESS
# =============================================================================


def main():
    print("=" * 70)
    print("Clinical Trials Embeddings Generation")
    print("=" * 70)

    # Step 1: Connect to MongoDB
    print("\n[1/5] Connecting to MongoDB...")
    mongo_client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]

    total_trials = collection.count_documents({})
    print(f"✓ Connected. Found {total_trials:,} trials")

    # Step 2: Initialize OpenAI client
    print("\n[2/5] Initializing OpenAI embeddings API")
    print(f"Using {EMBEDDING_MODEL} (1536 dimensions)")
    print("✓ OpenAI client ready")

    # Step 3: Initialize ChromaDB
    print(f"\n[3/5] Initializing ChromaDB at: {CHROMADB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

    # Delete existing collection if it exists (for clean start)
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME_CHROMA)
        print("✓ Deleted existing collection")
    except Exception:
        pass

    # Create new collection
    chroma_collection = chroma_client.create_collection(
        name=COLLECTION_NAME_CHROMA,
        metadata={"description": "Clinical trials semantic search embeddings"}
    )
    print(f"✓ Created ChromaDB collection: {COLLECTION_NAME_CHROMA}")

    # Step 4: Process all trials and generate embeddings
    print(f"\n[4/5] Processing {total_trials:,} trials and generating embeddings...")
    print("(This may take 5-10 minutes depending on your machine)")

    trials = list(collection.find({}))
    batch_size = 100

    for i in tqdm(range(0, len(trials), batch_size), desc="Processing batches"):
        batch = trials[i:i+batch_size]

        # Prepare data for this batch
        ids = []
        texts = []
        metadatas = []

        for trial in batch:
            nct_id = trial.get('nct_id', str(trial['_id']))

            # Create text for embedding
            text = build_embedding_text(trial, max_summary_chars=500)

            # Create metadata (store key info for retrieval)
            metadata = {
                'nct_id': nct_id,
                'title': trial.get('title', '')[:500],  # Chroma has limits
                'status': trial.get('status', ''),
                'conditions': json.dumps(trial.get('conditions', []))[:500],
                'interventions': json.dumps(trial.get('interventions', []))[:500]
            }

            ids.append(nct_id)
            texts.append(text)
            metadatas.append(metadata)

        # Generate embeddings using OpenAI
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]

        # Store in ChromaDB
        chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # Small delay to avoid rate limits
        time.sleep(0.5)

    print(f"\n✓ Successfully processed {total_trials:,} trials")

    # Step 5: Test the embeddings
    print("\n[5/5] Testing semantic search...")
    test_query = "diabetes treatment"

    # Generate embedding for test query
    query_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[test_query]
    )
    query_embedding = query_response.data[0].embedding

    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"\n✓ Test query: '{test_query}'")
    print("Top 3 semantic matches:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n  {i}. {metadata['nct_id']}")
        print(f"     {metadata['title'][:100]}...")

    print("\n" + "=" * 70)
    print("✓ Embeddings generation complete!")
    print("=" * 70)
    print(f"\nChromaDB location: {CHROMADB_PATH}")
    print(f"Total embeddings: {chroma_collection.count()}")
    print("\nYou can now use semantic search in your application!")


if __name__ == '__main__':
    main()
