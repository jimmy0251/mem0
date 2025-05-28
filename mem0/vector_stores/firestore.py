import logging
import os
from typing import Any, Dict, List, Optional, Union

# import numpy as np # No longer needed for client-side calculations
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore import Client, CollectionReference # Corrected imports
from google.cloud.firestore_v1.base_query import FieldFilter # Import FieldFilter

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


# Added OutputData model
class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float] # distance/similarity score
    payload: Optional[Dict]  # metadata


class FirestoreDB(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        service_account_key_path: Optional[str] = None,
        vector_field_name: str = "vector",
        metric: str = "cosine",  # Added from config
        batch_size: int = 100,  # Added from config
        embedding_model_dims: Optional[int] = None, # Added from config
    ):
        """
        Initialize the Firestore vector store.

        Args:
            collection_name (str): Name of the collection.
            service_account_key_path (str, optional): Path to the Firebase service account key JSON file.
            vector_field_name (str, optional): Name of the field in Firestore documents that stores the vector embeddings.
            metric (str, optional): Distance metric for vector similarity. Supported: 'cosine', 'euclidean', 'dotproduct'.
            batch_size (int, optional): Batch size for write operations.
            embedding_model_dims (Optional[int]): Dimensions of the embedding model. If provided, create_col is called.
        """
        self.collection_name = collection_name
        self.vector_field_name = vector_field_name
        self.metric = metric.lower() # Ensure metric is stored from config if create_col is called later
        self.batch_size = batch_size
        self.embedding_model_dims = embedding_model_dims # Store for potential reference

        try:
            if not firebase_admin._apps:
                if service_account_key_path:
                    cred = credentials.Certificate(service_account_key_path)
                    firebase_admin.initialize_app(cred)
                else:
                    # If no path is provided, initialize_app will try to use
                    # GOOGLE_APPLICATION_CREDENTIALS environment variable.
                    firebase_admin.initialize_app()
                logger.info("Firebase Admin SDK initialized.")
            else:
                logger.info("Firebase Admin SDK already initialized.")
        except Exception as e:
            logger.error(f"Error initializing Firebase Admin SDK: {e}")
            raise

        self.db: Client = firestore.client()
        self.collection_ref: CollectionReference = self.db.collection(self.collection_name)
        logger.info(f"Initialized FirestoreDB with collection: {self.collection_name}, vector field: {self.vector_field_name}")
        # Initialize other necessary attributes here, e.g., batch_size for writes
        self.batch_size = 100 # Default batch size, can be configurable

        if self.embedding_model_dims is not None:
            logger.info(f"embedding_model_dims provided ({self.embedding_model_dims}), calling create_col during init.")
            self.create_col(name=self.collection_name, vector_size=self.embedding_model_dims, distance=self.metric)
        else:
            # vector_size and distance_metric will be set when create_col is called explicitly.
            # Store the configured metric for when create_col is called without a distance argument, though create_col should prioritize its args.
            self.distance_metric = self.metric # Temporarily store, create_col will formalize it.
            logger.warning(
                f"embedding_model_dims not provided in FirestoreConfig for collection '{self.collection_name}'. "
                f"Ensure create_col() is called with the correct vector_size before usage."
            )


    def create_col(self, name: str, vector_size: int, distance: str):
        """Create a new collection (conceptual for Firestore).
        Stores vector_size and distance_metric for later use with native search.
        Actual Firestore collection creation is implicit.
        Vector indexes must be configured separately in Firebase console or via gcloud/API.
        """
        logger.info(f"Configuring FirestoreDB for collection '{self.collection_name}' (was '{name}'). Vector size: {vector_size}, distance metric: {distance}")
        self.collection_name = name # Ensure collection_name is updated if different from init
        self.vector_size = vector_size
        
        metric_lower = distance.lower()
        # Supported by Firestore: EUCLIDEAN, COSINE, DOT_PRODUCT
        # Our internal representation matches these strings.
        supported_metrics = ["euclidean", "cosine", "dotproduct"]
        if metric_lower not in supported_metrics:
            logger.warning(f"Unsupported distance metric '{distance}'. Must be one of {supported_metrics}. Defaulting to 'cosine'.")
            self.distance_metric = "cosine"
        else:
            self.distance_metric = metric_lower
        
        logger.info(f"FirestoreDB configured. Collection: {self.collection_name}, Vector Field: {self.vector_field_name}, Vector Size: {self.vector_size}, Distance: {self.distance_metric}")
        logger.warning(f"Ensure a vector index is properly configured in Firestore for field '{self.vector_field_name}' in collection '{self.collection_name}' with appropriate dimension and distance measure.")
        # No actual creation step for collection itself in Firestore client SDK.
        # Index creation is done via Firebase Console, gcloud CLI, or Terraform.
        pass

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[Union[str, int]]] = None,
    ):
        """Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to an empty dict for each vector.
            ids (list, optional): List of IDs corresponding to vectors. If None, Firestore will auto-generate IDs.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        batch = self.db.batch()
        count_in_batch = 0

        for i, vector in enumerate(vectors):
            payload = payloads[i] if payloads and i < len(payloads) else {}
            doc_id = str(ids[i]) if ids and i < len(ids) else None

            doc_data = {
                "vector": vector,
                "payload": payload,
                # Potentially add created_at/updated_at timestamps
                "created_at": firestore.SERVER_TIMESTAMP
            }

            if doc_id:
                doc_ref = self.collection_ref.document(doc_id)
            else:
                doc_ref = self.collection_ref.document() # Firestore auto-generates ID
            
            batch.set(doc_ref, doc_data)
            count_in_batch += 1

            if count_in_batch >= self.batch_size:
                batch.commit()
                logger.info(f"Committed batch of {count_in_batch} documents.")
                batch = self.db.batch()
                count_in_batch = 0

        if count_in_batch > 0:
            batch.commit()
            logger.info(f"Committed final batch of {count_in_batch} documents.")

    def search(
        self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """Search for similar vectors using Firestore's native find_nearest.

        Args:
            query (str): Original query string (for logging/context, not directly used in find_nearest query itself).
            vectors (list): Query vector to search for.
            limit (int, optional): Number of results to return. Max 1000 for Firestore.
            filters (dict, optional): Filters to apply *before* the vector search (pre-filtering).
                                    Example: {"payload.field_name": "value", "payload.numeric_field": {"gte": 10, "lte": 20}}
                                    Note: Firestore's find_nearest itself doesn't combine with .where() in the same query object for all SDK versions/complex cases smoothly.
                                    The filtering here is a basic attempt to pre-filter if possible, or might need to be applied post-hoc if find_nearest limitations are hit.
                                    For now, we apply .where() before .find_nearest().

        Returns:
            list: Search results as List[OutputData]. Score is the distance from find_nearest.
        """
        logger.info(f"Native Firestore searching in '{self.collection_name}' for query: '{query}'. Limit: {limit}, Filters: {filters}")

        if limit > 1000:
            logger.warning(f"Firestore find_nearest limit cannot exceed 1000. Requested {limit}, using 1000.")
            limit = 1000

        query_vector = Vector(vectors)
        
        distance_measure_map = {
            "euclidean": DistanceMeasure.EUCLIDEAN,
            "cosine": DistanceMeasure.COSINE,
            "dotproduct": DistanceMeasure.DOT_PRODUCT
        }
        firestore_distance_measure = distance_measure_map.get(self.distance_metric, DistanceMeasure.COSINE)

        # Start with the base collection reference
        current_query = self.collection_ref

        # Apply pre-filters if provided
        # Note: Complex filtering with find_nearest might have limitations or require specific SDK versions/syntax.
        # This attempts to apply filters before the vector search.
        # Fields in filters should typically refer to fields within the 'payload' sub-object or other top-level fields.
        if filters:
            for key, value in filters.items():
                # Assuming filters might be on payload fields e.g., "payload.category" == "news"
                # Or direct fields if they exist outside payload in your schema
                if isinstance(value, dict):
                    if "gte" in value:
                        current_query = current_query.where(filter=FieldFilter(key, ">=", value["gte"]))
                    if "lte" in value:
                        current_query = current_query.where(filter=FieldFilter(key, "<=", value["lte"]))
                    if "eq" in value:
                        current_query = current_query.where(filter=FieldFilter(key, "==", value["eq"]))
                    # Firestore also supports array_contains, in, etc. which could be added here
                else:
                    current_query = current_query.where(filter=FieldFilter(key, "==", value))
        
        vector_query = current_query.find_nearest(
            vector_field=self.vector_field_name,
            query_vector=query_vector,
            limit=limit,
            distance_measure=firestore_distance_measure
        )

        try:
            # The `documents` property of VectorQuerySnapshot holds the DocumentSnapshot objects
            # Each DocumentSnapshot has a `distance` attribute when returned from find_nearest
            document_snapshots = vector_query.get()
            
            results = []
            for doc_snapshot in document_snapshots: # Iterate directly over the list of snapshots
                doc_data = doc_snapshot.to_dict()
                payload = doc_data.get("payload")
                
                # The distance is directly available on the DocumentSnapshot
                # For COSINE and DOT_PRODUCT, higher is better. For EUCLIDEAN, lower is better.
                # The `distance` attribute from Firestore find_nearest provides the raw distance.
                # We need to ensure our `score` in OutputData consistently means "higher is better".
                raw_distance = doc_snapshot.distance
                score = 0.0
                if firestore_distance_measure == DistanceMeasure.EUCLIDEAN:
                    # Invert Euclidean so higher is better (e.g., 1 / (1 + dist) or -dist)
                    # Using negative distance for simplicity in sorting if needed, though find_nearest sorts already.
                    score = -raw_distance 
                elif firestore_distance_measure == DistanceMeasure.COSINE:
                    score = raw_distance # Cosine similarity: higher is better
                elif firestore_distance_measure == DistanceMeasure.DOT_PRODUCT:
                    score = raw_distance # Dot product: higher is better
                else:
                    score = raw_distance # Default, should not happen if mapped correctly

                results.append(OutputData(id=doc_snapshot.id, score=score, payload=payload))
            
            # Firestore find_nearest already returns sorted results by distance.
            # If we modified scores (e.g. for Euclidean), we might need to re-sort if the original order isn't preserved as desired.
            # However, since we want "higher score = better", and find_nearest sorts by its metric:
            # - Euclidean: sorts ascending (smaller distance is better). Our negated score also means sort descending for "better".
            # - Cosine: sorts descending (larger similarity is better). Our score matches.
            # - Dot Product: sorts descending (larger product is better). Our score matches.
            # So, the order from Firestore should be fine if we use these scores.

            return results
        except Exception as e:
            logger.error(f"Error during Firestore find_nearest query: {e}")
            logger.error("Ensure that a vector index is configured for field '%s' in collection '%s' with distance measure %s and dimension %s." % (self.vector_field_name, self.collection_name, self.distance_metric, self.vector_size))
            return []

    def delete(self, vector_id: Union[str, int]):
        """Delete a vector by ID."""
        logger.info(f"Deleting vector with ID '{vector_id}' from collection {self.collection_name}")
        try:
            doc_ref = self.collection_ref.document(str(vector_id))
            doc_ref.delete()
            logger.info(f"Successfully deleted vector with ID '{vector_id}'.")
        except Exception as e:
            logger.error(f"Error deleting vector ID '{vector_id}': {e}")
            # Decide if we want to raise the exception or just log it
            # raise

    def update(self, vector_id: Union[str, int], vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        """Update a vector and its payload.

        Args:
            vector_id (Union[str, int]): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        logger.info(f"Updating vector ID '{vector_id}' in collection {self.collection_name}")
        doc_ref = self.collection_ref.document(str(vector_id))

        update_data = {}
        if vector is not None:
            update_data["vector"] = vector
        if payload is not None:
            update_data["payload"] = payload
        
        update_data["updated_at"] = firestore.SERVER_TIMESTAMP

        if not update_data:
            logger.warning(f"No update data provided for vector ID '{vector_id}'.")
            return

        try:
            doc_ref.update(update_data)
            logger.info(f"Successfully updated vector ID '{vector_id}'.")
        except Exception as e:
            logger.error(f"Error updating vector ID '{vector_id}': {e}")
            # Decide if we want to raise the exception or just log it
            # raise

    def get(self, vector_id: Union[str, int]) -> Optional[OutputData]:
        """Retrieve a vector by ID.

        Args:
            vector_id (Union[str, int]): ID of the vector to retrieve.

        Returns:
            Optional[OutputData]: Retrieved data including id, payload, or None if not found.
                                Score is not applicable for a direct get and will be 0.0.
        """
        logger.info(f"Retrieving vector with ID '{vector_id}' from collection {self.collection_name}")
        try:
            doc_ref = self.collection_ref.document(str(vector_id))
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                return OutputData(
                    id=doc.id,
                    score=0.0, # Score is not relevant for a direct get
                    payload=data.get("payload")
                    # We don't return the vector itself here as per OutputData and Pinecone's get
                )
            else:
                logger.warning(f"Vector with ID '{vector_id}' not found.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving vector ID '{vector_id}': {e}")
            return None

    def list_cols(self):
        """List all collections (indexes in Pinecone terminology).
        Firestore client SDK does not support listing all collections directly.
        This method would typically require admin privileges or a predefined list.
        Returning the current collection name as a list for basic compatibility.
        """
        logger.warning("Firestore client SDK cannot list all collections. Returning current collection name.")
        # This is a limitation. In a real scenario, you might have a known list of collections
        # or use the Admin SDK on a backend to get a true list.
        return [self.collection_name] # Or an object that mimics Pinecone's response if necessary

    def delete_col(self):
        """Delete a collection (all documents within it).
        This is a potentially destructive and slow operation.
        """
        logger.warning(f"Deleting all documents from collection '{self.collection_name}'. This can take a while.")
        
        # Firestore requires deleting documents in batches.
        # Based on https://firebase.google.com/docs/firestore/manage-data/delete-data#collections
        def delete_collection_batch(coll_ref, batch_size):
            docs = coll_ref.limit(batch_size).stream()
            deleted = 0
            batch = self.db.batch()
            for doc in docs:
                batch.delete(doc.reference)
                deleted += 1
            
            if deleted > 0:
                batch.commit()
                return deleted
            return 0

        deleted_total = 0
        while True:
            num_deleted = delete_collection_batch(self.collection_ref, self.batch_size)
            deleted_total += num_deleted
            if num_deleted < self.batch_size:
                break
        logger.info(f"Successfully deleted {deleted_total} documents from collection '{self.collection_name}'.")

    def col_info(self) -> Dict:
        """Get information about a collection.
        Provides basic info like name and an estimated count (if count is implemented).
        """
        logger.info(f"Getting info for collection '{self.collection_name}'.")
        # Firestore doesn't have a direct equivalent to Pinecone's describe_index for rich metadata.
        # We can return the name and potentially a count if implemented and efficient.
        num_items = self.count() # This might be inefficient
        return {
            "name": self.collection_name,
            "vector_size": self.vector_size,
            "distance_metric": self.distance_metric,
            "item_count_estimate": num_items # Note: count() can be slow
        }

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[OutputData]:
        """List vectors in a collection with optional filtering and limit.
        This is similar to search but without a query vector, returning documents.
        """
        logger.info(f"Listing documents from '{self.collection_name}'. Limit: {limit}, Filters: {filters}")

        query_builder = self.collection_ref

        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    if "gte" in value:
                        query_builder = query_builder.where(filter=FieldFilter(key, ">=", value["gte"]))
                    if "lte" in value:
                        query_builder = query_builder.where(filter=FieldFilter(key, "<=", value["lte"]))
                    if "eq" in value:
                         query_builder = query_builder.where(filter=FieldFilter(key, "==", value["eq"]))
                else:
                    query_builder = query_builder.where(filter=FieldFilter(key, "==", value))

        if limit:
            query_builder = query_builder.limit(limit)
        
        try:
            docs = query_builder.stream()
        except Exception as e:
            logger.error(f"Error listing documents from Firestore: {e}")
            return []

        results = []
        for doc in docs:
            doc_data = doc.to_dict()
            results.append(OutputData(
                id=doc.id,
                score=None, # No similarity score for a simple list operation
                payload=doc_data.get("payload")
            ))
        return results

    def count(self) -> int:
        """
        Count number of vectors/documents in the collection.
        WARNING: Firestore does not support a direct count of all documents in a collection 
        without reading them (or using Aggregation queries which are more recent).
        This implementation iterates through all documents if no filters applied and can be slow and costly for large collections.
        Consider maintaining a counter document or using server-side aggregation for production.
        For simplicity here, we stream IDs which is slightly better than full docs if no filters applied.
        """
        logger.warning(
            "count() iterates over documents to count them, which can be inefficient and costly for large collections."
            " Consider server-side aggregations or a counter document."
        )
        # This is a naive count. Firestore now supports server-side aggregation queries (count()) which are much better.
        # However, to keep it simple for now and avoid adding more complex query types:
        # aggregation_query = self.collection_ref.count()
        # query_result = aggregation_query.get()
        # return query_result[0][0].value
        # For now, let's do a less efficient but universally available approach (streaming docs)
        # A slightly more optimized but still potentially slow way without full doc reads:
        # docs = self.collection_ref.select([]).stream() # Select no fields, just get existence
        # return sum(1 for _ in docs)
        # Let's stick to what we have in list() and get its length, but this reads full docs if no filters
        # Actually, the base class expects `count` to be potentially more efficient.
        # The best approach for a client-side SDK without full reads is to use an aggregation query.
        # Let's use the aggregation query, assuming a recent enough `firebase-admin` SDK version.
        try:
            aggregation_query = self.collection_ref.count()
            query_result = aggregation_query.get()
            if query_result and query_result[0]:
                 return query_result[0][0].value
            return 0
        except Exception as e:
            logger.error(f"Error counting documents using aggregation: {e}. Falling back to slower method (not recommended for large collections).")
            # Fallback to less efficient method if aggregation fails (e.g. older SDK version or specific error)
            # This is very inefficient: Do not use in production for large collections.
            # docs_stream = self.collection_ref.stream() # Streams all documents
            # doc_count = sum(1 for _ in docs_stream)
            # return doc_count
            logger.warning("Fallback count method by streaming all docs is very inefficient. Count may be inaccurate or fail.")
            return -1 # Indicate an issue or use a very slow count

    def reset(self):
        """Reset by deleting the collection (all its documents) and it's ready for new data."""
        logger.warning(f"Resetting collection {self.collection_name} by deleting all documents...")
        self.delete_col() # This deletes all documents
        # Firestore collections are implicitly created, so no explicit re-creation step is needed after deletion.
        logger.info(f"Collection {self.collection_name} has been reset (all documents deleted).")
        # Re-initialize vector_size and distance_metric if they were cleared or if create_col needs to be called.
        # Assuming they are preserved on the instance, or create_col would be called externally if needed.
        pass 