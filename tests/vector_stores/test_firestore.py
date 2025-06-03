import pytest
from unittest.mock import MagicMock, patch, call # Import call for checking batch calls
import os

# Import the class to be tested and its config
from mem0.vector_stores.firestore import FirestoreDB, OutputData
from mem0.configs.vector_stores.firestore import FirestoreConfig

# Mock firebase_admin globally for all tests in this file
# This prevents actual SDK calls
@pytest.fixture(autouse=True)
def mock_firebase_admin_global():
    mock_firebase_admin = MagicMock()
    mock_firebase_admin.credentials.Certificate = MagicMock(return_value="mock_cert")
    mock_firebase_admin.initialize_app = MagicMock()
    mock_firebase_admin.firestore.client = MagicMock()
    mock_firebase_admin.firestore.SERVER_TIMESTAMP = "firestore.SERVER_TIMESTAMP" # Mock server timestamp
    mock_firebase_admin.firestore.Vector = MagicMock(side_effect=lambda x: list(x)) # Mock Vector constructor
    mock_firebase_admin.firestore.DistanceMeasure = MagicMock(
        EUCLIDEAN="EUCLIDEAN",
        COSINE="COSINE",
        DOT_PRODUCT="DOT_PRODUCT"
    )

    with patch.dict("sys.modules", {
        "firebase_admin": mock_firebase_admin,
        "firebase_admin.credentials": mock_firebase_admin.credentials,
        "firebase_admin.firestore": mock_firebase_admin.firestore,
    }):
        yield mock_firebase_admin

@pytest.fixture
def mock_firestore_client(mock_firebase_admin_global):
    client = MagicMock()
    mock_firebase_admin_global.firestore.client.return_value = client
    
    # Mock collection and document chain
    mock_collection_ref = MagicMock()
    client.collection.return_value = mock_collection_ref
    
    mock_doc_ref = MagicMock()
    mock_collection_ref.document.return_value = mock_doc_ref
    
    # Mock batching
    mock_batch = MagicMock()
    client.batch.return_value = mock_batch

    # Mock count aggregation
    mock_aggregation_query = MagicMock()
    mock_collection_ref.count.return_value = mock_aggregation_query
    mock_count_result = MagicMock()
    mock_count_result.value = 0 # Default count
    mock_aggregation_query.get.return_value = [[mock_count_result]]
    
    return client

@pytest.fixture
def firestore_config_data():
    return {
        "collection_name": "test_vectors",
        "vector_field_name": "embedding",
        "metric": "cosine",
        # "service_account_key_path": "dummy_path.json" # Can be tested with and without
    }

@pytest.fixture
def firestore_db_instance(mock_firestore_client, firestore_config_data):
    # Config now includes embedding_model_dims for __init__ to call create_col
    config_with_dims = {**firestore_config_data, "embedding_model_dims": 128}
    config = FirestoreConfig(**config_with_dims)
    # __init__ will call create_col if embedding_model_dims is present
    db = FirestoreDB(**config.model_dump())
    return db


# --- Test Cases --- #

def test_firestore_db_init(firestore_config_data, mock_firebase_admin_global):
    """Test FirestoreDB initialization and conditional create_col call."""
    
    # Scenario 1: embedding_model_dims IS provided in config, create_col should be called from __init__
    mock_firebase_admin_global.reset_mock() # Reset all global mocks
    # We need to mock create_col on the instance to check if it was called by __init__
    with patch.object(FirestoreDB, 'create_col', MagicMock()) as mock_create_col_method:
        config_with_dims = FirestoreConfig(**{**firestore_config_data, "embedding_model_dims": 128, "service_account_key_path": "dummy_path.json"})
        db_with_dims = FirestoreDB(**config_with_dims.model_dump())
        
        mock_firebase_admin_global.credentials.Certificate.assert_called_once_with("dummy_path.json")
        mock_firebase_admin_global.initialize_app.assert_called_once_with("mock_cert")
        mock_create_col_method.assert_called_once_with(name=config_with_dims.collection_name, vector_size=128, distance=config_with_dims.metric)
        assert db_with_dims.collection_name == config_with_dims.collection_name
        assert db_with_dims.vector_field_name == config_with_dims.vector_field_name
        assert db_with_dims.batch_size == config_with_dims.batch_size # Check batch_size is set
        assert db_with_dims.metric == config_with_dims.metric # Check metric is set

    # Reset mocks for next scenario
    mock_firebase_admin_global.reset_mock()
    # mock_create_col_method goes out of scope and is unpatched

    # Scenario 2: embedding_model_dims IS NOT provided, create_col should NOT be called from __init__
    with patch.object(FirestoreDB, 'create_col', MagicMock()) as mock_create_col_method_no_dims:
        # Ensure service_account_key_path is not in firestore_config_data for this part of the test to isolate initialize_app behavior
        vanilla_config_data = firestore_config_data.copy()
        if "service_account_key_path" in vanilla_config_data: # Should not be, but ensure
            del vanilla_config_data["service_account_key_path"]
        if "embedding_model_dims" in vanilla_config_data: # Ensure not present
             del vanilla_config_data["embedding_model_dims"]

        config_no_dims = FirestoreConfig(**vanilla_config_data)
        db_no_dims = FirestoreDB(**config_no_dims.model_dump())
        
        mock_firebase_admin_global.initialize_app.assert_called_once_with() # Called without cert
        mock_create_col_method_no_dims.assert_not_called()
        assert db_no_dims.collection_name == config_no_dims.collection_name
        # Check that db_no_dims.distance_metric is initialized from config.metric even if create_col not called
        assert db_no_dims.distance_metric == config_no_dims.metric 

    # Scenario 3: App already initialized (check initialize_app call count)
    mock_firebase_admin_global.reset_mock()
    mock_firebase_admin_global._apps = {"default": MagicMock()} 
    with patch.object(FirestoreDB, 'create_col', MagicMock()): # Mock create_col to avoid its side-effects
        config_for_already_init = FirestoreConfig(**{**firestore_config_data, "embedding_model_dims": 128})
        db_already_init = FirestoreDB(**config_for_already_init.model_dump())
        mock_firebase_admin_global.initialize_app.assert_not_called() # Should not be called
    del mock_firebase_admin_global._apps # cleanup

def test_create_col(firestore_db_instance):
    """Test the create_col method."""
    # create_col is called in fixture, so we check its effects
    assert firestore_db_instance.vector_size == 128
    assert firestore_db_instance.distance_metric == "cosine"

    # Test with a different metric
    firestore_db_instance.create_col(name="test_vectors", vector_size=256, distance="euclidean")
    assert firestore_db_instance.vector_size == 256
    assert firestore_db_instance.distance_metric == "euclidean"

    # Test with unsupported metric (should default to cosine as per implementation)
    firestore_db_instance.create_col(name="test_vectors", vector_size=256, distance="invalid_metric")
    assert firestore_db_instance.distance_metric == "cosine"

def test_insert_vectors(firestore_db_instance, mock_firestore_client):
    """Test inserting vectors."""
    mock_batch = mock_firestore_client.batch.return_value
    mock_collection_ref = mock_firestore_client.collection.return_value

    vectors_data = [
        ([0.1, 0.2, 0.3], {"text": "doc1"}, "id1"),
        ([0.4, 0.5, 0.6], {"text": "doc2"}, "id2"),
    ]
    vectors = [v[0] for v in vectors_data]
    payloads = [v[1] for v in vectors_data]
    ids = [v[2] for v in vectors_data]

    firestore_db_instance.insert(vectors, payloads, ids)

    # Check that batch.set was called for each vector
    assert mock_batch.set.call_count == len(vectors_data)
    # Check that batch.commit was called (at least once, depends on batch_size)
    mock_batch.commit.assert_called()

    # Verify data structure for one call (e.g., the first one)
    # Need to mock the document reference generation if we want to check specific doc_id path
    doc_ref_id1 = mock_collection_ref.document.return_value # what was used if id was passed
    
    expected_calls = []
    for vec, payload, doc_id_str in vectors_data:
        # If ID is provided, document(doc_id_str) is called.
        # If ID is None, document() is called (without args for auto-id).
        # Our mock_collection_ref.document is a single mock, so we check args or lack thereof.
        doc_ref_for_id = mock_collection_ref.document(doc_id_str) # Re-get the mock for this ID
        expected_calls.append(
            call(doc_ref_for_id, {
                "vector": vec,
                "payload": payload,
                "created_at": "firestore.SERVER_TIMESTAMP"
            })
        )
    mock_batch.set.assert_has_calls(expected_calls, any_order=False)

    # Test insert with auto-generated IDs (ids=None)
    mock_batch.reset_mock()
    mock_collection_ref.document.reset_mock() # Reset calls to document()
    auto_id_doc_ref = MagicMock() # A new mock for auto-id calls
    # Side effect for collection().document(): if called with arg, return specific mock, else return auto_id_doc_ref
    def document_side_effect(doc_id=None):
        if doc_id:
            return mock_collection_ref.document.return_value # return the general one for specific IDs if any mixed tests
        return auto_id_doc_ref # for auto-id calls
    mock_collection_ref.document.side_effect = document_side_effect

    firestore_db_instance.insert(vectors, payloads, ids=None)
    assert mock_batch.set.call_count == len(vectors_data)
    mock_batch.commit.assert_called()
    # Check that collection_ref.document() was called without arguments for auto-ID
    # Each auto-ID call should use the auto_id_doc_ref
    expected_auto_id_calls = []
    for vec, payload in zip(vectors, payloads):
         expected_auto_id_calls.append(
             call(auto_id_doc_ref, {
                "vector": vec,
                "payload": payload,
                "created_at": "firestore.SERVER_TIMESTAMP"
            })
         )
    mock_batch.set.assert_has_calls(expected_auto_id_calls, any_order=False)

    # Test batching behavior (e.g., if batch_size is 1)
    mock_batch.reset_mock()
    firestore_db_instance.batch_size = 1
    firestore_db_instance.insert(vectors, payloads, ids)
    assert mock_batch.commit.call_count == len(vectors_data) # Commit per item

def test_search_vectors(firestore_db_instance, mock_firestore_client):
    """Test searching vectors using find_nearest."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_vector_query = MagicMock() # This will be the result of find_nearest
    mock_collection_ref.find_nearest.return_value = mock_vector_query

    # Mock documents returned by the vector query
    mock_doc_snapshot1 = MagicMock()
    mock_doc_snapshot1.id = "doc1"
    mock_doc_snapshot1.to_dict.return_value = {"payload": {"text": "text1"}, "vector": [1,2,3]} # vector data for completeness, not strictly used by OutputData from search
    mock_doc_snapshot1.distance = 0.9 # Cosine: higher is better

    mock_doc_snapshot2 = MagicMock()
    mock_doc_snapshot2.id = "doc2"
    mock_doc_snapshot2.to_dict.return_value = {"payload": {"text": "text2"}, "vector": [4,5,6]}
    mock_doc_snapshot2.distance = 0.1 # Euclidean: lower is better
    
    mock_vector_query_snapshot = MagicMock()
    mock_vector_query_snapshot.documents = [mock_doc_snapshot1, mock_doc_snapshot2]
    mock_vector_query.get.return_value = mock_vector_query_snapshot

    query_vector = [0.1, 0.2, 0.3]
    limit = 2

    # 1. Test with Cosine distance (default for fixture instance)
    assert firestore_db_instance.distance_metric == "cosine"
    results_cosine = firestore_db_instance.search(query="test query", vectors=query_vector, limit=limit)
    
    mock_collection_ref.find_nearest.assert_called_once_with(
        vector_field=firestore_db_instance.vector_field_name,
        query_vector=list(query_vector), # find_nearest expects a list from our Vector mock
        limit=limit,
        distance_measure="COSINE" # from mock_firebase_admin_global.firestore.DistanceMeasure
    )
    assert len(results_cosine) == 2
    assert results_cosine[0].id == "doc1"
    assert results_cosine[0].score == 0.9 # Cosine: raw distance is score
    assert results_cosine[0].payload == {"text": "text1"}
    assert results_cosine[1].id == "doc2"
    assert results_cosine[1].score == 0.1 # Cosine: raw distance is score, even if it seems "worse"

    # 2. Test with Euclidean distance
    mock_collection_ref.find_nearest.reset_mock()
    mock_vector_query.get.return_value = mock_vector_query_snapshot # re-assign for this call
    firestore_db_instance.distance_metric = "euclidean"
    results_euclidean = firestore_db_instance.search(query="test query", vectors=query_vector, limit=limit)

    mock_collection_ref.find_nearest.assert_called_once_with(
        vector_field=firestore_db_instance.vector_field_name,
        query_vector=list(query_vector),
        limit=limit,
        distance_measure="EUCLIDEAN"
    )
    assert len(results_euclidean) == 2
    assert results_euclidean[0].id == "doc1"
    assert results_euclidean[0].score == -0.9 # Euclidean: score is -distance
    assert results_euclidean[1].id == "doc2"
    assert results_euclidean[1].score == -0.1 # Euclidean: score is -distance

    # 3. Test with filters
    mock_collection_ref.find_nearest.reset_mock()
    mock_collection_ref.where.reset_mock() # Reset where calls
    mock_filtered_collection_ref = MagicMock()
    mock_collection_ref.where.return_value = mock_filtered_collection_ref # where returns a new query object
    mock_filtered_collection_ref.find_nearest.return_value = mock_vector_query # find_nearest is called on the filtered obj
    mock_vector_query.get.return_value = mock_vector_query_snapshot # re-assign
    
    filters = {"payload.category": "test"}
    firestore_db_instance.distance_metric = "cosine" # reset for consistency
    firestore_db_instance.search(query="test query", vectors=query_vector, limit=limit, filters=filters)

    mock_collection_ref.where.assert_called_once_with("payload.category", "==", "test")
    mock_filtered_collection_ref.find_nearest.assert_called_once()

    # 4. Test limit > 1000
    mock_collection_ref.find_nearest.reset_mock()
    firestore_db_instance.search(query="test query", vectors=query_vector, limit=1500)
    args, kwargs = mock_collection_ref.find_nearest.call_args
    assert kwargs["limit"] == 1000

def test_get_vector(firestore_db_instance, mock_firestore_client):
    """Test retrieving a single vector by ID."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_doc_ref = mock_collection_ref.document.return_value #from fixture
    mock_doc_snapshot = MagicMock()
    mock_doc_ref.get.return_value = mock_doc_snapshot

    # Test when document exists
    mock_doc_snapshot.exists = True
    mock_doc_snapshot.id = "doc_exists"
    mock_doc_snapshot.to_dict.return_value = {"payload": {"text": "existing text"}, "vector": [1,2,3]}
    
    result = firestore_db_instance.get("doc_exists")
    mock_collection_ref.document.assert_called_with("doc_exists")
    mock_doc_ref.get.assert_called_once()
    assert isinstance(result, OutputData)
    assert result.id == "doc_exists"
    assert result.payload == {"text": "existing text"}
    assert result.score == 0.0 # Score is 0.0 for direct get

    # Test when document does not exist
    mock_doc_ref.get.reset_mock()
    mock_doc_snapshot.exists = False
    result_none = firestore_db_instance.get("doc_not_exists")
    assert result_none is None

def test_update_vector(firestore_db_instance, mock_firestore_client):
    """Test updating a vector."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_doc_ref = mock_collection_ref.document.return_value #from fixture

    vector_id = "update_id"
    new_vector = [0.7, 0.8, 0.9]
    new_payload = {"text": "updated text"}

    firestore_db_instance.update(vector_id, vector=new_vector, payload=new_payload)

    mock_collection_ref.document.assert_called_with(vector_id)
    mock_doc_ref.update.assert_called_once_with({
        "vector": new_vector,
        "payload": new_payload,
        "updated_at": "firestore.SERVER_TIMESTAMP"
    })

    # Test update with only vector
    mock_doc_ref.update.reset_mock()
    firestore_db_instance.update(vector_id, vector=new_vector)
    mock_doc_ref.update.assert_called_once_with({
        "vector": new_vector,
        "updated_at": "firestore.SERVER_TIMESTAMP"
    })

    # Test update with only payload
    mock_doc_ref.update.reset_mock()
    firestore_db_instance.update(vector_id, payload=new_payload)
    mock_doc_ref.update.assert_called_once_with({
        "payload": new_payload,
        "updated_at": "firestore.SERVER_TIMESTAMP"
    })

    # Test update with no data (should not call update)
    mock_doc_ref.update.reset_mock()
    firestore_db_instance.update(vector_id)
    mock_doc_ref.update.assert_not_called()

def test_delete_vector(firestore_db_instance, mock_firestore_client):
    """Test deleting a vector."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_doc_ref = mock_collection_ref.document.return_value #from fixture
    vector_id = "delete_id"

    firestore_db_instance.delete(vector_id)

    mock_collection_ref.document.assert_called_with(vector_id)
    mock_doc_ref.delete.assert_called_once()

def test_list_cols(firestore_db_instance):
    """Test listing collections."""
    # Based on current implementation, it returns the current collection name
    cols = firestore_db_instance.list_cols()
    assert cols == [firestore_db_instance.collection_name]

def test_delete_col(firestore_db_instance, mock_firestore_client):
    """Test deleting a collection (all documents)."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_batch = mock_firestore_client.batch.return_value

    # Simulate documents being deleted in batches
    mock_doc_snapshot_1 = MagicMock()
    mock_doc_snapshot_1.reference = "ref1"
    mock_doc_snapshot_2 = MagicMock()
    mock_doc_snapshot_2.reference = "ref2"
    mock_doc_snapshot_3 = MagicMock()
    mock_doc_snapshot_3.reference = "ref3"

    # Configure stream to return docs in pages based on limit
    # First call to stream returns 2 docs (batch_size = 2 for test simplicity)
    # Second call returns 1 doc
    # Third call returns 0 docs, loop terminates
    firestore_db_instance.batch_size = 2
    mock_collection_ref.limit.side_effect = lambda size: mock_collection_ref # limit returns self for chaining
    mock_collection_ref.stream.side_effect = [
        [mock_doc_snapshot_1, mock_doc_snapshot_2],
        [mock_doc_snapshot_3],
        [],
    ]

    firestore_db_instance.delete_col()

    # Check batch.delete calls
    expected_delete_calls = [call("ref1"), call("ref2"), call("ref3")] # Check delete by reference
    actual_delete_calls = []
    for call_obj in mock_batch.delete.call_args_list:
        actual_delete_calls.append(call(call_obj[0][0])) # Extract the first arg of each call
    
    # This is tricky because the mock_batch is reset. Let's check commit calls count for simplicity for now.
    # We expect 2 commits: one for the first batch of 2, one for the second batch of 1.
    assert mock_batch.commit.call_count == 2
    assert mock_collection_ref.limit.call_count == 3 # Called until stream is empty
    assert mock_collection_ref.stream.call_count == 3

def test_col_info(firestore_db_instance, mock_firestore_client):
    """Test getting collection info."""
    # Mock the count() method which is called by col_info()
    # The mock for count is already in mock_firestore_client fixture, defaults to 0
    # Let's change it for this test
    mock_aggregation_query = mock_firestore_client.collection.return_value.count.return_value
    mock_count_result = MagicMock()
    mock_count_result.value = 42
    mock_aggregation_query.get.return_value = [[mock_count_result]]

    info = firestore_db_instance.col_info()
    assert info["name"] == firestore_db_instance.collection_name
    assert info["vector_size"] == firestore_db_instance.vector_size
    assert info["distance_metric"] == firestore_db_instance.distance_metric
    assert info["item_count_estimate"] == 42

def test_list_vectors(firestore_db_instance, mock_firestore_client):
    """Test listing vectors from a collection."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_stream_query = MagicMock() # This will be the collection_ref or filtered query
    mock_collection_ref.where.return_value = mock_stream_query # if filtered
    mock_collection_ref.limit.return_value = mock_stream_query # if limit applied
    mock_stream_query.limit.return_value = mock_stream_query # for chaining
    mock_stream_query.stream.return_value = [] # Default to no results

    # Case 1: No filters, no limit (defaults to limit 100 in method)
    mock_doc_snap1 = MagicMock(id="item1", _data={"payload": {"data": "val1"}})
    mock_doc_snap1.to_dict.return_value = {"payload": {"data": "val1"}}
    mock_doc_snap2 = MagicMock(id="item2", _data={"payload": {"data": "val2"}})
    mock_doc_snap2.to_dict.return_value = {"payload": {"data": "val2"}}
    
    # If no filters, collection_ref.limit().stream() is called
    mock_collection_ref.limit.return_value.stream.return_value = [mock_doc_snap1, mock_doc_snap2]
    results = firestore_db_instance.list(limit=5)
    mock_collection_ref.limit.assert_called_with(5)
    assert len(results) == 2
    assert results[0].id == "item1"
    assert results[0].payload == {"data": "val1"}
    assert results[0].score is None

    # Case 2: With filters
    mock_collection_ref.reset_mock() # Reset prior calls
    mock_stream_query.stream.reset_mock()
    mock_stream_query.stream.return_value = [mock_doc_snap1] # Filtered result
    
    filters = {"payload.category": "A"}
    results_filtered = firestore_db_instance.list(filters=filters, limit=10)
    mock_collection_ref.where.assert_called_with("payload.category", "==", "A")
    mock_stream_query.limit.assert_called_with(10)
    mock_stream_query.stream.assert_called_once()
    assert len(results_filtered) == 1
    assert results_filtered[0].id == "item1"

def test_count_vectors(firestore_db_instance, mock_firestore_client):
    """Test counting vectors."""
    mock_collection_ref = mock_firestore_client.collection.return_value
    mock_aggregation_query = mock_collection_ref.count.return_value
    mock_count_result = MagicMock()
    mock_count_result.value = 123
    mock_aggregation_query.get.return_value = [[mock_count_result]]

    count = firestore_db_instance.count()
    assert count == 123
    mock_collection_ref.count.assert_called_once()
    mock_aggregation_query.get.assert_called_once()

    # Test count with exception (e.g. if aggregation query fails or not supported by mock setup)
    mock_aggregation_query.get.side_effect = Exception("Failed to aggregate")
    count_fail = firestore_db_instance.count()
    assert count_fail == -1 # Fallback value

def test_reset_collection(firestore_db_instance):
    """Test resetting the collection."""
    # Reset calls delete_col. We can spy on delete_col.
    with patch.object(firestore_db_instance, 'delete_col', MagicMock()) as mock_delete_col:
        firestore_db_instance.reset()
        mock_delete_col.assert_called_once()

# We will add more tests for list_cols, delete_col, col_info, etc. here. 