# from flask import Flask, request, jsonify
# import os
# import time
# import pandas as pd
# import tensorflow as tf
# from tqdm.auto import tqdm
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# from symspellpy import SymSpell
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from pymongo import MongoClient
# import logging


# logging.basicConfig(level=logging.INFO)

# app = Flask(__name__)

# API_KEY = 'ff7d854b-3c80-4f7c-84fd-0781022e293a'

# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# dictionary_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
# bigram_path = os.path.join(os.path.dirname(__file__), "frequency_bigramdictionary_en_243_342.txt")
# sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
# sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# def getAPI(api_key):
#     pc = Pinecone(api_key=api_key)
#     cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
#     region = os.environ.get('PINECONE_REGION') or 'us-east-1'
#     spec = ServerlessSpec(cloud=cloud, region=region)
#     return pc, spec

# def create_index_if_not_exists(index_name, pc, spec, dimension):
#     existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
#     if index_name not in existing_indexes:
#         pc.create_index(
#             index_name,
#             dimension=dimension,
#             metric='cosine',
#             spec=spec
#         )
#         while not pc.describe_index(index_name).status['ready']:
#             time.sleep(1)

# def index_data(index_name, pc, dataset):
#     index = pc.Index(index_name)
#     for i in tqdm(range(0, len(dataset), 500)):
#         batch = dataset.iloc[i:i+500]
#         embeddings = batch['embeddings'].tolist()
#         ids = batch.index.astype(str).tolist()
#         metadata = batch[['combined_text']].to_dict(orient='records')
#         index.upsert(list(zip(ids, embeddings, metadata)))
#     return index

# def fetch_data_from_mongodb():
#     client = MongoClient("mongodb+srv://akshayakrishnap03:2u%25QH%40ETp%24DzNRz@searchengine.zbosqye.mongodb.net/?retryWrites=true&w=majority&appName=searchEngine")
#     db = client['searchEngine']
#     collection = db['search_engine']
#     data = list(collection.find({}))
#     dataset = pd.DataFrame(data)
#     return dataset

# def prepare_dataset(dataset, model):
#     dataset['combined_text'] = dataset['productName'] + ' ' + dataset['category'] + ' ' + dataset['description']
#     embeddings = model.encode(dataset['combined_text'].tolist(), convert_to_tensor=True)
#     dataset['embeddings'] = embeddings.tolist()
#     return dataset

# def save_index_data(dataset, save_path):
#     dataset.to_csv(save_path, index=False)

# def load_index_data(save_path):
#     dataset = pd.read_csv(save_path)
#     dataset['embeddings'] = dataset['embeddings'].apply(eval) 

# def perform_query(index, model, query, category=None):
#     suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
#     corrected_query = suggestions[0].term if suggestions else query
#     xq = model.encode([corrected_query], convert_to_tensor=True).tolist()
#     xc = index.query(vector=xq[0], top_k=5, include_metadata=True)
#     results = []
#     logging.info(f"Query: {corrected_query}")
#     for result in xc['matches']:
#         score = result['score']
#         text = result['metadata']['combined_text']
#         result_category = result['metadata'].get('category', 'Unknown')  
#         logging.info(f"Result: {result_category} - {text[:30]}... (Score: {score})")
#         if category is None or result_category.lower() == category.lower():
#             results.append((score, text))
#     return results

# def perform_batched_queries(index, model, queries, category=None):
#     batch_results = []
#     batch_size = 100
#     for i in range(0, len(queries), batch_size):
#         batch_queries = queries[i:i + batch_size]
#         encoded_queries = model.encode(batch_queries, convert_to_tensor=True).tolist()
#         results = index.query(vector=encoded_queries[0], top_k=5, include_metadata=True)['matches']
#         batch_results.extend(results)
    
#     filtered_results = []
#     for result in batch_results:
#         score = result['score']
#         text = result['metadata']['combined_text']
#         result_category = result['metadata'].get('category', 'Unknown')  
#         logging.info(f"Result: {result_category} - {text[:30]}... (Score: {score})")
#         if category is None or result_category.lower() == category.lower():
#             filtered_results.append({'score': round(score, 2), 'text': text})
#         logging.info(f"Guru: {filtered_results} - {text[:30]}... (Score: {score})")    
#     return filtered_results

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.json
#     query = data.get('query')
#     category = data.get('category')
#     logging.info(f"Received query: {query} with category: {category}")
#     if query:
#         queries = [query]
#         results = perform_batched_queries(index, model, queries, category)
#         logging.info(f"Search results: {results}")
#         return jsonify(results)
#     else:
#         return jsonify({'error': 'No query provided'}), 400

# if __name__ == '__main__':
#     save_path = 'indexed_data.csv'
#     index_name = 'semantic-search-fast'
#     dimension = 384

#     device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

#     dataset = fetch_data_from_mongodb()
#     dataset = prepare_dataset(dataset, model)
#     save_index_data(dataset, save_path)

#     pc, spec = getAPI(API_KEY)
#     create_index_if_not_exists(index_name, pc, spec, dimension)

#     indexData = pd.read_csv(save_path)
#     if os.path.exists(save_path):
#         index = pc.Index(index_name)
#     elif len(dataset) != len(indexData):
#         index = index_data(index_name, pc, dataset)
#     else:
#         index = index_data(index_name, pc, dataset)

#     app.run(host='0.0.0.0', port=5000)



# from flask import Flask, request, jsonify
# import os
# import time
# import pandas as pd
# import tensorflow as tf
# from tqdm.auto import tqdm
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# from symspellpy import SymSpell
# from pymongo import MongoClient
# import logging

# logging.basicConfig(level=logging.INFO)

# app = Flask(__name__)

# API_KEY = 'ff7d854b-3c80-4f7c-84fd-0781022e293a'

# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# dictionary_path = "frequency_dictionary_en_82_765.txt"
# bigram_path = "frequency_bigramdictionary_en_243_342.txt"
# sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
# sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
# def log_root_directory_files():
#     root_dir = '/'
#     try:
#         files_and_dirs = os.listdir(root_dir)
#         logging.info(f"Files and directories in '{root_dir}': {files_and_dirs}")
#     except Exception as e:
#         logging.error(f"Error accessing root directory: {e}")
# def getAPI(api_key):
#     pc = Pinecone(api_key=api_key)
#     cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
#     region = os.environ.get('PINECONE_REGION') or 'us-east-1'
#     spec = ServerlessSpec(cloud=cloud, region=region)
#     return pc, spec

# def create_index_if_not_exists(index_name, pc, spec, dimension):
#     logging.info(f"Checking index: {index_name}")
#     existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
#     if index_name not in existing_indexes:
#         pc.create_index(
#             index_name,
#             dimension=dimension,
#             metric='cosine',
#             spec=spec
#         )
#         while not pc.describe_index(index_name).status['ready']:
#             time.sleep(1)

# def index_data(index_name, pc, dataset):
#     index = pc.Index(index_name)
#     for i in tqdm(range(0, len(dataset), 500)):
#         batch = dataset.iloc[i:i+500]
#         embeddings = batch['embeddings'].tolist()
#         ids = batch.index.astype(str).tolist()
#         metadata = batch[['combined_text']].to_dict(orient='records')
#         index.upsert(list(zip(ids, embeddings, metadata)))
#     return index

# def fetch_data_from_mongodb():
#     client = MongoClient("mongodb+srv://raj:abcd123@cluster0.cmhoid0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
#     db = client['Products']
#     collection = db['products']
#     data = list(collection.find({}))
#     dataset = pd.DataFrame(data)
#     logging.info(f"Dataset shape:{dataset.shape}")
#     dataset = prepare_dataset(dataset, model)
#     return dataset

# def prepare_dataset(dataset, model):
#     dataset['combined_text'] = dataset['productName'] + ' ' + dataset['category'] + ' ' + dataset['description']
#     embeddings = model.encode(dataset['combined_text'].tolist(), convert_to_tensor=True)
#     dataset['embeddings'] = embeddings.tolist()
#     return dataset

# def perform_query(index_name, model, query):
#     suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
#     corrected_query = suggestions[0].term if suggestions else query
#     xq = model.encode([corrected_query], convert_to_tensor=True).tolist()
    
#     index = pc.Index(index_name)
#     xc = index.query(vector=xq[0], top_k=5, include_metadata=True)
    
#     results = []
#     logging.info(f"Query: {corrected_query}")
#     for result in xc['matches']:
#         score = result['score']
#         text = result['metadata']['combined_text']
#         result_category = result['metadata'].get('category', 'Unknown')
#         logging.info(f"Result: {result_category} - {text[:30]}... (Score: {score})")
#         results.append((score, text))
    
#     return results

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.json
#     query = data.get('query')
#     category = data.get('category')
    
#     if not query:
#         return jsonify({'error': 'No query provided'}), 400

#     if not category:
#         return jsonify({'error': 'No category provided'}), 400
    
#     index_name = f'semantic-search-{category.lower()}'
#     logging.info(f"Received query: {query} for category: {category}")
    
#     # Perform the query on the category-specific index
#     results = perform_query(index_name, model, query)
#     logging.info(f"Search results: {results}")
    
#     return jsonify(results)

# if __name__ == '__main__':
#     dimension = 384
#     device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
#     log_root_directory_files()

#     dataset = fetch_data_from_mongodb()
    

#     pc, spec = getAPI(API_KEY)


#     categories = dataset['category'].unique()
#     for category in categories:
#         index_name = f'semantic-search-{category.lower().replace(" ", "-")}'
#         logging.info(f"Search results: {index_name}")
#         category_data = dataset[dataset['category'] == category]
#         create_index_if_not_exists(index_name, pc, spec, dimension)
#         index_data(index_name.lower(), pc, category_data)

#     app.run(host='0.0.0.0', port=5000, debug=True)
# from flask import Flask, request, jsonify
# import os
# import time
# import pandas as pd
# import tensorflow as tf
# from tqdm.auto import tqdm
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
# from symspellpy import SymSpell
# from pymongo import MongoClient
# import logging

# logging.basicConfig(level=logging.INFO)

# app = Flask(__name__)

# API_KEY = 'ff7d854b-3c80-4f7c-84fd-0781022e293a'

# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# dictionary_path = "frequency_dictionary_en_82_765.txt"
# bigram_path = "frequency_bigramdictionary_en_243_342.txt"
# sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
# sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# def getAPI(api_key):
#     pc = Pinecone(api_key=api_key)
#     cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
#     region = os.environ.get('PINECONE_REGION') or 'us-east-1'
#     spec = ServerlessSpec(cloud=cloud, region=region)
#     return pc, spec

# def create_index_if_not_exists(index_name, pc, spec, dimension):
#     logging.info(f"Checking index: {index_name}")
#     existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
#     if index_name not in existing_indexes:
#         pc.create_index(
#             index_name,
#             dimension=dimension,
#             metric='cosine',
#             spec=spec
#         )
#         while not pc.describe_index(index_name).status['ready']:
#             time.sleep(1)

# def index_data(index_name, pc, dataset):
#     index = pc.Index(index_name)
#     for i in tqdm(range(0, len(dataset), 500)):
#         batch = dataset.iloc[i:i+500]
#         embeddings = batch['embeddings'].tolist()
#         ids = batch.index.astype(str).tolist()
#         metadata = batch[['combined_text']].to_dict(orient='records')
#         index.upsert(list(zip(ids, embeddings, metadata)))
#     return index

# def prepare_dataset(dataset, model):
#     dataset['combined_text'] = dataset['productName'] + ' ' + dataset['category'] + ' ' + dataset['description']
#     embeddings = model.encode(dataset['combined_text'].tolist(), convert_to_tensor=True)
#     dataset['embeddings'] = embeddings.tolist()
#     return dataset

# def fetch_data_from_mongodb():
#     client = MongoClient("mongodb+srv://raj:abcd123@cluster0.cmhoid0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
#     db = client['Products']
#     collection = db['products']
#     data = list(collection.find({}))
#     dataset = pd.DataFrame(data)
#     logging.info(f"Dataset shape:{dataset.shape}")
#     dataset = prepare_dataset(dataset, model)
#     return dataset

# def perform_query(index_name, model, query, limit, offset):
#     suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
#     corrected_query = suggestions[0].term if suggestions else query
#     xq = model.encode([corrected_query], convert_to_tensor=True).tolist()

#     index = pc.Index(index_name)
#     xc = index.query(vector=xq[0], top_k=limit + offset, include_metadata=True)
    
#     results = []
#     total_documents = len(xc['matches'])  # Assuming all documents match query
#     matches = xc['matches'][offset:offset + limit]  # Paginate results
    
#     for result in matches:
#         score = result['score']
#         text = result['metadata']['combined_text']
#         results.append({'score': score, 'text': text})
    
#     return results, total_documents

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.json
#     query = data.get('query')
#     category = data.get('category')
#     limit = int(data.get('limit', 10))  # Default limit to 10
#     offset = int(data.get('offset', 0))  # Default offset to 0

#     if not query:
#         return jsonify({'error': 'No query provided'}), 400

#     if not category:
#         return jsonify({'error': 'No category provided'}), 400

#     index_name = f'semantic-search-{category.lower()}'
    
#     results, total_documents = perform_query(index_name, model, query, limit, offset)
    
#     return jsonify({
#         'results': results,
#         'total_documents': total_documents,
#         'limit': limit,
#         'offset': offset
#     })

# @app.route('/add_product', methods=['POST'])
# def add_product():
#     data = request.json
#     product_id = data.get('product_id')
#     category = data.get('category')
#     combined_text = data.get('combined_text')
    
#     if not product_id or not category or not combined_text:
#         return jsonify({'error': 'Missing fields'}), 400

#     index_name = f'semantic-search-{category.lower()}'
#     embedding = model.encode([combined_text], convert_to_tensor=True).tolist()
    
#     index = pc.Index(index_name)
#     index.upsert([(product_id, embedding[0], {'combined_text': combined_text})])
    
#     return jsonify({'message': 'Product added to index'}), 200

# @app.route('/remove_product', methods=['POST'])
# def remove_product():
#     data = request.json
#     product_id = data.get('product_id')
#     category = data.get('category')

#     if not product_id or not category:
#         return jsonify({'error': 'Missing fields'}), 400

#     index_name = f'semantic-search-{category.lower()}'
#     index = pc.Index(index_name)
#     index.delete(ids=[product_id])
    
#     return jsonify({'message': 'Product removed from index'}), 200

# if __name__ == '__main__':
#     dimension = 384
#     device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    
#     pc, spec = getAPI(API_KEY)

#     dataset = fetch_data_from_mongodb()
#     categories = dataset['category'].unique()
#     for category in categories:
#         index_name = f'semantic-search-{category.lower().replace(" ", "-")}'
#         category_data = dataset[dataset['category'] == category]
#         create_index_if_not_exists(index_name, pc, spec, dimension)
#         index_data(index_name.lower(), pc, category_data)

#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
import os
import time
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

API_KEY = 'ff7d854b-3c80-4f7c-84fd-0781022e293a'

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
bigram_path = "frequency_bigramdictionary_en_243_342.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# Pinecone API setup
def getAPI(api_key):
    pc = Pinecone(api_key=api_key)
    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)
    return pc, spec

# Create index if it does not exist
def create_index_if_not_exists(index_name, pc, spec, dimension):
    logging.info(f"Checking index: {index_name}")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            index_name,
            dimension=dimension,
            metric='cosine',
            spec=spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

# Index data in Pinecone
def index_data(index_name, pc, dataset):
    index = pc.Index(index_name)
    for i in tqdm(range(0, len(dataset), 500)):
        batch = dataset.iloc[i:i + 500]
        embeddings = batch['embeddings'].tolist()
        ids = batch.index.astype(str).tolist()
        metadata = batch[['combined_text']].to_dict(orient='records')
        index.upsert(list(zip(ids, embeddings, metadata)))
    return index

# Prepare the dataset by generating embeddings
def prepare_dataset(dataset, model):
    dataset['combined_text'] = dataset['productName'] + ' ' + dataset['category'] + ' ' + dataset['description']
    embeddings = model.encode(dataset['combined_text'].tolist(), convert_to_tensor=True)
    dataset['embeddings'] = embeddings.tolist()
    return dataset

# Fetch data from MongoDB
def fetch_data_from_mongodb():
    client = MongoClient("mongodb+srv://raj:abcd123@cluster0.cmhoid0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client['Products']
    collection = db['products']
    data = list(collection.find({}))
    dataset = pd.DataFrame(data)
    logging.info(f"Dataset shape:{dataset.shape}")
    dataset = prepare_dataset(dataset, model)
    return dataset

# Perform semantic search query
def perform_query(index_name, model, query, limit, offset):
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    corrected_query = suggestions[0].term if suggestions else query
    xq = model.encode([corrected_query], convert_to_tensor=True).tolist()

    index = pc.Index(index_name)
    xc = index.query(vector=xq[0], top_k=limit + offset, include_metadata=True)
    
    results = []
    total_documents = len(xc['matches'])  # Assuming all documents match query
    matches = xc['matches'][offset:offset + limit]  # Paginate results
    
    for result in matches:
        score = result['score']
        text = result['metadata']['combined_text']
        results.append({'score': score, 'text': text})
    
    return results, total_documents

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    category = data.get('category')
    city_code = data.get('city_code')  # Added city_code to the search request
    limit = int(data.get('limit', 10))  # Default limit to 10
    offset = int(data.get('offset', 0))  # Default offset to 0

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if not category:
        return jsonify({'error': 'No category provided'}), 400
    
    if not city_code:
        return jsonify({'error': 'No city code provided'}), 400  # Check for city_code

    # Construct the index name using both category and city_code
    index_name = f'semantic-search-{category.lower().replace(" ", "-")}-{city_code.lower().replace(" ", "-")}'
    
    results, total_documents = perform_query(index_name, model, query, limit, offset)
    
    return jsonify({
        'results': results,
        'total_documents': total_documents,
        'limit': limit,
        'offset': offset
    })

@app.route('/add_product', methods=['POST'])
def add_product():
    data = request.json
    product_id = data.get('product_id')
    category = data.get('category')
    city_code = data.get('city_code')
    combined_text = data.get('combined_text')
    
    if not product_id or not category or not city_code or not combined_text:
        return jsonify({'error': 'Missing fields'}), 400

    index_name = f'semantic-search-{category.lower().replace(" ", "-")}-{city_code.lower().replace(" ", "-")}'
    embedding = model.encode([combined_text], convert_to_tensor=True).tolist()
    
    index = pc.Index(index_name)
    index.upsert([(product_id, embedding[0], {'combined_text': combined_text})])
    
    return jsonify({'message': 'Product added to index'}), 200

@app.route('/remove_product', methods=['POST'])
def remove_product():
    data = request.json
    product_id = data.get('product_id')
    category = data.get('category')
    city_code = data.get('city_code')

    if not product_id or not category or not city_code:
        return jsonify({'error': 'Missing fields'}), 400

    index_name = f'semantic-search-{category.lower().replace(" ", "-")}-{city_code.lower().replace(" ", "-")}'
    index = pc.Index(index_name)
    index.delete(ids=[product_id])
    
    return jsonify({'message': 'Product removed from index'}), 200

if __name__ == '__main__':
    dimension = 384
    device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    
    pc, spec = getAPI(API_KEY)

    # Fetch data from MongoDB
    dataset = fetch_data_from_mongodb()

    # Ensure dataset has both 'category' and 'cityCode' columns
    categories = dataset['category'].unique()
    city_codes = dataset['cityCode'].unique()  # Ensure 'cityCode' field exists in dataset

    # Loop over each combination of category and cityCode to create respective indexes
    for category in categories:
        for city_code in city_codes:
            # Create the index name using both category and cityCode
            index_name = f'semantic-search-{category.lower().replace(" ", "-")}-{city_code.lower().replace(" ", "-")}'
            create_index_if_not_exists(index_name, pc, spec, dimension)
            # Index the data for each category and cityCode
            index_data(index_name, pc, dataset[dataset['category'] == category][dataset['cityCode'] == city_code])

    app.run(host='0.0.0.0', port=5000, debug=True)
