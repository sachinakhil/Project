import json
from typing import Dict, List, Optional
from pymilvus import MilvusClient, DataType
import os 
from pymilvus.client.types import LoadState
from models.models import RelatedPDPResponse, SearchField
from gcp.KeywordExtractor import KeywordExtractor
from config.config import ENVIRONMENT_CONFIG
from langchain_ibm import  WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from utils.logger import logger

DIM_VALUE = ENVIRONMENT_CONFIG.DIM_VALUE
MILVUS_URI = ENVIRONMENT_CONFIG.MILVUS_URI
MILVUS_USER = ENVIRONMENT_CONFIG.MILVUS_USER
MILVUS_PASSWORD = ENVIRONMENT_CONFIG.MILVUS_PASSWORD
EMBEDDING_MODEL = ENVIRONMENT_CONFIG.MILVUS_EMBEDDING_MODEL

class MilvusManager():
    def __init__(self, uri:str = MILVUS_URI, user:str = MILVUS_USER, password:str = MILVUS_PASSWORD,dim_value:str=DIM_VALUE,embedding_model_value:str = eval(EMBEDDING_MODEL).value):
        logger.info("Initializing MilvusManager instance")
        self.uri = uri
        self.user = user
        self.password = password
        credentials = {
                    "url": ENVIRONMENT_CONFIG.IBM_URL,
                    "apikey": ENVIRONMENT_CONFIG.IBM_API_KEY,
                    "project_id" : ENVIRONMENT_CONFIG.IBM_PROJECT_ID
                    }
        
        self.credentials = credentials
        self.project_id = credentials["project_id"]   
        DIM_VALUE = dim_value or DIM_VALUE     
        self.embedding_model_value = embedding_model_value
        self.client = None #self._connect()  # establish connection 
        self.embedding = self._embedding(self.embedding_model_value)
             
    def _embedding(self,embedding_model_value):
            logger.info("Executing _embedding") 
            embeddings = WatsonxEmbeddings(
                model_id= embedding_model_value,
                url=self.credentials["url"],
                apikey=self.credentials["apikey"],
                project_id=self.credentials["project_id"]
            )
            logger.info("Embeddings instance created successfully")
            return embeddings
        
    def _embedding_text(self, text:str):
        logger.info("Executing _embedding_text")
        max_length = 512
        tokens = text.split()
        segments= [tokens[i:i+max_length] for i in range(0,len(tokens),max_length)]           
        embeddings_merge = []
        for segment in segments:
            segment_text =  ''.join(segment)
            logger.info("segment_text:%d",len(segment_text))
            result = self.embedding.embed_query(segment_text)
            embeddings_merge.extend(result) 
                      
        logger.info("Text embedded successfully")
        
        return embeddings_merge
    def _connect(self):
        try:
            logger.info("Executing _connect")
            if self.client is None:
                logger.info("Creating MilvusClient instance")
                self.client = MilvusClient(
                            uri=self.uri
                            , user=self.user
                            , password=self.password
                            , secure=True
                            , keep_alive=True
                            )
                logger.info("MilvusClient instance is created successfully")
            return self.client
        except Exception as e:
            logger.error("An error occurred during _connect method :{e}")
            raise

    def _load_schema(self):
        # Get the directory path of the current module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the absolute path to the JSON file
        json_file_path = os.path.join(module_dir, 'schema.json')
        with open(json_file_path, 'r') as file:
            schema = json.load(file)
        return schema
    
    def list_all_collection(self):
        return self.client.list_collections()
    
    def drop_all_collection(self):
        for collection in self.client.list_collections():
            self.client.drop_collection(collection)
       
    def _create_collections_from_json(self):
        schema = self._load_schema()
        for collection_info in schema['collections']:
            collection_name = collection_info['name']
            collection_description = collection_info['description']
            fields = collection_info['fields']
            
            schema = self.client.create_schema(
                        auto_id=False,
                        enable_dynamic_fields=True,
                        description=collection_description,
                    )
            
            #get collection fields
            collection_fields = [
                schema.add_field(field_name=field["name"], datatype=DataType[field["dtype"]], **field.get("other_params", {}))
                for field in fields
            ]

            # check and create collections
            if not self.client.has_collection(collection_name):
                # collection_schema = CollectionSchema(fields=collection_fields, description=collection_description, )
                self.client.create_collection(collection_name=collection_name,
                                              schema=schema,
                                              consistency_level="Strong")
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name='id')
                for field in fields:
                    if field["dtype"] == "FLOAT_VECTOR":
                        index_params.add_index(field_name=field["name"],
                                        index_type="IVF_FLAT",
                                        metric_type="L2",
                                        params={"nlist": 128})
                self.client.create_index(collection_name=collection_name,index_params=index_params)
            else:
                logger.info(f"Collection '{collection_name}' already exists")
    
    def _init(self):
        logger.info("Executing _init")
        self.client.drop_all_collection()
        self._create_collections_from_json()
        
    def bulk_insert_data(self, rows, is_with_milvus_init:bool = False):
        logger.info("Executing bulk_insert_data")
        try:
            if is_with_milvus_init:
                self._init()
            data_to_add = self.generate_vector_data(rows)
            for collection_name in data_to_add.keys():
                chunk_size = 10
                for i in range(0, len(data_to_add[collection_name]), chunk_size):
                    data_toinsert = data_to_add[collection_name][i:i + chunk_size]
                    insert_result = self.client.insert(
                            collection_name=collection_name,
                            data=data_toinsert,
                        )
                # insert_result = self.client.insert(
                #             collection_name=collection_name,
                #             data=data_to_add[collection_name],
                #         )
                logger.info(f"Number of entities in Milvus: {insert_result['insert_count']}") 
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
        
    def deletepdpinfo(self, pdp_ids:str, collection_name:str = 'SEARCH_PDP_ENTITY'):
        result =[]
        # delete Vectors and PDP Entity 
        result.append(
            self.client.delete(
                collection_name= collection_name,
                ids=pdp_ids
            )
        )
        return result  
        
    def load_collections(self, collection_names:List[str]):
        for collection_name in collection_names:
            self.client.load_collection(collection_name)
        
    def _handle_empty_or_none(self, value, field_dtype):
        if value is None or (isinstance(value, str) and not value.strip()):  # Check if value is None or empty string
            if field_dtype == 'FLOAT_VECTOR':
                value = "__EMPTY__"  # Replace with embedding of "empty"
            elif field_dtype == 'INT64':
                value = -1  # Replace with a placeholder value for INT64
            elif field_dtype == 'STRING':
                value = ""  # Replace with an empty string for STRING
            # Add more conditions for other data types as needed
        return value
    
    def _read_metadata(self, data, prompt):
        user_keywords = prompt.split()
        metadata_list:RelatedPDPResponse = []
        seen_brands = set()
        extractor = KeywordExtractor()
        for sublist in data:  # Iterate through the outer list
            # logger.info(f"sublist:{sublist}")
            for item in sublist:  # Iterate through the inner list
                # logger.info(f"item:{item}")
                metadata = item['entity']['metadata']
                id = metadata.get('id', None)
                title = metadata.get('title', 'No Title')
                description = metadata.get('description', 'No Description')
                brand = metadata.get('brand', 'No Brand')
                # logger.info(f"brand:{brand}")
                bullets_data = metadata.get('bullets_data', '').split('    ')
                keywords = metadata.get('keywords', '').split('?')
                
                keywords_set = set(keywords)
                user_keywords_set = set(user_keywords)

                # Calculate the intersection
                matched_keyword_count = len(keywords_set.intersection(user_keywords_set))
                if brand  not in seen_brands :
                    # logger.info("brand not in seen_brands")
                    seen_brands.add(brand)
                    metadata_list.append({
                        'id': id,
                        'title': title,
                        'description': description,
                        'bullets_data': [bullet.strip() for bullet in bullets_data if bullet.strip()],
                        'brand': brand,
                        'matched_keywords_count': matched_keyword_count,
                        'keywords': keywords
                    })
            
                if len(metadata_list) >= 3:
                    # logger.info("len is 3")
                    break    

        return metadata_list
    
    def generate_vector_data(self, rows):
        vector_data = {}
        schema = self._load_schema()
        extractor = KeywordExtractor()
        for collection_info in schema['collections']:
            collection_name = collection_info['name']
            collection_fields = collection_info['fields']
            
            if collection_name not in vector_data:
                vector_data[collection_name] = []
            for row in rows:
                collection_row = {}
                extracted_keywords = str(row.get('keywords',''))
                if extracted_keywords:
                        extracted_keywords = extracted_keywords.lower()
                        extracted_keywords = set(extracted_keywords.split(","))
                       # logger.info(f"Using keywords from table:{extracted_keywords}")
                else:                       
                        title =  str(row.get('title', ''))
                        description = str(row.get('description', ''))
                        bullets_data = str(row.get('bullets_data', ''))
                        brand = str(row.get('brand', ''))
                        text_corpus = f"{title} {description} {bullets_data}"
                        text_corpus = text_corpus.lower().strip()
                        extracted_keywords = extractor.get_keywords_from_product_info(text_corpus,brand)
                        if not extracted_keywords:
                            logger.info("No_keywords from table and extraction process")
                            extracted_keywords = 'no_keywords'
                            extracted_keywords = extracted_keywords.lower()
                            extracted_keywords = set(extracted_keywords.split(","))
                            
                for field in collection_fields:
                    field_name = field['name']
                    field_dtype = field['dtype']
                    
                    if field_dtype == 'FLOAT_VECTOR':
                        vector_field_name = field['property_name']
                        if isinstance(vector_field_name, list):
                            # values = [str(row.get(field_key, '')) for field_key in vector_field_name]
                            # # Concatenate the values into a single string
                            # text_corpus = ' '.join(values)
                            # text_corpus = text_corpus.lower().strip()
                            # extracted_keywords = extractor.extract_keywords(text_corpus)
                            value = ' '.join(extracted_keywords)
                        else:
                            value = row.get(vector_field_name, None)
                    elif field_dtype == 'JSON':
                        # PDP metadata
                        extracted_keywords = '?'.join(extracted_keywords)
                        pdp_metadata = {
                            'id': row.get('id', None),
                            'title': row.get('title', None),
                            'description': row.get('description',None),
                            'bullets_data': row.get('bullets_data', None),
                            'brand': row.get('brand', None),
                            'keywords': extracted_keywords
                        }
                        value = pdp_metadata
                    else:
                        value = row.get(field_name, None)
                    
                    
                    value = self._handle_empty_or_none(value, field_dtype)
                    if field_dtype == 'FLOAT_VECTOR':
                        value = self._embedding_text(value)
                         
                    collection_row[field_name] = value
                vector_data[collection_name].append(collection_row)
        return vector_data
 
    def _read_query_metadata(self, data):
        metadata_list:RelatedPDPResponse = []

        for item in data:  # Iterate through the outer list
            metadata = item['metadata']
            id = metadata.get('id', None)
            title = metadata.get('title', 'No Title')
            description = metadata.get('description', 'No Description')
            brand = metadata.get('brand', 'No Brand')
            bullets_data = metadata.get('bullets_data', '').split('    ')
            keywords = metadata.get('keywords', '').split()
            metadata_list.append({
                'id': id,
                'title': title,
                'description': description,
                'bullets_data': [bullet.strip() for bullet in bullets_data if bullet.strip()],
                'brand': brand,
                'keywords': keywords
            })

        return metadata_list
    
    #https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Vector/query.md
    # always specify the output_fields
    def query(self, filter:str, collection_name:str = 'SEARCH_PDP_ENTITY'):
        try:
            is_loaded = self.load_collection(collection_name)
            if is_loaded:
                records = self.client.query(
                    collection_name=collection_name,
                    filter=filter,
                    output_fields=["metadata"])
                return self._read_query_metadata(records)
            else:
                return None
        except Exception as e:
            logger.error(f"Error querying data:  {e}")
            
            return None
        finally:
            self.release_collection(collection_name)
            
    #https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Vector/search.md
    def search(self, prompt:str, filter:str,  field:SearchField = None, output_fields:List[str]= ["metadata"], collection_name:str = 'SEARCH_PDP_ENTITY', limit:int = 20, 
               search_params:Optional[Dict] = {"metric_type": "L2", "params": {"nprobe": 10}}):
        try:
            prompt = prompt.lower().strip()
            data = self._embedding_text(prompt)
            data = [data]
            
            is_loaded = self.load_collection(collection_name)
            if is_loaded:
                search_args = {
                "data": data,
                "anns_field": "vector",
                "collection_name": collection_name,
                "limit": limit,
                "output_fields": output_fields,
                "search_params": search_params
                }

                if filter is not None:
                    search_args["filter"] = filter
                record = self.client.search(**search_args)
                
                if len(record)> 0:
                    return self._read_metadata(record, prompt)
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"Error querying data:  {e}")
            return None
        finally:
            self.release_collection(collection_name)
             
    def load_collection(self, collection_name:str):
        self.client.load_collection(collection_name)
        result =self.client.get_load_state(
                collection_name=collection_name,
            ) # Loaded
        return result["state"]== LoadState.Loaded
    
    def release_collection(self, collection_name:str):
        self.client.release_collection(collection_name)
        result =self.client.get_load_state(
                collection_name=collection_name,
            ) # Unloaded
        return result["state"]== LoadState.NotLoad
    
    def close_connection(self):
        if self.client is not None:
            self.client.close()

    def __enter__(self):
            self._connect()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_connection()