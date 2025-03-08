from pymongo import MongoClient

def connect_to_mongodb(host='localhost', port=27017, db_name='mydb'):
    """
    Connect to a local MongoDB instance
    
    Args:
        host (str): MongoDB host (default: localhost)
        port (int): MongoDB port (default: 27017)
        db_name (str): Database name to connect to
        
    Returns:
        tuple: (client, db) - MongoDB client and database objects
    """
    try:
        # Create a connection using MongoClient
        client = MongoClient(host, port)
        
        # Access the specified database
        db = client[db_name]
        
        # Print connection info
        print(f"Connected to MongoDB at {host}:{port}")
        print(f"Database: {db_name}")
        
        return client, db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None

def main():
    # Connect to MongoDB
    client, db = connect_to_mongodb()
    
    if db is not None:
        # Examples of basic operations
        
        # Get a collection
        collection = db['my_collection']
        
        # Insert a document
        document = {"name": "John", "age": 30, "city": "New York"}
        result = collection.insert_one(document)
        print(f"Inserted document ID: {result.inserted_id}")
        
        # Find documents
        query = {"name": "John"}
        found_document = collection.find_one(query)
        print(f"Found document: {found_document}")
        
        # Close the connection when done
        client.close()
        print("MongoDB connection closed")

if __name__ == "__main__":
    main()