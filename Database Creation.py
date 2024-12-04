import psycopg2
from psycopg2 import sql

# Connect to the PostgreSQL server
conn = psycopg2.connect(
    dbname="postgres",  # Default database
    user="olasuaifan",
    password="Ola@1234",
    host="localhost",   # Adjust if your database is hosted elsewhere
    port="5432"         # Default PostgreSQL port
)
conn.autocommit = True  # To allow creating databases
cur = conn.cursor()

# Create a new database
database_name = "vendors_database"
cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
print(f"Database {database_name} created successfully.")

# Connect to the newly created database
conn = psycopg2.connect(
    dbname="vendors_database",  # Use the newly created database
    user="olasuaifan",
    password="Ola@1234",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# SQL query to create a table
create_table_query = """
CREATE TABLE vendors (
    vendor_id VARCHAR(20) PRIMARY KEY,
    vendor_name VARCHAR(255) NOT NULL,
    phone_number VARCHAR(100),
    Address VARCHAR(255),
    description VARCHAR(255),
    category VARCHAR(100),
    instagram_url VARCHAR(255),
    embedding_code VARCHAR(255),
    follower_count INT,
    posts_count INT,
    website VARCHAR(255),
    cluster_id INT
);
"""
cur.execute(create_table_query)
print("Table 'vendors' created successfully.")

# Commit the transaction and close the connection
conn.commit()

for index, row in df.iterrows():
    vendor_id = row['IG username']  # extract vendor_id from IG username
    phone_number = row['Phone Number']
    address = row['Address']
    description = row['Bio']
    category = row['category']
    embedding_code = row['embeding code']
    website = row['website']
    posts_number = row['posts']
    followers_number = row['followers']

    # insert data in the table
    cursor.execute('''
    INSERT OR REPLACE INTO vendors (
    vendor_id, phone_number, address, description, category, website,
    embedding_code, posts_number, followers_number
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (vendor_id, phone_number, address, description, category, website,
          embedding_code, posts_number, followers_number))
conn.commit()
conn.close()