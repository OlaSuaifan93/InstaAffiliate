from flask import Flask, request, jsonify
import asyncio
import asyncpg
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load data and pre-trained language model
df = pd.read_csv('Vendors dataset.csv')
nlp = spacy.load("en_core_web_md")

# Extract categories and their vectors
categories = df['category'].tolist()
category_vectors = [nlp(service).vector for service in categories]


def find_closest_category(query):
    query_vector = nlp(query).vector
    similarities = cosine_similarity([query_vector], category_vectors)
    closest_idx = similarities.argmax()
    return categories[closest_idx]


# Flask app
app = Flask(__name__)


async def get_db_connection():
    """Establish a new database connection."""
    return await asyncpg.connect(
        user='olasuaifan',
        password='Ola@1234',
        database='postgres',
        host='localhost',
        port='5432'
    )


@app.route('/search', methods=['GET'])
async def search_vendors():
    service = request.args.get('service')
    if not service:
        return jsonify({"error": "Service parameter is required"}), 400

    # Find the closest category
    category = find_closest_category(service)

    try:
        conn = await get_db_connection()

        # Step 1: Find the cluster for the given category
        cluster_row = await conn.fetchrow("SELECT cluster FROM vendors WHERE category = $1", category)
        if not cluster_row:
            await conn.close()
            return jsonify({"error": "Category not found"}), 404

        cluster = cluster_row['cluster']

        # Step 2: Retrieve vendors in the same cluster
        rows = await conn.fetch("SELECT * FROM vendors WHERE cluster = $1", cluster)

        # Convert rows to dictionaries
        vendors = [dict(row) for row in rows]

        await conn.close()

        return jsonify(vendors)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app in a way compatible with async calls
    import nest_asyncio

    nest_asyncio.apply()  # Allows Flask and asyncio to coexist
    app.run(debug=True)
