from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import phonenumbers
import pyap

df=pd.read_csv('vendors_first_draft.csv')

# Extract Contact and Personal information
def find_PhoneNumber(text):
    for match in phonenumbers.PhoneNumberMatcher(text, "US"):
        result=phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
        return result

df['Phone Number'] = df['Bio'].apply(lambda x: find_PhoneNumber(x) if pd.notna(x) else None)

def find_address(text):
    address=pyap.parse(text,country="US")
    return address

df['Address'] = df['Bio'].apply(lambda x: find_address(x) if pd.notna(x) else None)
df.fillna('Not Available',inplace=True)

# Data Cleaning
class TextCleaner():
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

cleaner = TextCleaner()
df['cleaned_text']=df['Bio'].apply(lambda x: cleaner.clean_text (str(x)) if pd.notnull(x) else x)

print(df.head())

# Feature Extraction for Clustering:
# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings
# Extract BERT embeddings for bio text
df['embeddings'] = df.apply(lambda row: get_bert_embeddings(row['Bio'] + " " + row['category']).numpy().flatten(), axis=1)

# Combine all features into one feature matrix (you can stack the embeddings and numerical features)
X = pd.concat([df['embeddings'].apply(pd.Series), df[['followers', 'posts']]], axis=1)

#Clustering Algorithms

#1. K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#convert features names to string data type
X.columns = X.columns.astype(str)
# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # Set the number of clusters
df['cluster'] = kmeans.fit_predict(X_scaled)

# 2. DBSCAN:
from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps is the maximum distance between samples
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

# 3. Hierarchical Clustering:
from sklearn.cluster import AgglomerativeClustering

# Apply Agglomerative Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=2)
df['agglo_cluster'] = agglo.fit_predict(X_scaled)

# Evaluation of Clusters

from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate(col):
    # Evaluate clustering performance using Silhouette Score
    sil_score = silhouette_score(X_scaled, df[col])
    print(f'Silhouette Score: {sil_score}')

    # Evaluate clustering performance using Davies-Bouldin Index
    db_score = davies_bouldin_score(X_scaled, df[col])
    print(f'Davies-Bouldin Index: {db_score}')

print("The k-means evaluation:",evaluate('cluster'))
print("The DBSCAN evaluation:",evaluate('dbscan_cluster'))
print("The agglo_cluster:",evaluate('agglo_cluster'))

df.to_csv('Vendors dataset.csv')
print("The data was saved to Vendors dataset.csv")