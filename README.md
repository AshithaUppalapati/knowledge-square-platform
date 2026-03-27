🟢 Knowledge Square — Enterprise Document & Knowledge Platform
> Enterprise-scale knowledge lakehouse combining Spark batch ETL, graph-based knowledge indexing (JanusGraph + Solr), and ML-driven recommendations via TensorFlow and Rasa — served through a Django/Flask API layer.
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)
![Apache Spark](https://img.shields.io/badge/Apache-Spark-E25A1C?style=flat-square&logo=apachespark)
![Django](https://img.shields.io/badge/Django-4.2-092E20?style=flat-square&logo=django)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=flat-square&logo=tensorflow)
![Solr](https://img.shields.io/badge/Apache-Solr-D9411E?style=flat-square)
---
📐 Architecture
```
  DOCUMENT SOURCES
  ┌──────────────────────────────────────────────────┐
  │  ECM Systems · File Servers · KMP Repositories  │
  └────────────────┬─────────────────────────────────┘
                   │  Manifold Crawler + Spark Jobs
                   ▼
  INGESTION & STORAGE
  ┌──────────────────────────────────────────────────┐
  │  HDFS  ──────► Spark ETL ──────► PostgreSQL     │
  │                    │                             │
  │                    ▼                             │
  │              Delta Lake / Parquet               │
  └────────────────┬─────────────────────────────────┘
                   │  Index & Graph Build
          ┌────────┴────────┐
          ▼                 ▼
    Apache Solr        JanusGraph
   (full-text         (knowledge graph
    search index)      relationships)
          │                 │
          └────────┬────────┘
                   │
  SERVING LAYER
  ┌──────────────────────────────────────────────────┐
  │  Flask APIs  ────►  Django UI                   │
  │       │                                          │
  │  TensorFlow (recommendations)                   │
  │  Rasa (conversational query interface)           │
  └──────────────────────────────────────────────────┘
                   │
          End Users / Enterprise Teams
```
---
🗂️ Repository Structure
```
knowledge-square-platform/
├── ingestion/
│   ├── spark_jobs/
│   │   ├── document_ingestor.py       # Spark job: crawl → HDFS → Solr
│   │   ├── metadata_extractor.py      # Extracts document metadata
│   │   └── incremental_loader.py      # Daily incremental Oozie-triggered load
│   └── crawler/
│       └── manifold_config.yaml       # Manifold crawler config
├── graph/
│   ├── janusgraph_schema.py           # JanusGraph schema definition
│   ├── graph_builder.py               # Builds entity relationships
│   └── graph_queries.py               # Gremlin query examples
├── search/
│   ├── solr_schema.xml                # Solr schema for document indexing
│   └── solr_indexer.py                # Spark → Solr indexer
├── api/
│   ├── flask_app/
│   │   ├── app.py                     # Flask REST API
│   │   ├── routes/
│   │   │   ├── search.py              # Search endpoint
│   │   │   ├── recommendations.py     # ML recommendations
│   │   │   └── graph.py              # Knowledge graph queries
│   │   └── models/
│   │       └── document.py
│   └── django_ui/
│       ├── manage.py
│       ├── knowledge_ui/
│       │   ├── views.py
│       │   └── templates/
│       └── requirements.txt
├── ml/
│   ├── recommender/
│   │   ├── train_model.py             # TensorFlow recommender training
│   │   ├── inference.py               # Serve recommendations
│   │   └── model_config.yaml
│   └── rasa_bot/
│       ├── domain.yml                 # Rasa conversational interface
│       ├── data/nlu.yml
│       └── actions/actions.py         # Custom Rasa actions → Flask APIs
├── orchestration/
│   ├── oozie/
│   │   └── workflow.xml               # Oozie workflow for daily loads
│   └── databricks/
│       └── notebook_runner.py         # Databricks notebook orchestration
└── README.md
```
---
⚙️ Key Features
1. Spark Document Ingestion Pipeline
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, current_timestamp
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("KnowledgeSquare-Ingestor").getOrCreate()

# Incremental load — only new/modified documents
docs_df = (spark.read
    .format("jdbc")
    .option("url", ECM_JDBC_URL)
    .option("dbtable", "documents")
    .option("fetchsize", "10000")
    .load()
    .filter(col("modified_at") > last_watermark))

# Extract text content via UDF
extract_text_udf = udf(lambda path: extract_text_from_file(path), StringType())
docs_df = docs_df.withColumn("content", extract_text_udf(col("file_path")))

# Write to HDFS and PostgreSQL
docs_df.write.mode("append").parquet(HDFS_OUTPUT_PATH)
docs_df.write.jdbc(POSTGRES_URL, "documents", mode="append")
```
2. JanusGraph Knowledge Graph Builder
```python
from gremlin_python.driver import client as gremlin_client

def build_document_graph(doc_id, keywords, related_docs):
    g = traversal().withRemote(connection)

    # Add document vertex
    doc_vertex = g.addV('document').property('doc_id', doc_id).next()

    # Add keyword vertices and edges
    for keyword in keywords:
        kw_vertex = (g.V().has('keyword', 'term', keyword)
                       .fold()
                       .coalesce(unfold(), addV('keyword').property('term', keyword))
                       .next())
        g.addE('contains').from_(doc_vertex).to(kw_vertex).iterate()

    # Add related document edges
    for related_id in related_docs:
        g.V().has('document', 'doc_id', related_id) \
             .addE('related_to').from_(doc_vertex).iterate()
```
3. TensorFlow Recommender
```python
import tensorflow as tf

class DocumentRecommender(tf.keras.Model):
    def __init__(self, num_docs, embedding_dim=64):
        super().__init__()
        self.doc_embedding = tf.keras.layers.Embedding(num_docs, embedding_dim)
        self.dense = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_docs, activation='softmax')

    def call(self, inputs):
        x = self.doc_embedding(inputs)
        x = self.dense(x)
        return self.output_layer(x)
```
4. Flask Search API
```python
from flask import Flask, request, jsonify
import pysolr

app = Flask(__name__)
solr = pysolr.Solr('http://solr:8983/solr/knowledge_docs')

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    filters = request.args.get('filter', '')
    results = solr.search(query, fq=filters, rows=20, fl='id,title,author,score')
    return jsonify({
        "total": results.hits,
        "docs": [dict(d) for d in results]
    })
```
---
🚀 Getting Started
```bash
git clone https://github.com/ashitha-u/knowledge-square-platform
cd knowledge-square-platform

# Start infrastructure
docker-compose up -d  # Solr, JanusGraph, PostgreSQL

# Run ingestion
spark-submit ingestion/spark_jobs/document_ingestor.py \
  --config config/dev.yaml

# Build knowledge graph
python graph/graph_builder.py

# Start APIs
cd api/flask_app && flask run --port 5000

# Start Django UI
cd api/django_ui && python manage.py runserver
```
---
🧠 Concepts Demonstrated
✅ Spark batch ETL for large-scale document processing
✅ JanusGraph knowledge graph with Gremlin queries
✅ Apache Solr full-text search indexing
✅ TensorFlow content-based recommendation model
✅ Rasa conversational query interface
✅ Flask REST API + Django UI serving layer
✅ Oozie + Databricks orchestration
✅ Incremental watermark-based loading
---
