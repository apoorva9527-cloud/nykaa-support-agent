# Nykaa Domain Support Agent

A demo e-commerce support agent for policy questions and order-status lookup.

> This is an educational project. All policies and order data are fictional and do not represent Nykaa's real policies.

## Features completed in Part 1

- 50 synthetic e-commerce order records
- 12 policy documents in the knowledge base
- Fixed-size and recursive chunking strategies
- ChromaDB vector database
- Semantic policy retrieval using `all-MiniLM-L6-v2`
- Retrieval evaluation with Precision@3 and Recall@3

## Project structure

```text
nykaa-support-agent/
├── data/
│   └── orders.csv
├── knowledge_base/
│   └── 12 policy documents
├── src/
│   ├── data_generator.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── retrieval.py
│   └── evaluation.py
├── requirements.txt
└── README.md