# AI Research Assistant RAG

AI Research Assistant RAG is a starter retrieval-based research assistant that finds relevant information from a small knowledge base and generates a simple source-aware response.

This project is designed as a foundation for a future Retrieval-Augmented Generation workflow using LangChain, vector search, and document loaders.

## Features

- Simple document knowledge base
- Query-based document retrieval
- Basic relevance scoring
- Source-aware response generation
- Ready to extend with LangChain and vector databases

## Tech Stack

- Python
- LangChain
- FAISS
- OpenAI API

## How It Works

User question  
↓  
Search knowledge base  
↓  
Retrieve relevant document  
↓  
Generate answer  
↓  
Display source  

## Example Output

Question: What is RAG in AI applications?

Answer: Based on the retrieved document, Retrieval-Augmented Generation combines document retrieval with language generation to produce more grounded responses.

Source: RAG Overview

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
