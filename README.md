# Building a RAG pipeline from scratch

Local RAG pipeline we're going to build:

![image](https://github.com/Raghoeveer/RAG_pipeline_from_scratch/assets/130668192/b7a3e96b-2c1b-4fc0-8375-bdf7151973b2)


This is a flowchart describing a simple local retrieval-augmented generation (RAG) workflow for document processing and embedding creation, followed by search and answer functionality. The process begins with a collection of documents, such as PDFs or a 1200-page nutrition textbook, which are preprocessed into smaller chunks, for example, groups of 10 sentences each. These chunks are used as context for the Large Language Model (LLM). A cool person (potentially the user) asks a query such as "What are the macronutrients? And what do they do?" This query is then transformed by an embedding model into a numerical representation using sentence transformers or other options from Hugging Face, which are stored in a torch.tensor format for efficiency, especially with large numbers of embeddings (around 100k+). For extremely large datasets, a vector database/index may be used. The numerical query and relevant document passages are processed on a local GPU, specifically an RTX 4090. The LLM generates output based on the context related to the query, which can be interacted with through an optional chat web app interface. All of this processing happens on a local GPU. The flowchart includes icons for documents, processing steps, and hardware, with arrows indicating the flow from document collection to user interaction with the generated text and resources

All designed to run locally on a NVIDIA GPU.

All the way from PDF ingestion to "chat with PDF" style features.

All using open-source tools.

In our specific example, we'll build NutriChat, a RAG workflow that allows a person to query a 1200 page PDF version of a Nutrition Textbook and have an LLM generate responses back to the query based on passages of text from the textbook.

PDF source: https://pressbooks.oer.hawaii.edu/humannutrition2/ 



## What is RAG?

RAG stands for Retrieval Augmented Generation.

It was introduced in the paper [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401).

Each step can be roughly broken down to:

* **Retrieval** - Seeking relevant information from a source given a query. For example, getting relevant passages of Wikipedia text from a database given a question.
* **Augmented** - Using the relevant retrieved information to modify an input to a generative model (e.g. an LLM).
* **Generation** - Generating an output given an input. For example, in the case of an LLM, generating a passage of text given an input prompt.



