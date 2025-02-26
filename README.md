
# Optimizing Ad Placement: A Multi-Faceted Approach with Web Scraping and RAG-LLM

# Overview

This project introduces an AI-driven **one-to-one advertising system** that combines **Web Scraping, NLP, LLMs, and Retrieval Augmented Generation (RAG)** to optimize ad placements. The addition of RAG enhances personalization by retrieving real-time contextual data before ad generation. This approach improves ad relevance, engagement, and privacy-conscious personalization, offering a robust framework for next-generation digital advertising. Below is an overview of the analysis, along with sample outputs and results. This project was done in May' 2024.





## Publication

- This paper was presented in the “2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT)”
- This paper is yet to be published by IEEE.


## Block Diagram

- The below block diagram gives an overview of the overall funtionality of the implemented project

 <p align="center">
  <img src="https://i.postimg.cc/ydhZ7sDH/x48.png" alt="App Screenshot" width="500">
</p>


## Features

- **Web Scraping & Data Extraction**: The system leverages web scraping and APIs to extract real-time data from websites, enabling personalized advertisement recommendations based on user behavior and market trends.

- **Retrieval-Augmented Generation (RAG) for Contextual Ads**: By integrating RAG with LLMs, the model retrieves relevant documents from external sources (PDFs, Wikipedia, and scraped data) to generate precise and context-aware ad strategies.

    OpenAI's GPT-4o LLM was used. Other LLMs such as Google's Gemini, Cohere and Llama3 were also tested but OpenAI LLM worked better for this particular application. 
     <p align="center">
  <img src="https://i.postimg.cc/hG86tvN2/hyt.png" alt="App Screenshot" width="300">
</p>



- **AI-Generated Marketing Campaigns**: The framework produces detailed advertisement strategies along with AI-generated images using DALL·E, enhancing user engagement through visually appealing content.
    
    An ad recommendation strategy suggested by the proposed model
<p align="center">
  <img src="https://i.postimg.cc/QCHRhWwv/ffd.png" alt="App Screenshot" width="30%" style="margin-right: 20px;">
  <img src="https://i.postimg.cc/q79xF601/f1.png" alt="App Screenshot" width="30%" style="margin-right: 10px;">
  <img src="https://i.postimg.cc/q79xF601/f1.png" alt="App Screenshot" width="30%">
</p>






## Tech Stack

1. Web Scraping & Data Extraction
- WikipediaAPI – To fetch advertising-related data from Wikipedia.
- WebBaseLoader – For crawling and converting website content into structured text data.
- PyPDFDirectoryLoader – To extract and process text from advertising strategy PDFs.

2. Retrieval-Augmented Generation (RAG) Framework
- LangChain – The backbone of the RAG implementation, integrating various tools and models.
- FAISS (Facebook AI Similarity Search) – To store and retrieve embeddings efficiently.
- OpenAI Embeddings – For creating vectorized text representations.
- DALL·E – To generate AI-driven advertisement images based on marketing strategies.

3. Front End: 
- Steamlit : frontend interface for allowing users to input website URLs, specify advertisement preferences, and visualize the generated ad strategies and AI-generated images.
## Installation

1. Update File Paths – Ensure all file paths in the scripts match your local system.

2. Run `code.ipynb` – This file integrates all the LangChain models and serves as the core of the project.

3. Execute `retriever.ipynb` – Implements the RAG architecture using custom PDFs stored in the `/files` directory. You can replace these PDFs as needed.

4. Run `results.ipynb` – Compares different LLM models, including Gemini, Cohere, and Llama 3, to analyze their performance.

5. Launch `frontend.py` – This file runs the Streamlit-based frontend, providing an interface for users to prompt the model and generate results.

6. Prerequisites:
- Python
- LangChain
- API keys for LLM models: GPT-4o, Google Gemini, Cohere, and Llama 3







## Running Tests

The project can be implemented and tested to verify funtionality

