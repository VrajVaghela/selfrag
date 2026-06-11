# Self-RAG - Self-Reflective Retrieval-Augmented Generation

Advanced RAG implementation featuring self-reflection mechanisms. Self-RAG enables language models to assess their own performance, determine when to retrieve additional information, and improve answers through iterative refinement.

## Overview

Self-RAG improves upon traditional RAG by:
- **Self-Assessment**: Models evaluate their own output quality
- **Adaptive Retrieval**: Retrieves documents only when necessary
- **Iterative Refinement**: Refines answers based on self-assessment
- **Relevance Scoring**: Models score retrieved documents themselves
- **Confidence Estimation**: Provides confidence measures for answers
- **Error Detection**: Identifies and flags potential inaccuracies

## Key Features

- **PDF Document Processing**: Multi-document PDF ingestion and processing
- **Semantic Chunking**: Intelligent splitting with context preservation (600 tokens, 150 overlap)
- **FAISS Vector Store**: Fast similarity search over document embeddings
- **HuggingFace Embeddings**: High-quality semantic representations
- **Reflection Prompts**: Self-evaluation mechanisms in generation
- **Adaptive Thresholds**: Dynamic decision-making for when to retrieve
- **LangGraph State Management**: Robust workflow orchestration

## Project Structure

```
selfrag/
├── selfrag.py            # Core Self-RAG implementation
├── documents/            # PDF documents directory
│   ├── Company_policies.pdf
│   ├── Company_profile.pdf
│   └── Product_and_pricing.pdf
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── vector_store/         # FAISS vector store (generated)
```

## Getting Started

### Prerequisites

- Python 3.10+
- PDFs in `documents/` directory
- HuggingFace API token (free tier available)
- OpenAI or HuggingFace LLM access

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create documents directory
mkdir -p documents

# Add your PDF files to documents/
```

### Configuration

Create a `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key  # If using OpenAI
```

## How Self-RAG Works

### 1. Document Processing & Chunking

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = (
    PyPDFLoader("./documents/Company_policies.pdf").load()
    + PyPDFLoader("./documents/Company_profile.pdf").load()
    + PyPDFLoader("./documents/Product_and_pricing.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=600,      # Smaller chunks for precise matching
    chunk_overlap=150    # Context overlap
).split_documents(docs)
```

### 2. Vector Store Creation

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}  # Retrieve top-4 relevant chunks
)
```

### 3. Self-Reflection Mechanism

The model performs self-assessment on multiple dimensions:

```
Question: "What is our product pricing?"

Initial Response: "We offer three pricing tiers..."

Self-Reflection:
├─ Relevance Check: "Does this answer the question?" → Yes (0.95)
├─ Completeness Check: "Is more information needed?" → No (0.92)
├─ Accuracy Check: "Am I confident in this answer?" → Yes (0.88)
└─ Retrieval Need: "Should I retrieve documents?" → Yes (0.75)
```

### 4. Adaptive Retrieval

Based on self-assessment scores:

```python
if reflection_score < 0.7:
    # Confidence too low - retrieve more documents
    additional_docs = retriever.get_relevant_documents(query)
    refined_answer = generate_with_docs(query, additional_docs)
else:
    # Confidence sufficient - use current answer
    return current_answer
```

### 5. Iterative Refinement

```python
answer = initial_response
iteration = 0

while iteration < max_iterations:
    # Self-assess current answer
    reflection = self_assess(answer, question)
    
    if reflection['needs_refinement']:
        # Retrieve additional context
        docs = retriever.get_relevant_documents(question)
        # Generate improved answer
        answer = refine_answer(question, answer, docs)
        iteration += 1
    else:
        break

return {
    "answer": answer,
    "iterations": iteration,
    "confidence": reflection['confidence']
}
```

## Self-Reflection Scores

The system generates four key reflection metrics:

### 1. **Relevance** (0-1)
How well does the response address the query?

### 2. **Completeness** (0-1)
Does the response fully cover the question scope?

### 3. **Accuracy** (0-1)
How confident are we in the factual correctness?

### 4. **Retrieval Necessity** (0-1)
Should we retrieve documents to improve the answer?

## Technologies

- **LangChain**: RAG pipeline and tool chains
- **LangGraph**: State-managed workflow orchestration
- **FAISS**: Vector similarity search
- **HuggingFace**: Embeddings and language models
- **Pydantic**: Data validation and typing
- **PyPDF**: PDF document loading

## Architecture

```
User Query
    ↓
[Initial LLM Response]
    ↓
[Self-Assessment Module]
    ├─ Relevance Score
    ├─ Completeness Score
    ├─ Accuracy Score
    └─ Retrieval Need Score
    ↓
[Decision Logic]
    ├─ If Score < Threshold → Retrieve & Refine
    └─ If Score ≥ Threshold → Return Answer
    ↓
[Refined Response]
    ↓
[Final Output with Confidence]
```

## Usage Example

```python
from selfrag import SelfRAG

# Initialize Self-RAG
rag = SelfRAG()

# Query with self-reflection
result = rag.process_query({
    "question": "What are our company policies on remote work?",
    "max_iterations": 3,
    "confidence_threshold": 0.85
})

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Refinement Iterations: {result['iterations']}")
```

## Configuration Parameters

### Document Processing

```python
chunk_size = 600              # Smaller chunks = more precise matching
chunk_overlap = 150           # Context preservation overlap
min_chunk_length = 100        # Skip very small chunks
```

### Reflection Thresholds

```python
confidence_threshold = 0.75   # Trigger retrieval if below
relevance_threshold = 0.70    # Minimum relevance score
completeness_threshold = 0.75 # Minimum completeness score
```

### Retrieval Parameters

```python
k = 4                         # Documents to retrieve
search_type = "similarity"    # Or "mmr" for diversity
max_iterations = 3            # Max refinement loops
```

## Performance Optimization

### Embedding Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    return embeddings.embed_query(text)
```

### Batch Processing

```python
queries = [q1, q2, q3]
results = [rag.process_query(q) for q in queries]
```

### Index Optimization

For large document sets (>10k chunks):

```python
# Use more efficient FAISS index
index = faiss.IndexIVFFlat(
    faiss.IndexFlatL2(dim),
    nlist=100
)
```

## Evaluation Metrics

Measure Self-RAG effectiveness:

- **Answer Quality**: Human evaluation of response accuracy
- **Efficiency**: Number of retrievals vs. answer quality
- **Confidence Calibration**: How well confidence scores match actual accuracy
- **Hallucination Rate**: Percentage of unsupported claims
- **Refinement Effectiveness**: Quality improvement per iteration

## Troubleshooting

### Low Confidence Scores

1. Improve document quality and relevance
2. Adjust chunking parameters (smaller chunks = better matching)
3. Use stronger embedding models
4. Provide more context in prompts

### Excessive Refinement Loops

1. Increase confidence thresholds
2. Reduce max_iterations
3. Improve initial prompt quality
4. Verify document relevance

### Slow Performance

1. Use GPU acceleration for embeddings
2. Implement caching layers
3. Optimize FAISS index for your data size
4. Consider batch processing

## Advanced Features

### Custom Reflection Prompts

```python
reflection_prompt = """
Evaluate this response:
{response}

Score on these dimensions:
1. Factual Accuracy (0-1)
2. Completeness (0-1)
3. Clarity (0-1)
4. Should retrieve more docs? (yes/no)
"""
```

### Confidence Estimation

```python
confidence = (
    0.3 * relevance_score +
    0.3 * completeness_score +
    0.4 * accuracy_score
)
```

### Multi-Turn Conversations

Maintain conversation history for context-aware refinement:

```python
conversation_history = []
for turn in user_queries:
    result = rag.process_query(
        turn,
        context=conversation_history
    )
    conversation_history.append(result)
```

## License

MIT

## Contributing

Contributions welcome! Focus areas:
- Improved reflection mechanisms
- Advanced confidence calibration
- Multi-modal document support
- Performance optimization
- Additional embedding models
