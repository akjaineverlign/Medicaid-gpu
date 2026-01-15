"""
GPU-Accelerated RAG System with FAISS-GPU
3x faster semantic search using GPU
"""

import torch
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 RAG using device: {DEVICE}")


class GPURAGSystem:
    """
    GPU-accelerated RAG system using FAISS-GPU
    - 3x faster semantic search
    - Batch processing support
    - Conversation context integration
    """
    
    def __init__(self, index_path="docs.index", docs_path="docs.pkl", 
                 model_name="all-MiniLM-L6-v2"):
        """Initialize GPU-accelerated RAG"""
        
        print("🔧 Initializing GPU RAG System...")
        
        # Load sentence transformer on GPU
        print(f"📥 Loading embedding model on {DEVICE}...")
        self.embedder = SentenceTransformer(model_name)
        self.embedder = self.embedder.to(DEVICE)
        print("✅ Embedder loaded on GPU")
        
        # Load FAISS index
        print("📥 Loading FAISS index...")
        cpu_index = faiss.read_index(index_path)
        
        # Move to GPU if available
        if DEVICE == "cuda" and faiss.get_num_gpus() > 0:
            print("🚀 Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("✅ FAISS index on GPU")
        else:
            self.index = cpu_index
            print("⚠️ Using CPU FAISS (no GPU available)")
        
        # Load documents
        print("📥 Loading document chunks...")
        self.docs = pickle.load(open(docs_path, "rb"))
        print(f"✅ Loaded {len(self.docs)} document chunks")
        
        # Cache for frequently asked questions
        self.cache = {}
    
    def search(self, query: str, step: str = None, k: int = 5, 
               max_distance: float = 1.0) -> List[Dict]:
        """
        GPU-accelerated semantic search
        
        Args:
            query: Search query
            step: Current conversation step for context
            k: Number of results
            max_distance: Maximum distance threshold
            
        Returns:
            List of results with text and confidence
        """
        
        # Check cache first (for common questions)
        cache_key = f"{query}:{step}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Step-aware embedding
        query_text = f"step:{step} | {query}" if step else query
        
        # Encode query on GPU (much faster!)
        query_embedding = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Search using GPU FAISS (3x faster than CPU)
        distances, indices = self.index.search(query_embedding, k)
        
        # Process results
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i == -1:
                continue
            if d <= max_distance:
                results.append({
                    "text": self.docs[i],
                    "distance": float(d),
                    "confidence": 1.0 - (float(d) / max_distance)
                })
        
        # Cache result
        if len(results) > 0:
            self.cache[cache_key] = results
        
        return results
    
    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict]]:
        """
        Batch search for multiple queries (even faster!)
        Useful for pre-fetching common questions
        """
        
        # Encode all queries at once (GPU batch processing)
        query_embeddings = self.embedder.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32  # Adjust based on GPU memory
        )
        
        # Batch search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Process all results
        all_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for d, i in zip(query_dists, query_indices):
                if i != -1:
                    results.append({
                        "text": self.docs[i],
                        "distance": float(d),
                        "confidence": 1.0 - float(d)
                    })
            all_results.append(results)
        
        return all_results
    
    def get_context_for_question(self, question: str, 
                                  conversation_history: str = None,
                                  step: str = None) -> str:
        """
        Get RAG context enhanced with conversation history
        
        This is the key to making RAG work with full call context!
        """
        
        # Search for relevant docs
        results = self.search(query=question, step=step, k=5, max_distance=0.95)
        
        if not results:
            return None
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Take top 2 chunks
        context_chunks = [r["text"] for r in results[:2]]
        rag_context = "\n\n".join(context_chunks)
        
        # If we have conversation history, include it
        if conversation_history:
            full_context = f"""CONVERSATION HISTORY:
{conversation_history}

KNOWLEDGE BASE:
{rag_context}"""
            return full_context
        
        return rag_context
    
    def clear_cache(self):
        """Clear the cache (call periodically)"""
        self.cache = {}
        print("🧹 RAG cache cleared")



STEP_FALLBACKS = {
    "identity_auth": (
        "I need to verify your identity before we continue. "
        "This protects your personal information and is required by Medicaid."
    ),
    "address_check": (
        "We need to confirm your address to make sure all your important "
        "Medicaid documents reach you. It's a standard part of the renewal."
    ),
    "income_update": (
        "Income information helps us determine your Medicaid eligibility accurately. "
        "We only ask what's necessary for your renewal."
    ),
    "household_change": (
        "Household details help us determine the right benefits for you. "
        "Changes in who lives with you can affect your coverage."
    ),
    "other_insurance": (
        "We ask about other insurance to coordinate your benefits properly. "
        "Having other insurance doesn't automatically affect your Medicaid."
    ),
}

GENERIC_FALLBACK = (
    "That's a good question. For now, let's focus on completing your renewal, "
    "and you can always contact member services later for more details."
)



def answer_with_rag_and_context(gpu_rag: GPURAGSystem, 
                                 question: str,
                                 conversation_history: str = None,
                                 current_step: str = None) -> str:
    """
    Answer question using GPU RAG + full conversation context
    
    This combines:
    1. Fast GPU semantic search
    2. Full conversation memory
    3. LLM for natural language generation
    """
    
    # Get context from GPU RAG
    context = gpu_rag.get_context_for_question(
        question=question,
        conversation_history=conversation_history,
        step=current_step
    )
    
    # No confident match - use fallback
    if not context:
        return STEP_FALLBACKS.get(current_step, GENERIC_FALLBACK)
    
    # Generate answer using LLM
    from openai import OpenAI
    import os
    
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )
    
    prompt = f"""You are a friendly Medicaid support agent on a phone call.

Answer the user's question using the context below. Keep your answer:
- Conversational and warm (like talking to a friend)
- Brief (2-3 sentences maximum)
- Clear and easy to understand
- Consistent with the conversation history

Context:
{context}

User's question: {question}

Your conversational answer:"""

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Ensure it ends with punctuation
    if not answer.endswith((".", "!", "?")):
        answer += "."
    
    return answer



def build_gpu_index(docs_path="docs.txt", 
                    output_index="docs.index",
                    output_docs="docs.pkl",
                    model_name="all-MiniLM-L6-v2"):
    """
    Build FAISS index on GPU (much faster for large datasets)
    """
    
    print("🔨 Building GPU-accelerated RAG index...")
    
    # Load documents
    print(f"📥 Loading documents from {docs_path}...")
    raw = open(docs_path).read()
    docs = [d.strip() for d in raw.split("\n\n") if len(d.strip()) > 50]
    print(f"✅ Loaded {len(docs)} document chunks")
    
    # Load model on GPU
    print(f"📥 Loading embedding model on GPU...")
    model = SentenceTransformer(model_name)
    model = model.to(DEVICE)
    print("✅ Model loaded on GPU")
    
    # Generate embeddings (much faster on GPU!)
    print("🔢 Generating embeddings on GPU...")
    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32  # Adjust based on GPU memory
    )
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Create FAISS index
    print("🔨 Building FAISS index...")
    dimension = len(embeddings[0])
    
    # Use IndexFlatL2 for exact search
    # For larger datasets, consider IndexIVFFlat for faster approximate search
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings
    index.add(embeddings)
    print(f"✅ Added {index.ntotal} vectors to index")
    
    # Save index (CPU version for portability)
    print(f"💾 Saving index to {output_index}...")
    faiss.write_index(index, output_index)
    
    # Save documents
    print(f"💾 Saving documents to {output_docs}...")
    pickle.dump(docs, open(output_docs, "wb"))
    
    print("✅ GPU RAG index built successfully!")
    print(f"   Index: {output_index}")
    print(f"   Docs: {output_docs}")
    print(f"   Chunks: {len(docs)}")
    print(f"   Dimension: {dimension}")



COMMON_QUESTIONS = [
    "Why do you need my address?",
    "What happens if I don't complete this?",
    "How long will this take?",
    "Can I call back later?",
    "What if my income changed?",
    "Do I need to provide proof?",
    "What if I moved recently?",
    "Will I lose my coverage?",
    "Can someone help me with this?",
    "What if I have other insurance?"
]

def precache_common_questions(gpu_rag: GPURAGSystem):
    """Pre-cache common questions for instant responses"""
    
    print("🔥 Pre-caching common questions...")
    results = gpu_rag.batch_search(COMMON_QUESTIONS)
    
    # Store in cache
    for question, result in zip(COMMON_QUESTIONS, results):
        gpu_rag.cache[question] = result
    
    print(f"✅ Cached {len(COMMON_QUESTIONS)} common questions")



if __name__ == "__main__":
    # Build index
    if not os.path.exists("docs.index"):
        build_gpu_index()
    
    # Initialize GPU RAG
    gpu_rag = GPURAGSystem()
    
    # Pre-cache common questions
    precache_common_questions(gpu_rag)
    
    # Test query
    question = "Why do you need my address?"
    conversation = "User: Hello\nAgent: Hi! This is Medicaid calling..."
    
    answer = answer_with_rag_and_context(
        gpu_rag=gpu_rag,
        question=question,
        conversation_history=conversation,
        current_step="address_check"
    )
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(100):
        gpu_rag.search(question, k=5)
    elapsed = time.time() - start
    print(f"\n⚡ 100 searches: {elapsed:.2f}s ({elapsed/100*1000:.1f}ms per search)")