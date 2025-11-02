#!/usr/bin/env python3
"""
Example 8: Long Context Handling for Documents >2048 Tokens
Strategies for processing long documents in vLLM

Techniques:
- Chunking with overlap
- Sliding window
- Map-reduce pattern
- Hierarchical summarization
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chunk:
    """Document chunk with metadata"""
    text: str
    index: int
    start_pos: int
    end_pos: int
    overlap_prev: int = 0
    overlap_next: int = 0


def chunk_with_overlap(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200
) -> List[Chunk]:
    """
    Chunk document with overlapping windows

    Args:
        text: Full document text
        chunk_size: Tokens per chunk (leave room for prompt)
        overlap: Overlapping tokens between chunks

    Returns:
        List of chunks with metadata
    """
    # Simplified: using characters as proxy for tokens (4 chars ≈ 1 token)
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + char_chunk_size, len(text))
        chunk_text = text[start:end]

        chunk = Chunk(
            text=chunk_text,
            index=index,
            start_pos=start,
            end_pos=end,
            overlap_prev=char_overlap if index > 0 else 0,
            overlap_next=char_overlap if end < len(text) else 0,
        )

        chunks.append(chunk)
        start = end - char_overlap  # Move forward with overlap
        index += 1

    return chunks


def chunking_strategy_example():
    """Example: Chunking strategy for long documents"""
    print(f"\n{'='*80}")
    print(f"Strategy 1: Chunking with Overlap")
    print(f"{'='*80}\n")

    # Simulate long document
    long_document = """
    Financial Report Q4 2024...
    """ * 500  # ~1500 words, ~2000 tokens

    print(f"Document size: ~{len(long_document)} characters (~{len(long_document) // 4} tokens)")

    # Chunk it
    chunks = chunk_with_overlap(long_document, chunk_size=1500, overlap=200)

    print(f"Created {len(chunks)} chunks with 200-token overlap\n")

    # Process each chunk
    print("Processing chunks:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"  Chunk {i + 1}: {len(chunk.text) // 4} tokens (pos {chunk.start_pos // 4}-{chunk.end_pos // 4})")

    print(f"  ...")
    print(f"\n💡 Use overlap to maintain context between chunks")


def sliding_window_example():
    """Example: Sliding window for continuous processing"""
    print(f"\n{'='*80}")
    print(f"Strategy 2: Sliding Window")
    print(f"{'='*80}\n")

    print(f"Use case: Real-time document analysis\n")

    code_example = '''
def process_with_sliding_window(document: str, window_size: int = 1500):
    """
    Process document with sliding window

    Useful for:
    - Finding patterns across document
    - Continuous analysis
    - Maintaining full context
    """
    stride = window_size // 2  # 50% overlap

    results = []
    for start in range(0, len(document), stride):
        window = document[start:start + window_size]

        # Process window
        result = llm.generate(
            f"Analyze this section:\\n{window}",
            max_tokens=200
        )

        results.append({
            "position": start,
            "analysis": result
        })

    return combine_results(results)
'''

    print(code_example)
    print(f"\n💡 Sliding window maintains context but increases cost (more overlapping processing)")


def map_reduce_example():
    """Example: Map-reduce for long documents"""
    print(f"\n{'='*80}")
    print(f"Strategy 3: Map-Reduce Pattern")
    print(f"{'='*80}\n")

    print(f"Process: Chunk → Summarize Each → Combine Summaries\n")

    code_example = '''
def map_reduce_summarization(document: str, chunk_size: int = 1500):
    """
    Map-reduce pattern for long document summarization

    Step 1 (Map): Summarize each chunk independently
    Step 2 (Reduce): Combine summaries into final summary
    """

    # Step 1: Map - Process each chunk
    chunks = chunk_with_overlap(document, chunk_size)

    chunk_summaries = []
    for chunk in chunks:
        summary = llm.generate(
            f"Summarize this section concisely:\\n{chunk.text}",
            max_tokens=150
        )
        chunk_summaries.append(summary)

    print(f"Created {len(chunk_summaries)} chunk summaries")

    # Step 2: Reduce - Combine summaries
    combined = "\\n\\n".join(chunk_summaries)

    final_summary = llm.generate(
        f"Create a comprehensive summary from these section summaries:\\n{combined}",
        max_tokens=500
    )

    return final_summary


# Example usage
long_doc = load_10k_filing()  # 50K tokens
summary = map_reduce_summarization(long_doc, chunk_size=1500)
print(summary)
'''

    print(code_example)
    print(f"\n💡 Most cost-effective for summarization tasks")
    print(f"   Each chunk processed once, then combined")


def hierarchical_summarization_example():
    """Example: Hierarchical summarization for very long documents"""
    print(f"\n{'='*80}")
    print(f"Strategy 4: Hierarchical Summarization")
    print(f"{'='*80}\n")

    print(f"For documents >100K tokens\n")

    print(f"Level 1: Chunk into 20 sections → 20 summaries")
    print(f"Level 2: Chunk 20 summaries into 4 groups → 4 meta-summaries")
    print(f"Level 3: Combine 4 meta-summaries → 1 final summary")

    code_example = '''
def hierarchical_summarize(document: str, levels: int = 3):
    """
    Multi-level hierarchical summarization

    Example: 100K token document
    - Level 1: 100K → 20 chunks → 20 x 200 token summaries = 4K tokens
    - Level 2: 4K → 4 chunks → 4 x 200 token summaries = 800 tokens
    - Level 3: 800 → 1 final summary = 200 tokens
    """
    current_text = document

    for level in range(levels):
        # Chunk current text
        chunks = chunk_with_overlap(current_text, chunk_size=1500)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = llm.generate(
                f"Concise summary (level {level + 1}):\\n{chunk.text}",
                max_tokens=200
            )
            summaries.append(summary)

        # Next level input = current summaries combined
        current_text = "\\n\\n".join(summaries)

        print(f"Level {level + 1}: {len(chunks)} chunks → {len(summaries)} summaries")

        # Stop if small enough
        if len(current_text) < 2000:  # ~500 tokens
            break

    return current_text
'''

    print(code_example)
    print(f"\n💡 Scales to arbitrarily long documents")
    print(f"   Each level reduces size by ~5-10x")


def rag_long_context_example():
    """Example: RAG with long retrieved context"""
    print(f"\n{'='*80}")
    print(f"Strategy 5: RAG with Context Compression")
    print(f"{'='*80}\n")

    print(f"Problem: Retrieved 10 relevant documents = 15K tokens\n")
    print(f"Solution: Extract only relevant passages\n")

    code_example = '''
def rag_with_compression(query: str, documents: List[str]):
    """
    RAG with automatic context compression

    Steps:
    1. Retrieve relevant documents
    2. Extract only query-relevant passages
    3. Combine compressed context
    4. Generate answer
    """

    # Step 1: Retrieve documents (already done)
    # documents = vector_search(query)  # Returns 10 docs

    # Step 2: Extract relevant passages from each doc
    relevant_passages = []
    for doc in documents:
        # Ask LLM to extract relevant part
        passage = llm.generate(
            f"Extract ONLY the passage relevant to: '{query}'\\n\\nDocument:\\n{doc[:2000]}",
            max_tokens=200
        )
        relevant_passages.append(passage)

    # Step 3: Combine compressed context
    context = "\\n\\n".join(relevant_passages)  # Now ~2K tokens instead of 15K

    # Step 4: Generate answer
    answer = llm.generate(
        f"Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:",
        max_tokens=300
    )

    return answer
'''

    print(code_example)
    print(f"\n💡 Reduces context from 15K to 2K tokens (87% reduction)")
    print(f"   Uses LLM to extract only relevant information")


def cost_comparison():
    """Compare costs of different strategies"""
    print(f"\n{'='*80}")
    print(f"💰 Cost Comparison (50K token document)")
    print(f"{'='*80}\n")

    doc_size = 50000
    cost_per_1m = 0.10

    strategies = [
        ("Chunking (no overlap)", doc_size, "1x"),
        ("Chunking (20% overlap)", doc_size * 1.2, "1.2x"),
        ("Map-reduce (2-level)", doc_size * 1.1, "1.1x"),
        ("Hierarchical (3-level)", doc_size * 1.05, "1.05x"),
        ("RAG compression", doc_size * 0.3, "0.3x"),
    ]

    print(f"{'Strategy':<30} {'Tokens Processed':<20} {'Cost':<15} {'vs Baseline'}")
    print(f"{'-'*30} {'-'*20} {'-'*15} {'-'*10}")

    for name, tokens, multiplier in strategies:
        cost = (tokens / 1_000_000) * cost_per_1m
        print(f"{name:<30} {tokens:>15,}{'':<5} ${cost:>10.6f}    {multiplier}")

    print(f"\n💡 Choose based on use case:")
    print(f"   - Summarization: Map-reduce or hierarchical")
    print(f"   - Q&A: RAG with compression")
    print(f"   - Analysis: Chunking with overlap")


def main():
    """Run long context handling examples"""
    print(f"{'='*80}")
    print(f"Long Context Handling Strategies")
    print(f"{'='*80}\n")

    print(f"Problem: vLLM max context = 2048 tokens")
    print(f"Solution: Multiple strategies for different use cases\n")

    chunking_strategy_example()
    sliding_window_example()
    map_reduce_example()
    hierarchical_summarization_example()
    rag_long_context_example()
    cost_comparison()

    print(f"\n{'='*80}")
    print(f"✅ Summary")
    print(f"{'='*80}")
    print(f"\n1. Chunking - Simple, works for most cases")
    print(f"2. Sliding window - Continuous analysis")
    print(f"3. Map-reduce - Best for summarization")
    print(f"4. Hierarchical - Scales to 100K+ tokens")
    print(f"5. RAG compression - Most cost-effective for Q&A")
    print(f"\n📊 Recommendations:")
    print(f"   Use case          → Strategy")
    print(f"   Summarization     → Map-reduce or hierarchical")
    print(f"   Q&A               → RAG with compression")
    print(f"   Analysis          → Chunking with 200-token overlap")
    print(f"   Pattern finding   → Sliding window")
    print(f"\n💰 Cost Impact:")
    print(f"   Without strategy: Can't process >2K tokens")
    print(f"   With strategy: Handle 100K+ tokens efficiently")


if __name__ == "__main__":
    main()
