import math
import nltk
from app.ingestion.embedder import embed_text

SEMANTIC_THRESHOLD = 800
SIMILARITY_THRESHOLD = 0.75

def _count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def _semantic_split(sentences: list[str], section_metadata: dict) -> list[dict]:
    print(f"Semantic chunking section '{section_metadata['section_title']}': {len(sentences)} sentences...")
    if not sentences:
        return []
        
    embeddings = [embed_text(s) for s in sentences]
    
    chunks = []
    current_chunk_sentences = []
    
    for i in range(len(sentences)):
        current_chunk_sentences.append(sentences[i])
        
        # Check if we should split after this sentence
        if i < len(sentences) - 1:
            sim = _cosine_similarity(embeddings[i], embeddings[i+1])
            # Split if similarity drops below threshold AND we have at least 3 sentences
            if sim < SIMILARITY_THRESHOLD and len(current_chunk_sentences) >= 3:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                
    if current_chunk_sentences:
        # Minimum chunk size: 3 sentences (merge tiny chunks with their neighbor)
        if len(current_chunk_sentences) < 3 and len(chunks) > 0:
            chunks[-1] += " " + " ".join(current_chunk_sentences)
        else:
            chunks.append(" ".join(current_chunk_sentences))
            
    final_chunk_dicts = []
    for chunk_text in chunks:
        words = chunk_text.split()
        if len(words) > SEMANTIC_THRESHOLD:
            # apply a hard split at 800 words with 80 word overlap
            start = 0
            while start < len(words):
                end = min(start + SEMANTIC_THRESHOLD, len(words))
                part_text = " ".join(words[start:end])
                
                chunk_dict = section_metadata.copy()
                chunk_dict["content"] = part_text
                chunk_dict["chunking_method"] = "hard_split"
                final_chunk_dicts.append(chunk_dict)
                
                start += SEMANTIC_THRESHOLD - 80 # 80 overlap
                
                if start >= len(words):
                    break
        else:
            chunk_dict = section_metadata.copy()
            chunk_dict["content"] = chunk_text
            chunk_dict["chunking_method"] = "semantic"
            final_chunk_dicts.append(chunk_dict)
            
    return final_chunk_dicts

def chunk_elements(elements: list[dict]) -> list[dict]:
    # Ensure punkt tokenizer is downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    final_chunks = []
    chunk_counter = 1
    
    # Step 1: group text elements by section_title to form section blocks
    section_blocks = {}
    
    for el in elements:
        t = el.get("type")
        if t in ["table", "figure"]:
            # LAYER 1 & 2: Tables and Figures never split
            chunk = el.copy()
            chunk["chunk_id"] = f"{el.get('source_filename', 'unknown')}_page{el.get('page_number', 1)}_{t}_chunk{chunk_counter}"
            chunk["chunking_method"] = "no_split"
            final_chunks.append(chunk)
            chunk_counter += 1
        elif t == "text":
            sec_title = el.get("section_title", "Unknown")
            if sec_title not in section_blocks:
                section_blocks[sec_title] = {
                    "texts": [],
                    "metadata": {
                        "type": "text",
                        "image_path": None,
                        "page_number": el.get("page_number", 1),
                        "section_title": sec_title,
                        "source_filename": el.get("source_filename", "unknown")
                    }
                }
            section_blocks[sec_title]["texts"].append(el.get("content", ""))
            
    # Process the grouped text blocks
    for sec_title, block in section_blocks.items():
        combined_text = "\n".join(block["texts"]).strip()
        if not combined_text:
            continue
            
        word_count = _count_words(combined_text)
        metadata = block["metadata"]
        
        # Step 3a: if word count is under 800
        if word_count < SEMANTIC_THRESHOLD:
            chunk = metadata.copy()
            chunk["content"] = combined_text
            chunk["chunk_id"] = f"{metadata['source_filename']}_page{metadata['page_number']}_text_chunk{chunk_counter}"
            chunk["chunking_method"] = "no_split"
            final_chunks.append(chunk)
            chunk_counter += 1
        else:
            # Step 3b: semantic chunking
            sentences = nltk.sent_tokenize(combined_text)
            semantic_chunks = _semantic_split(sentences, metadata)
            for sc in semantic_chunks:
                sc["chunk_id"] = f"{metadata['source_filename']}_page{metadata['page_number']}_text_chunk{chunk_counter}"
                final_chunks.append(sc)
                chunk_counter += 1
                
    return final_chunks

def chunk_pdf(pdf_path: str) -> list[dict]:
    from app.ingestion.pdf_parser import parse_pdf
    elements = parse_pdf(pdf_path)
    return chunk_elements(elements)

if __name__ == "__main__":
    try:
        chunks = chunk_pdf("test.pdf")
        print(f"Total chunks: {len(chunks)}")
        methods = {}
        for c in chunks:
            methods[c["chunking_method"]] = methods.get(c["chunking_method"], 0) + 1
        print(f"Chunking methods used: {methods}")
        for c in chunks[:5]:
            print(c["type"], c["chunk_id"], c["chunking_method"], c["content"][:60])
    except Exception as e:
        print(f"Error testing chunker: {e}")
