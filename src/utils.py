import json
import os

def build_vocabulary(caption_file, output_file, max_words=5000):
    """
    Reads a caption file, tokenizes words, and creates a vocabulary mapping.
    
    Args:
        caption_file (str): Path to the CSV/TXT file containing captions.
        output_file (str): Path where the vocab.json will be saved.
        max_words (int): Maximum number of words to include in vocabulary.
    """
    
    if not os.path.exists(caption_file):
        print(f"[!] Error: {caption_file} not found. Please check your 'data' folder.")
        return

    print(f"[*] Processing: {caption_file}")
    freq = {}

    # Read and count word frequencies
    with open(caption_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # Skip header row
            
            try:
                # Expecting format: image_name,caption
                _, caption = line.strip().split(",", 1)
                for w in caption.lower().split():
                    freq[w] = freq.get(w, 0) + 1
            except ValueError:
                continue # Skip malformed lines

    # Initialize vocabulary with a padding token
    # <pad> is essential for handling variable length sequences in neural networks
    vocab = {"<pad>": 0}

    # Sort words by frequency and take top MAX_WORDS
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])[:max_words]
    
    for w, _ in sorted_words:
        vocab[w] = len(vocab)
        
    # Save to JSON for model training/inference
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print(f"✅ Vocabulary created: {output_file}")
    print(f"[*] Total Tokens: {len(vocab)} (Limit: {max_words + 1})")

if __name__ == "__main__":
    # --- PATH UPDATES FOR NEW STRUCTURE ---
    # Script is in src/, data is in ../data/
    INPUT_PATH = os.path.join("..", "data", "captions.txt")
    OUTPUT_PATH = os.path.join("..", "data", "vocab.json")
    
    build_vocabulary(INPUT_PATH, OUTPUT_PATH)