import json
import argparse
from nltk.tokenize import sent_tokenize
import nltk
from preprocess import extract_text

nltk.download('punkt_tab')

def sentence_chunks(text: str, max_sentences: int = 5):
    sentences = sent_tokenize(text)
    for i in range(0, len(sentences), max_sentences):
        yield " ".join(sentences[i : i + max_sentences])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=str,
        default="/home/asif/Documents/rag_project/data/HSC26_Bangla_1st_paper.pdf",
        help="Path to PDF corpus"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="How many sentences per chunk"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/asif/Documents/rag_project/vector_store/chunks.json",
        help="Where to write chunk metadata"
    )
    args = parser.parse_args()

    text = extract_text(args.pdf)
    chunks = [
        {"id": idx, "text": chunk}
        for idx, chunk in enumerate(sentence_chunks(text, args.max_sentences))
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(chunks)} chunks to {args.output}")


