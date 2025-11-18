#!/usr/bin/env python3
import argparse
import json
import sys
import uuid

def load_words(path):
    seen = set()
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            w = raw.strip()
            if not w:
                continue
            if w not in seen:
                seen.add(w)
                words.append(w)
    return words

def build_items(words, value):
    return [
        {
            "id": str(uuid.uuid4()),
            "text": w,
            "value": value
        }
        for w in words
    ]

def main():
    ap = argparse.ArgumentParser(description="Convert newline-separated words to SillyTavern bias JSON.")
    ap.add_argument("infile", help="Input text file (one word per line)")
    ap.add_argument("-o", "--out", help="Output JSON file (defaults to stdout)")
    ap.add_argument("--value", type=int, default=10, help="Bias value (percentage), default 10")
    args = ap.parse_args()

    words = load_words(args.infile)
    items = build_items(words, args.value)

    out_text = json.dumps(items, ensure_ascii=False, indent=4)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
    else:
        sys.stdout.write(out_text + "\n")

if __name__ == "__main__":
    main()