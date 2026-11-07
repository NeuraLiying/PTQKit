import json
import sys
import re

# Common Unicode zero-width characters
ZERO_WIDTH = (
    "\u200b"  # zero width space
    "\u200c"  # zero width non-joiner
    "\u200d"  # zero width joiner
    "\ufeff"  # BOM
)

# C0 controls (keep \t \n \r for an initial replace pass)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Collapse consecutive whitespace
WS_RE = re.compile(r"\s+")

def clean_text(txt: str) -> str:
    if not txt:
        return ""

    # Normalize newlines and tabs to ordinary spaces
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = txt.replace("\t", " ")

    # Strip zero-width characters
    for ch in ZERO_WIDTH:
        txt = txt.replace(ch, "")

    # Strip other invisible control characters
    txt = CTRL_RE.sub("", txt)

    # Flatten in-sample newlines to spaces so each sample is one line
    txt = txt.replace("\n", " ")

    # Collapse consecutive whitespace
    txt = WS_RE.sub(" ", txt).strip()

    return txt

def main():
    if len(sys.argv) != 3:
        print("Usage: python jsonl_to_txt_clean.py <in.jsonl> <out.txt>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    kept = 0
    skipped = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue

            txt = obj.get("text", "")
            txt = clean_text(txt)
            if not txt:
                skipped += 1
                continue

            # One sample per line; blank line between samples for inspection and clear boundaries
            fout.write(txt + "\n\n")
            kept += 1

    print(f"kept={kept}, skipped={skipped}")

if __name__ == "__main__":
    main()
