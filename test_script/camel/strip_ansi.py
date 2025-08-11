import re
import json
import argparse
import os

def strip_ansi(text: str) -> str:
    """
    Strip all ANSI escape sequences, including CSI, OSC, and terminal control codes.
    """
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # ANSI escape sequences
            [@-Z\\-_] |          # 7-bit C1 control codes
            \[ [0-?]* [ -/]* [@-~] |  # CSI sequences
            \] .*? (?:\x07|\x1b\\)    # OSC (Operating System Command)
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', text)

def is_cast_file(first_line: str) -> bool:
    """Check if the file looks like a .cast (asciinema) JSON file."""
    try:
        obj = json.loads(first_line)
        return isinstance(obj, dict) and obj.get("version") == 2
    except Exception:
        return False

def clean_text_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned = strip_ansi(content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    print(f"✅ Cleaned plain text saved to: {output_path}")

def clean_cast_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for i, line in enumerate(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping invalid JSON line {i}: {line.strip()}")
            continue

        if isinstance(parsed, list) and len(parsed) == 3 and parsed[1] == "o":
            unescaped = parsed[2].encode('utf-8').decode('unicode_escape')
            parsed[2] = strip_ansi(unescaped)

        cleaned_lines.append(json.dumps(parsed))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(cleaned_lines))

    print(f"✅ Cleaned cast file saved to: {output_path}")

def get_default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_cleaned{ext}"

def main():
    parser = argparse.ArgumentParser(description="Strip ANSI escape sequences from plain text or .cast files.")
    parser.add_argument("-i", help="Input file path (.log, .txt, or .cast)")
    parser.add_argument("-o", "--output", default=None, required=False, help="Optional output file path")

    args = parser.parse_args()
    input_path = args.i
    output_path = args.output or f"{args.i}.strip"

    if not os.path.isfile(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    if is_cast_file(first_line):
        clean_cast_file(input_path, output_path)
    else:
        clean_text_file(input_path, output_path)

if __name__ == "__main__":
    main()
