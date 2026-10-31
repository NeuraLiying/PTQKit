#!/usr/bin/env python3
"""Fail if secrets, absolute home paths, private IPs, or weight files appear in first-party tree."""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {".git", "third_party", ".qs_cache", "__pycache__", ".venv", "venv", ".ruff_cache"}
SKIP_FILES = {"scrub_check.py", "scrub_check.sh", "scrub_token.re"}

# Hugging Face user access tokens (public prefix).
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")

# Absolute home / user checkout paths (generic — no hostnames or usernames listed).
HOME_PATH_RES = [
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/root/[A-Za-z0-9._-]+/"),
]

# RFC1918 / CGNAT-looking IPv4. Skip pip pins like nvidia-curand-cu12==10.3.7.77
# (negative lookbehind for "==" — fixed-width, RE2/Python OK).
PRIVATE_IP_RE = re.compile(
    r"(?<!==)\b(?:"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r"|100\.(?:6[4-9]|[7-9]\d|1[0-1]\d|12[0-7])\.\d{1,3}\.\d{1,3}"
    r")\b"
)

# Non-default SSH port invocations often embed lab bastion details.
SSH_PORT_RE = re.compile(r"ssh\s+-p\s+\d{2,5}\s+")

CODE_EXT = {
    ".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt", ".md",
    ".ini", ".css", ".html", ".js", ".ts", ".tsx", ".jsx", ".diff",
}
WEIGHT_EXT = {".pth", ".pt", ".dat", ".gguf"}

fail = False
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        if f in SKIP_FILES:
            continue
        p = os.path.join(dirpath, f)
        rel = os.path.relpath(p, ROOT)
        ext = os.path.splitext(f)[1].lower()
        if ext in WEIGHT_EXT:
            print("WEIGHT", rel)
            fail = True
            continue
        if ext not in CODE_EXT:
            continue
        try:
            text = open(p, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if TOKEN_RE.search(text):
            print("TOKEN", rel)
            fail = True
        for rx in HOME_PATH_RES:
            if rx.search(text):
                print("HOME_PATH", rel)
                fail = True
                break
        if PRIVATE_IP_RE.search(text):
            print("PRIVATE_IP", rel)
            fail = True
        if SSH_PORT_RE.search(text):
            print("SSH_PORT", rel)
            fail = True

if fail:
    print("FAIL")
    sys.exit(1)
print("scrub_check PASS")
