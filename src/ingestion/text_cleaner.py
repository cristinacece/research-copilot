# -*- coding: utf-8 -*-
"""
Text cleaning utilities for extracted PDF text.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Fix hyphenated line breaks (word- \\nbreak → wordbreak)
    - Normalize whitespace (collapse multiple spaces/newlines)
    - Normalize quotation marks to ASCII
    - Remove isolated page numbers
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Fix hyphenated line-breaks: "hyphen-\nated" → "hyphened"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Remove isolated page numbers (lines that contain only a number)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Normalize fancy quotation marks to ASCII
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')

    # Collapse multiple blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs to a single space (within lines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
