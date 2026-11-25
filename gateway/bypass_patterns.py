"""Word patterns for the skip rules (design §5.3).
These lists are the only place the patterns live, so tuning them never touches the logic in rules.py."""

import re

# Prompts whose right answer changes over time.
# "current" is deliberately NOT listed: "change my current password" is a generic question.
TIME_SENSITIVE = [
    re.compile(r"\b(today|tonight|tomorrow|yesterday|now|latest)\b", re.IGNORECASE),
    re.compile(r"\bthis (week|month|year)\b", re.IGNORECASE),  # "deals this week"
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b", re.IGNORECASE),  # ISO date: 2026-12-01
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.IGNORECASE),  # dd/mm/yyyy style: 01/12/2026
]

# Prompts about one specific thing: a cached answer about order #48213 is wrong for #55190.
IDENTIFIERS = [
    re.compile(r"#\s?\d{3,}", re.IGNORECASE),  # #48213
    re.compile(r"\b[a-z]{1,5}-?\d{4,}\b", re.IGNORECASE),  # ORD-99812, AB1234
    # A keyword then a number: "my ticket 482", "tracking number 123456789"
    re.compile(r"\b(order|ticket|booking|invoice|tracking|account)\b\D{0,12}\d{3,}", re.IGNORECASE),
    re.compile(r"\b\d{5,}\b", re.IGNORECASE),  # long bare number: 9876543
    re.compile(r"\+?\d[\d\s-]{8,}\d", re.IGNORECASE),  # phone: +91 98765 43210
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+", re.IGNORECASE),  # email: a@b.com
    re.compile(r"\{\{.*?\}\}", re.IGNORECASE),  # unfilled template placeholder: {{Order Number}}
]
