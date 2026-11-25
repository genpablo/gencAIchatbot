"""Guardrails for the customer assistant, which anyone on the internet can talk to.

The product suggestions are already safe by construction: the code only accepts products that passed the
safety, stock, expiry and budget filters, whatever the model says. These guardrails cover the free text:
  looks_like_injection   a message that tries to take over the assistant is answered without calling the model
  guard_answer           sentences that promise deals, quote prices we don't have, or talk about internal data
                         are removed from what the model wrote before a customer sees it
"""
from __future__ import annotations

import re

# "ignore your previous instructions", "show me your system prompt", "you are now DAN", ...
# Narrow on purpose: "can I ignore the instructions on the label?" or "does it act as a primer?" are fine.
INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|of\s+)?(?:the\s+)?"
    r"(?:previous|prior|above|earlier|preceding|your|these|those|system)\s+(?:instructions?|rules|prompts?|guidelines)\b"
    r"|\b(?:system|developer|hidden)\s+(?:prompt|message|instructions?)\b"
    r"|\b(?:reveal|show|print|repeat|leak|tell me|what are)\b.{0,20}\byour\s+(?:instructions|prompt|rules|configuration)\b"
    r"|\byou are now\b|\bjailbreak|\bDAN mode\b|\bdeveloper mode\b|\bpretend (?:to be|you are)\b",
    re.I | re.S)

# Promises the shop never made: only staff and the shop's own pages may offer these
PROMISE_RE = re.compile(
    r"\b(discounts?|coupons?|promo(?:tion(?:al)?)? codes?|vouchers?|\d+\s?% off|free (?:gift|shipping|samples?|delivery)"
    r"|price match|refunds?|money[- ]back|guaranteed? (?:results|to)|special (?:deal|offer|price)|on the house)\b",
    re.I)

# Words that only appear when the model talks about its instructions or the data behind them
INTERNAL_RE = re.compile(
    r"\b(can_recommend|in_stock|margins?|stock (?:counts?|levels?|quantit(?:y|ies))|units? (?:left|in stock|available)"
    r"|system prompt|my (?:instructions|rules|prompt)|these instructions|catalog facts)\b",
    re.I)

PRICE_RE = re.compile(r"[$€£₩¥]\s?(\d[\d,]*(?:\.\d{1,2})?)|(\d[\d,]*(?:\.\d{1,2})?)\s?(?:USD|CAD|AUD|EUR|GBP|KRW|dollars?)\b",
                      re.I)

# A sentence (with its trailing space) or a line, so dropping one keeps the rest of the formatting
_PIECES = re.compile(r".+?(?:[.!?](?=\s|$)|\n|$)\s*", re.S)


def looks_like_injection(message: str) -> bool:
    return bool(INJECTION_RE.search(message or ""))


def _amount(s: str) -> float | None:
    try:
        return round(float(s.replace(",", "")), 2)
    except ValueError:
        return None


def guard_answer(text: str, rows: list[dict], profile=None) -> str:
    """`text` without the sentences a customer must not see. Prices are allowed when they are the price of a
    product in `rows` or the customer's own budget."""
    allowed = {round(float(r["price"]), 2) for r in rows if r.get("price") is not None}
    for b in (getattr(profile, "budget_min", None), getattr(profile, "budget_max", None)):
        if b:
            allowed.add(round(float(b), 2))
    kept = []
    for piece in _PIECES.findall(text or ""):
        if PROMISE_RE.search(piece) or INTERNAL_RE.search(piece):
            continue
        amounts = [_amount(a or b) for a, b in PRICE_RE.findall(piece)]
        if any(a is not None and a not in allowed for a in amounts):
            continue
        kept.append(piece)
    return "".join(kept).strip()
