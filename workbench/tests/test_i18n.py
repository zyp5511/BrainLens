"""Check the offline UI message catalog without requiring a browser or JS runtime."""

import ast
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "src" / "brainlens" / "assets"
# Match a quoted first argument, including escaped quotes and multiline calls.
LITERAL_KEY = re.compile(
    r'''\b(?:t|setStatusKey|reviewSetPlaceholder|resetSetTarget)\(\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')'''
)
PLACEHOLDER = re.compile(r"\{(\w+)\}")


class MarkupKeys(HTMLParser):
    def __init__(self):
        super().__init__()
        self.keys = set()

    def handle_starttag(self, tag, attrs):
        if tag == "option" and "data-i18n" in dict(attrs):
            assert "value" in dict(attrs), "A translated label cannot double as its option value"
        for name, value in attrs:
            if name == "data-i18n" or name.startswith("data-i18n-"):
                assert name in {"data-i18n", "data-i18n-title", "data-i18n-placeholder", "data-i18n-aria-label"}
                self.keys.add(value)


def test_every_ui_message_has_both_languages_and_matching_parameters():
    # Each property in the bilingual catalog is an English message and its zh-CN translation.
    # The runtime builds its English dictionary from these exact keys.
    source = (ASSETS / "i18n.js").read_text(encoding="utf-8")
    encoded = source.split("const zhCN = ", 1)[1]
    catalog, _ = json.JSONDecoder().raw_decode(encoded)
    parser = MarkupKeys()
    parser.feed((ASSETS / "index.html").read_text(encoding="utf-8"))
    used = set(parser.keys)
    for path in ASSETS.glob("*.js"):
        used.update(ast.literal_eval(match) for match in LITERAL_KEY.findall(path.read_text(encoding="utf-8")))

    assert used - catalog.keys() == set()
    for english, chinese in catalog.items():
        assert english.strip() and chinese.strip(), english
        assert Counter(PLACEHOLDER.findall(english)) == Counter(PLACEHOLDER.findall(chinese)), english
