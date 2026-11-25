"""Validation for tool names and permission rules handed to the CLI.

Claude Code splits every ``--tools``, ``--allowedTools``, and
``--disallowedTools`` value -- and each agent ``tools`` / ``disallowedTools``
string sent on the initialize request -- into rules on commas and on spaces
outside parentheses, then trims each rule. The SDK joins list entries with
commas, so one entry must tokenize to exactly one rule, itself; otherwise a
single caller-supplied string can add rules (a bare ``Bash`` allows the whole
tool) that the caller never passed as list elements.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Characters JavaScript's String.prototype.trim() removes, which is what the
# CLI applies to each token.
_JS_TRIM_CHARS = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _tokenize_tool_rules(value: str) -> tuple[list[str], bool]:
    """Split ``value`` the way the CLI does.

    Returns the tokens and whether the scan ended inside parentheses. The CLI
    tracks parentheses with a single flag, not a depth counter: any ``)``
    leaves the parenthesized state, so ``Bash(a (b) c)`` splits at the space
    after ``(b)``.
    """
    tokens: list[str] = []
    current: list[str] = []
    in_parens = False

    def flush() -> None:
        token = "".join(current).strip(_JS_TRIM_CHARS)
        if token:
            tokens.append(token)
        current.clear()

    for char in value:
        if char == "(":
            in_parens = True
            current.append(char)
        elif char == ")":
            in_parens = False
            current.append(char)
        elif char in ", " and not in_parens:
            flush()
        else:
            current.append(char)
    flush()
    return tokens, in_parens


def validate_tool_rule(field: str, entry: object) -> None:
    """Reject an entry the CLI would not read back as exactly one rule.

    An entry that ends inside parentheses is rejected too: the CLI would
    swallow the comma that joins it to the next entry and merge the two.
    """
    if not isinstance(entry, str):
        raise TypeError(
            f"{field} entries must be strings, got {type(entry).__name__}: {entry!r}"
        )
    tokens, ends_in_parens = _tokenize_tool_rules(entry)
    if tokens != [entry] or ends_in_parens:
        raise ValueError(
            f"Invalid {field} entry {entry!r}: each entry must be a single tool"
            " name or permission rule. Commas, spaces outside parentheses,"
            " leading or trailing whitespace, and an unclosed '(' are not"
            " allowed, because the CLI would split or merge the entry into"
            " different rules. Pass each rule as its own list element."
        )


def validate_tool_rules(field: str, entries: Iterable[object]) -> None:
    for entry in entries:
        validate_tool_rule(field, entry)


def validate_agent_tool_rules(agents: dict[str, dict[str, Any]]) -> None:
    """Validate the tool lists of agent definitions bound for initialize."""
    for name, agent in agents.items():
        for key in ("tools", "disallowedTools"):
            value = agent.get(key)
            if value is None:
                continue
            entries = [value] if isinstance(value, str) else value
            validate_tool_rules(f"agents[{name!r}].{key}", entries)
