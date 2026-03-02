#!/usr/bin/env python3
"""Fail if hardcoded non-ACS seeds exist in deep_learning/*.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXPECTED_SEED = 2024
ROOT = Path(__file__).resolve().parents[1]

PATTERNS = [
    ("random_state", re.compile(r"\brandom_state\s*[:=][^#\n]*?(\d+)")),
    ("seed assignment", re.compile(r"\bseed\s*=\s*(\d+)")),
    ("tf.random.set_seed", re.compile(r"tf\.random\.set_seed\((\d+)\)")),
    ("np.random.seed", re.compile(r"np\.random\.seed\((\d+)\)")),
    ("random.seed", re.compile(r"\brandom\.seed\((\d+)\)")),
]


def main() -> int:
    violations = []

    for path in sorted(ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                for match in pattern.finditer(line):
                    value = int(match.group(1))
                    if value != EXPECTED_SEED:
                        violations.append((path, lineno, label, value, line.strip()))

    if violations:
        print("Found hardcoded non-ACS seeds/random_states:")
        for path, lineno, label, value, line in violations:
            print(f"- {path}:{lineno}: {label}={value} -> {line}")
        return 1

    print(f"Seed audit passed. No hardcoded seed/random_state values other than {EXPECTED_SEED}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
