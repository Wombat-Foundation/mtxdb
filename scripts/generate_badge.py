"""Render a small "Tests: P / F / I" shield-style badge to tests.svg."""

import sys


def main():
    """Read P/F/I counts from argv and write tests.svg."""
    if len(sys.argv) < 4:
        print("Usage: generate_badge.py <passed> <failed> <ignored>")
        sys.exit(1)

    passed = sys.argv[1]
    failed = sys.argv[2]
    ignored = sys.argv[3]

    text = (
        '<text x="80" y="14" fill="#fff" text-anchor="middle">'
        f"Tests: {passed} P / {failed} F / {ignored} I</text>"
    )
    badge_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="160" height="20">
    <rect width="160" height="20" fill="#555"/>
    {text}
</svg>"""

    with open("tests.svg", "w", encoding="utf-8") as f:
        f.write(badge_svg)


if __name__ == "__main__":
    main()
