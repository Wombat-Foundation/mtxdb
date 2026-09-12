import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: generate_badge.py <passed> <failed> <ignored>")
        sys.exit(1)

    passed = sys.argv[1]
    failed = sys.argv[2]
    ignored = sys.argv[3]

    badge_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="160" height="20">
    <rect width="160" height="20" fill="#555"/>
    <text x="80" y="14" fill="#fff" text-anchor="middle">Tests: {passed} P / {failed} F / {ignored} I</text>
</svg>"""

    with open("tests.svg", "w") as f:
        f.write(badge_svg)

if __name__ == "__main__":
    main()
