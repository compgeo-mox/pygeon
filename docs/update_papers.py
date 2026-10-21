"""Generate papers.rst from papers.bib, with the papers grouped by year."""

from collections import defaultdict
from pathlib import Path

import bibtexparser

HEADER = """Papers using PyGeoN
===================

If you use PyGeoN in your research, please cite it through its
`DOI <https://zenodo.org/badge/latestdoi/455087135>`_. To add your paper to this list,
open a pull request with its entry in ``docs/papers.bib``.

"""


def format_entry(entry: dict) -> str:
    """Format a BibTeX entry as an rst bullet, the title linked to the DOI."""
    clean = lambda s: " ".join(s.replace("{", "").replace("}", "").split())
    authors = "; ".join(a.strip() for a in clean(entry["author"]).split(" and "))
    title = clean(entry["title"]).rstrip(".")
    if "doi" in entry:
        title = f"`{title} <https://doi.org/{entry['doi']}>`__"
    journal = entry.get("journal", entry.get("booktitle", ""))

    line = f"- {authors} ({entry['year']}). {title}. *{journal}*."
    if "url" in entry:
        line += f" `Code <{entry['url']}>`__."
    return line


def main() -> None:
    docs_dir = Path(__file__).parent
    with open(docs_dir / "papers.bib", encoding="utf-8") as f:
        entries = bibtexparser.load(f).entries

    by_year = defaultdict(list)
    for entry in entries:
        by_year[entry["year"]].append(entry)

    sections = [
        f"{year}\n{'-' * len(year)}\n\n" + "\n".join(map(format_entry, by_year[year]))
        for year in sorted(by_year, reverse=True)
    ]
    out = docs_dir / "papers.rst"
    out.write_text(HEADER + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Written {len(entries)} papers in {len(by_year)} years to {out}")


if __name__ == "__main__":
    main()
