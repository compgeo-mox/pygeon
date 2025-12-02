import sys
from collections import defaultdict
from pathlib import Path

import bibtexparser

# Ensure script directory is on the path
sys.path.append(str(Path(__file__).parent.resolve()))


def apa_format_rst(entry):
    authors = entry.get("author", "").replace("\n", " ")
    authors = "; ".join(a.strip() for a in authors.split(" and "))

    year = entry.get("year", "n.d.")
    title = entry.get("title", "").rstrip(".")
    journal = entry.get("journal", entry.get("booktitle", ""))

    doi = entry.get("doi")
    url = entry.get("url")

    lines = [f"{authors} ({year}). *{title}*. {journal}."]
    if doi:
        lines.append(f"  DOI: `https://doi.org/{doi} <https://doi.org/{doi}>`_")
    if url:
        lines.append(f"  Repo: `Link <{url}>`_")

    return "\n".join(lines)


def build_publication_section(entries):
    papers_by_year = defaultdict(list)
    for entry in entries:
        year = entry.get("year", "Unknown")
        papers_by_year[year].append(entry)

    sorted_years = sorted(papers_by_year.keys(), reverse=True)

    lines = []
    for year in sorted_years:
        lines.append(year)
        lines.append("-" * len(year))  # RST underline for section
        for e in papers_by_year[year]:
            lines.append(f"- {apa_format_rst(e)}")
        lines.append("")  # blank line
    return "\n".join(lines)


def main():
    root = Path(__file__).parent
    bib_path = root / "papers.bib"
    readme_path = root / "papers.rst"

    try:
        with open(bib_path, encoding="utf-8") as f:
            try:
                db = bibtexparser.load(f)
            except Exception as e:
                print(f"Error: Failed to parse '{bib_path}'. Is the BibTeX syntax valid?")
                print(f"Details: {e}")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: '{bib_path}' not found. Please ensure the file exists.")
        sys.exit(1)
    except OSError as e:
        print(f"Error: Could not read '{bib_path}': {e}")
        sys.exit(1)

    papers_section = build_publication_section(db.entries)

    # Static content before/after can be hard-coded or read from template
    static_header = """Papers Using PyGeoN
===================

This page lists academic papers and research that use PyGeoN.

"""
    static_footer = """
Citing PyGeoN
-------------

If you use PyGeoN in your research, please cite it using the DOI:

.. image:: https://zenodo.org/badge/455087135.svg
   :target: https://zenodo.org/badge/latestdoi/455087135
   :alt: DOI

Contributing Your Paper
-----------------------

If you have published work using PyGeoN, we'd love to add it to our list! 
Please submit a pull request to add your paper to the papers list.
"""

    new_content = f"{static_header}{papers_section}\n{static_footer}"
    readme_path.write_text(new_content, encoding="utf-8")
    print("papers.rst updated.")


if __name__ == "__main__":
    main()
