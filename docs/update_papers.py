import sys
from collections import defaultdict
from pathlib import Path

import bibtexparser


def apa_format_rst(entry):
    """Format a single BibTeX entry in APA style for RST with DOI and Repo links."""
    authors = (
        entry.get("author", "").replace("\n", " ").replace("{", "").replace("}", "")
    )
    authors = "; ".join(a.strip() for a in authors.split(" and "))

    year = entry.get("year", "n.d.")
    title = entry.get("title", "").replace("{", "").replace("}", "").rstrip(".")
    journal = entry.get("journal", entry.get("booktitle", ""))

    doi = entry.get("doi")
    url = entry.get("url")

    lines = [f"{authors} ({year}). *{title}*. {journal}."]
    if doi:
        lines.append(f"DOI: `https://doi.org/{doi} <https://doi.org/{doi}>`_")
    if url:
        lines.append(f"Repo: `Link <{url}>`_")

    return "\n".join(lines)


def build_publication_section(entries):
    """Group entries by year and format for RST."""
    papers_by_year = defaultdict(list)
    for entry in entries:
        year = entry.get("year", "Unknown")
        papers_by_year[year].append(entry)

    def year_sort_key(year):
        try:
            # Sort numeric years descending, unknown/non-numeric last
            return (0, -int(year))
        except ValueError:
            return (1, year)

    sorted_years = sorted(papers_by_year.keys(), key=year_sort_key)

    lines = []
    for year in sorted_years:
        lines.append(year)
        lines.append("-" * len(year))  # RST underline
        for e in papers_by_year[year]:
            entry_text = apa_format_rst(e)
            # Indent continuation lines for proper RST bullet rendering
            indented_entry = entry_text.replace("\n", "\n  ")
            lines.append(f"- {indented_entry}")
        lines.append("")  # blank line after each year
    return "\n".join(lines)


def main():
    # Paths
    docs_dir = Path(__file__).parent
    bib_path = docs_dir / "papers.bib"
    readme_path = docs_dir / "papers.rst"  # write to repo root

    # Load BibTeX
    try:
        with open(bib_path, encoding="utf-8") as f:
            try:
                db = bibtexparser.load(f)
            except Exception as e:
                print(f"Error: Failed to parse '{bib_path}'. Check BibTeX syntax.")
                print(f"Details: {e}")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: '{bib_path}' not found.")
        sys.exit(1)
    except OSError as e:
        print(f"Error: Could not read '{bib_path}': {e}")
        sys.exit(1)

    # Build publications section
    papers_section = build_publication_section(db.entries)

    # Static header/footer
    static_header = """Papers Using PyGeoN
===================

This page lists academic papers and research that use PyGeoN.

Publications
------------
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

    # Write papers.rst
    new_content = f"{static_header}{papers_section}{static_footer}"
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"papers.rst updated at {readme_path}")


if __name__ == "__main__":
    main()
