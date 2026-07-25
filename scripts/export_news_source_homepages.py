"""Export unique source homepage URLs from raw Event Registry responses.

This is a small temporary utility for inspecting the collected raw news data.
It intentionally uses only the Python standard library and writes a minimal
.xlsx file without requiring openpyxl or xlsxwriter.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import urlparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export unique website homepage URLs from raw news responses."
    )
    parser.add_argument(
        "--season",
        required=True,
        type=int,
        help="Season start year, e.g. 2025 for 2025/26.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory with raw NewsAPI/Event Registry JSON files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .xlsx path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir is not None
        else Path("data") / "raw" / "newsapi" / str(args.season)
    )
    output_path = (
        Path(args.output)
        if args.output is not None
        else Path("data") / "interim" / f"news_source_homepages_{args.season}.xlsx"
    )

    homepage_counts = extract_homepage_counts(raw_dir)
    rows = [
        [homepage_url, str(article_count)]
        for homepage_url, article_count in homepage_counts
    ]
    write_xlsx(
        output_path,
        headers=["website_url", "article_count"],
        rows=rows,
    )

    print(f"Read raw responses from {raw_dir}")
    print(f"Exported {len(homepage_counts)} unique website URLs to {output_path}")


def extract_homepage_counts(raw_dir: Path) -> list[tuple[str, int]]:
    """Extract source homepage URLs with article counts from raw response files."""
    counts: Counter[str] = Counter()
    for path in sorted(raw_dir.glob("*.json")):
        response = read_json(path)
        results = response.get("articles", {}).get("results")
        if not isinstance(results, list):
            continue

        for article in results:
            if not isinstance(article, dict):
                continue

            homepage_url = get_homepage_url(article)
            if homepage_url is not None:
                counts[homepage_url] += 1

    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def get_homepage_url(article: dict) -> str | None:
    """Return the normalized homepage URL for an article source."""
    article_url = article.get("url")
    homepage_url = normalize_homepage_url(article_url)
    if homepage_url is not None:
        return homepage_url

    source = article.get("source")
    if isinstance(source, dict):
        return normalize_homepage_url(source.get("uri"))

    return None


def normalize_homepage_url(value: object) -> str | None:
    """Normalize an article URL or source URI to a site homepage URL."""
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None

    parsed = urlparse(stripped if "://" in stripped else f"https://{stripped}")
    host = parsed.netloc.lower()
    if not host:
        return None

    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]

    return f"https://{host}"


def write_xlsx(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Write XLSX with inline strings using only stdlib zipfile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = [
        build_row(row_number=1, values=headers),
        *[
            build_row(row_number=index + 2, values=row)
            for index, row in enumerate(rows)
        ],
    ]
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        f"{''.join(sheet_rows)}"
        "</sheetData>"
        "</worksheet>"
    )

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types_xml())
        xlsx.writestr("_rels/.rels", root_relationships_xml())
        xlsx.writestr("xl/workbook.xml", workbook_xml())
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_relationships_xml())
        xlsx.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        xlsx.writestr("xl/styles.xml", styles_xml())


def build_row(row_number: int, values: list[str]) -> str:
    cells = []
    for index, value in enumerate(values, start=1):
        cell_ref = f"{column_name(index)}{row_number}"
        cells.append(
            f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
        )
    return f'<row r="{row_number}">{"".join(cells)}</row>'


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )


def root_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def workbook_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        '<sheet name="sources" sheetId="1" r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )


def workbook_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
        "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
        "<borders count=\"1\"><border/></borders>"
        "<cellStyleXfs count=\"1\"><xf/></cellStyleXfs>"
        "<cellXfs count=\"1\"><xf xfId=\"0\"/></cellXfs>"
        "</styleSheet>"
    )


if __name__ == "__main__":
    main()
