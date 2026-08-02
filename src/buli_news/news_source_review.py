"""Export collected news-source homepages into a review-ready XLSX workbook."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import urlsplit


SOURCE_REVIEW_WORKSHEET = "sources"
SOURCE_REVIEW_COLUMNS = (
    "website_url",
    "article_count",
    "legal_text",
    "review_status",
    "review_date",
)
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def export_news_source_review(
    raw_dir: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> list[tuple[str, int]]:
    """Extract source counts and write the workbook used for legal review."""
    homepage_counts = extract_homepage_counts(raw_dir)
    write_source_review_workbook(
        output_path=output_path,
        homepage_counts=homepage_counts,
        overwrite=overwrite,
    )
    return homepage_counts


def extract_homepage_counts(raw_dir: Path) -> list[tuple[str, int]]:
    """Extract normalized source homepage URLs from raw news responses."""
    response_paths = sorted(raw_dir.glob("*.json"))
    if not response_paths:
        msg = f"No raw news response JSON files found in {raw_dir}."
        raise ValueError(msg)

    counts: Counter[str] = Counter()
    for path in response_paths:
        response = read_raw_response(path)
        articles = response.get("articles")
        if not isinstance(articles, dict):
            msg = f"Raw news response {path} has no articles object."
            raise ValueError(msg)
        results = articles.get("results")
        if not isinstance(results, list):
            msg = f"Raw news response {path} has no articles.results list."
            raise ValueError(msg)

        for article_index, article in enumerate(results, start=1):
            if not isinstance(article, dict):
                msg = (
                    f"Raw news response {path} article {article_index} "
                    "must be an object."
                )
                raise ValueError(msg)
            homepage_url = get_homepage_url(article)
            if homepage_url is None:
                msg = (
                    f"Raw news response {path} article {article_index} "
                    "has no valid source URL."
                )
                raise ValueError(msg)
            counts[homepage_url] += 1

    if not counts:
        msg = f"Raw news responses in {raw_dir} contain no articles."
        raise ValueError(msg)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def read_raw_response(path: Path) -> dict:
    """Read one raw response and require its top-level JSON object."""
    with path.open("r", encoding="utf-8") as file:
        response = json.load(file)
    if not isinstance(response, dict):
        msg = f"Raw news response {path} must contain a JSON object."
        raise ValueError(msg)
    return response


def get_homepage_url(article: dict) -> str | None:
    """Return the normalized homepage URL for one article source."""
    homepage_url = normalize_homepage_url(article.get("url"))
    if homepage_url is not None:
        return homepage_url

    source = article.get("source")
    if isinstance(source, dict):
        return normalize_homepage_url(source.get("uri"))
    return None


def normalize_homepage_url(value: object) -> str | None:
    """Normalize an article URL or source URI to an HTTPS homepage URL."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None

    candidate = stripped if "://" in stripped else f"https://{stripped}"
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return None

    host = parsed.hostname.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host or any(character.isspace() for character in host):
        return None
    return f"https://{host}"


def write_source_review_workbook(
    output_path: Path,
    homepage_counts: list[tuple[str, int]],
    *,
    overwrite: bool = False,
) -> None:
    """Write the fixed source-review schema as a styled XLSX workbook."""
    if output_path.exists() and not overwrite:
        msg = (
            f"Output workbook already exists: {output_path}. "
            "Use --overwrite only if its manual review data may be replaced."
        )
        raise ValueError(msg)
    if not homepage_counts:
        msg = "Cannot write a source-review workbook without source rows."
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(homepage_counts) + 1
    parts = {
        "[Content_Types].xml": content_types_xml(),
        "_rels/.rels": root_relationships_xml(),
        "xl/workbook.xml": workbook_xml(),
        "xl/_rels/workbook.xml.rels": workbook_relationships_xml(),
        "xl/worksheets/sheet1.xml": worksheet_xml(homepage_counts),
        "xl/worksheets/_rels/sheet1.xml.rels": worksheet_relationships_xml(),
        "xl/tables/table1.xml": table_xml(row_count),
        "xl/styles.xml": styles_xml(),
    }
    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as workbook:
        for part_name, content in parts.items():
            part = zipfile.ZipInfo(part_name, date_time=FIXED_ZIP_TIMESTAMP)
            part.compress_type = zipfile.ZIP_DEFLATED
            workbook.writestr(part, content)


def worksheet_xml(homepage_counts: list[tuple[str, int]]) -> str:
    """Build the single review worksheet with typed article counts."""
    rows = [build_header_row()]
    for row_number, (homepage_url, article_count) in enumerate(
        homepage_counts,
        start=2,
    ):
        rows.append(
            f'<row r="{row_number}">'
            f'{inline_string_cell(f"A{row_number}", homepage_url, style=1)}'
            f'<c r="B{row_number}" s="2"><v>{article_count}</v></c>'
            "</row>"
        )
    final_row = len(homepage_counts) + 1
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:E{final_row}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols>'
        '<col min="1" max="1" width="34" customWidth="1"/>'
        '<col min="2" max="2" width="14" customWidth="1"/>'
        '<col min="3" max="3" width="70" customWidth="1"/>'
        '<col min="4" max="4" width="32" customWidth="1"/>'
        '<col min="5" max="5" width="14" customWidth="1"/>'
        '</cols>'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" '
        'header="0.3" footer="0.3"/>'
        '<tableParts count="1"><tablePart r:id="rId1"/></tableParts>'
        '</worksheet>'
    )


def build_header_row() -> str:
    cells = "".join(
        inline_string_cell(f"{column_name(index)}1", value, style=3)
        for index, value in enumerate(SOURCE_REVIEW_COLUMNS, start=1)
    )
    return f'<row r="1" ht="22" customHeight="1">{cells}</row>'


def inline_string_cell(reference: str, value: str, *, style: int) -> str:
    return (
        f'<c r="{reference}" s="{style}" t="inlineStr">'
        f'<is><t>{escape(value)}</t></is></c>'
    )


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
        '<Override PartName="/xl/tables/table1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )


def root_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def workbook_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="sources" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )


def workbook_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )


def worksheet_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table" Target="../tables/table1.xml"/>'
        '</Relationships>'
    )


def table_xml(final_row: int) -> str:
    columns = "".join(
        f'<tableColumn id="{index}" name="{escape(name)}"/>'
        for index, name in enumerate(SOURCE_REVIEW_COLUMNS, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'id="1" name="SourceReview" displayName="SourceReview" ref="A1:E{final_row}" '
        'headerRowCount="1">'
        f'<autoFilter ref="A1:E{final_row}"/>'
        f'<tableColumns count="{len(SOURCE_REVIEW_COLUMNS)}">{columns}</tableColumns>'
        '<tableStyleInfo name="TableStyleMedium2" showFirstColumn="0" '
        'showLastColumn="0" showRowStripes="1" showColumnStripes="0"/>'
        '</table>'
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="4">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" '
        'applyAlignment="1"><alignment vertical="top"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" '
        'applyAlignment="1"><alignment horizontal="right"/></xf>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" '
        'applyAlignment="1"><alignment vertical="center"/></xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
