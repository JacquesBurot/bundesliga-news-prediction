"""Build a versioned news-source policy from a reviewed XLSX workbook."""

from __future__ import annotations

import hashlib
import posixpath
import re
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

from buli_news.news_source_review import (
    SOURCE_REVIEW_COLUMNS,
    SOURCE_REVIEW_WORKSHEET,
)


MAIN_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
REVIEW_STATUS_DECISIONS = {
    "complete_no_ml_clause": "include",
    "complete_explicit_ml_clause": "exclude",
}
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
CELL_REFERENCE_PATTERN = re.compile(r"([A-Z]+)[1-9][0-9]*")
EXCEL_DATE_EPOCH = datetime(1899, 12, 30)


def build_news_source_policy(
    workbook_path: Path,
    season: int,
    policy_version: int = 1,
) -> dict[str, Any]:
    """Convert one reviewed source workbook into a deterministic policy."""
    if isinstance(season, bool) or not isinstance(season, int):
        msg = "Season must be an integer start year."
        raise ValueError(msg)
    if isinstance(policy_version, bool) or not isinstance(policy_version, int):
        msg = "Policy version must be an integer."
        raise ValueError(msg)
    if policy_version < 1:
        msg = "Policy version must be greater than zero."
        raise ValueError(msg)

    rows = read_source_review_workbook(workbook_path)
    rules = [build_policy_rule(row, row_number) for row_number, row in rows]
    rules.sort(key=lambda rule: rule["host"])
    validate_unique_hosts(rules)

    decision_counts = Counter(rule["decision"] for rule in rules)
    review_dates = sorted({rule["review_date"] for rule in rules})
    policy = {
        "schema_version": 1,
        "policy_id": f"news_source_policy_{season}_v{policy_version}",
        "purpose": (
            "Control which collected news sources may enter local-LLM feature "
            "extraction and downstream match-outcome modeling."
        ),
        "source_review": {
            "workbook_path": workbook_path.as_posix(),
            "workbook_sha256": hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
            "worksheet": SOURCE_REVIEW_WORKSHEET,
            "reviewed_source_count": len(rules),
            "review_date_min": min(review_dates),
            "review_date_max": max(review_dates),
        },
        "decision_policy": {
            "unknown_host_decision": "exclude",
            "host_matching": (
                "Normalize the candidate hostname to lowercase, remove one "
                "leading 'www.' label, and require an exact match with rule.host."
            ),
            "include_status": "complete_no_ml_clause",
            "exclude_status": "complete_explicit_ml_clause",
            "interpretation": (
                "The policy reproduces the review decisions recorded in the "
                "source workbook. It does not independently determine legal "
                "permission."
            ),
        },
        "summary": {
            "include_source_count": decision_counts["include"],
            "exclude_source_count": decision_counts["exclude"],
            "unreviewed_source_count": 0,
            "reviewed_article_occurrence_count": sum(
                rule["article_count"] for rule in rules
            ),
            "included_article_occurrence_count_before_deduplication": sum(
                rule["article_count"]
                for rule in rules
                if rule["decision"] == "include"
            ),
            "excluded_article_occurrence_count_before_deduplication": sum(
                rule["article_count"]
                for rule in rules
                if rule["decision"] == "exclude"
            ),
        },
        "rules": rules,
    }
    validate_news_source_policy(policy)
    return policy


def read_source_review_workbook(
    workbook_path: Path,
) -> list[tuple[int, dict[str, str]]]:
    """Read and validate the fixed source-review worksheet from an XLSX file."""
    try:
        with zipfile.ZipFile(workbook_path) as archive:
            worksheet_path = find_worksheet_path(
                archive,
                worksheet_name=SOURCE_REVIEW_WORKSHEET,
            )
            shared_strings = read_shared_strings(archive)
            raw_rows = read_worksheet_rows(
                archive,
                worksheet_path=worksheet_path,
                shared_strings=shared_strings,
            )
    except zipfile.BadZipFile as exc:
        msg = f"Source review workbook is not a valid XLSX file: {workbook_path}."
        raise ValueError(msg) from exc
    except KeyError as exc:
        msg = f"Source review workbook is missing required XLSX content: {exc}."
        raise ValueError(msg) from exc

    if not raw_rows:
        msg = "Source review worksheet is empty."
        raise ValueError(msg)

    header_row_number, header_values = raw_rows[0]
    headers = tuple(header_values.get(index, "").strip() for index in range(1, 6))
    if header_row_number != 1 or headers != SOURCE_REVIEW_COLUMNS:
        msg = (
            "Source review worksheet must have the exact columns "
            f"{list(SOURCE_REVIEW_COLUMNS)} in row 1, got {list(headers)}."
        )
        raise ValueError(msg)
    extra_header_columns = [
        value
        for index, value in sorted(header_values.items())
        if index > len(SOURCE_REVIEW_COLUMNS) and value.strip()
    ]
    if extra_header_columns:
        msg = f"Source review worksheet has unexpected columns: {extra_header_columns}."
        raise ValueError(msg)

    records = []
    for row_number, values in raw_rows[1:]:
        if not any(value.strip() for value in values.values()):
            continue
        extra_values = [
            value
            for index, value in sorted(values.items())
            if index > len(SOURCE_REVIEW_COLUMNS) and value.strip()
        ]
        if extra_values:
            msg = f"Source review row {row_number} has values outside columns A-E."
            raise ValueError(msg)
        record = {
            column: values.get(index, "").strip()
            for index, column in enumerate(SOURCE_REVIEW_COLUMNS, start=1)
        }
        records.append((row_number, record))

    if not records:
        msg = "Source review worksheet contains no source rows."
        raise ValueError(msg)
    return records


def find_worksheet_path(
    archive: zipfile.ZipFile,
    worksheet_name: str,
) -> str:
    """Resolve a worksheet name through the XLSX relationship files."""
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook.findall(f".//{{{MAIN_NAMESPACE}}}sheet"):
        if sheet.attrib.get("name") == worksheet_name:
            relationship_id = sheet.attrib.get(
                f"{{{OFFICE_RELATIONSHIP_NAMESPACE}}}id"
            )
            break
    if relationship_id is None:
        msg = f"Source review workbook has no worksheet {worksheet_name!r}."
        raise ValueError(msg)

    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships.findall(
        f"{{{PACKAGE_RELATIONSHIP_NAMESPACE}}}Relationship"
    ):
        if relationship.attrib.get("Id") != relationship_id:
            continue
        target = relationship.attrib.get("Target")
        if not target:
            break
        worksheet_path = posixpath.normpath(posixpath.join("xl", target))
        if not worksheet_path.startswith("xl/"):
            msg = f"Worksheet path escapes the XLSX archive: {target!r}."
            raise ValueError(msg)
        return worksheet_path

    msg = f"Worksheet relationship {relationship_id!r} cannot be resolved."
    raise ValueError(msg)


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Read the optional XLSX shared-string table."""
    try:
        content = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ElementTree.fromstring(content)
    return [
        "".join(text.text or "" for text in item.iter(f"{{{MAIN_NAMESPACE}}}t"))
        for item in root.findall(f"{{{MAIN_NAMESPACE}}}si")
    ]


def read_worksheet_rows(
    archive: zipfile.ZipFile,
    worksheet_path: str,
    shared_strings: list[str],
) -> list[tuple[int, dict[int, str]]]:
    """Read sparse XLSX worksheet cells into row and column indices."""
    root = ElementTree.fromstring(archive.read(worksheet_path))
    rows = []
    for row in root.findall(f".//{{{MAIN_NAMESPACE}}}row"):
        row_number = int(row.attrib["r"])
        values = {}
        for cell in row.findall(f"{{{MAIN_NAMESPACE}}}c"):
            reference = cell.attrib.get("r", "")
            match = CELL_REFERENCE_PATTERN.fullmatch(reference)
            if match is None:
                msg = f"Invalid XLSX cell reference {reference!r}."
                raise ValueError(msg)
            column_index = excel_column_index(match.group(1))
            values[column_index] = read_cell_value(cell, shared_strings)
        rows.append((row_number, values))
    return rows


def read_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    """Return one XLSX cell's cached scalar value as text."""
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            text.text or ""
            for text in cell.iter(f"{{{MAIN_NAMESPACE}}}t")
        )

    value_node = cell.find(f"{{{MAIN_NAMESPACE}}}v")
    if value_node is None or value_node.text is None:
        return ""
    raw_value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (IndexError, ValueError) as exc:
            msg = f"Invalid XLSX shared-string index {raw_value!r}."
            raise ValueError(msg) from exc
    return raw_value


def excel_column_index(column_name: str) -> int:
    """Convert an Excel column name such as A or AA to a one-based index."""
    index = 0
    for character in column_name:
        index = index * 26 + ord(character) - ord("A") + 1
    return index


def build_policy_rule(row: dict[str, str], row_number: int) -> dict[str, Any]:
    """Validate one reviewed source row and convert it to one policy rule."""
    website_url = require_text(row, "website_url", row_number)
    host = normalize_reviewed_website_url(website_url, row_number)
    article_count = parse_positive_integer(row, "article_count", row_number)
    review_status = require_text(row, "review_status", row_number)
    try:
        decision = REVIEW_STATUS_DECISIONS[review_status]
    except KeyError as exc:
        msg = (
            f"Source review row {row_number} has unsupported review_status "
            f"{review_status!r}."
        )
        raise ValueError(msg) from exc

    legal_text = row["legal_text"].strip()
    evidence_urls = extract_evidence_urls(legal_text)
    if decision == "exclude" and not legal_text:
        msg = f"Excluded source review row {row_number} needs legal_text."
        raise ValueError(msg)
    if decision == "exclude" and not evidence_urls:
        msg = f"Excluded source review row {row_number} needs an evidence URL."
        raise ValueError(msg)
    if decision == "include" and legal_text:
        msg = (
            f"Included source review row {row_number} must leave legal_text empty."
        )
        raise ValueError(msg)

    return {
        "host": host,
        "website_url": website_url,
        "match_type": "exact_host",
        "decision": decision,
        "review_status": review_status,
        "review_date": parse_review_date(row["review_date"], row_number),
        "article_count": article_count,
        "evidence_urls": evidence_urls,
    }


def normalize_reviewed_website_url(value: str, row_number: int) -> str:
    """Validate one reviewed homepage URL and return its normalized host."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        msg = f"Source review row {row_number} has invalid website_url {value!r}."
        raise ValueError(msg)
    if parsed.username or parsed.password or parsed.port:
        msg = (
            f"Source review row {row_number} website_url must not contain "
            "user info or a port."
        )
        raise ValueError(msg)
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        msg = f"Source review row {row_number} website_url must be a homepage URL."
        raise ValueError(msg)

    host = parsed.hostname.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        msg = f"Source review row {row_number} has an empty website host."
        raise ValueError(msg)
    return host


def parse_positive_integer(row: dict[str, str], key: str, row_number: int) -> int:
    """Parse one required positive whole-number cell."""
    value = require_text(row, key, row_number)
    try:
        numeric_value = float(value)
    except ValueError as exc:
        msg = f"Source review row {row_number} field {key!r} must be numeric."
        raise ValueError(msg) from exc
    if not numeric_value.is_integer() or numeric_value <= 0:
        msg = f"Source review row {row_number} field {key!r} must be positive integer."
        raise ValueError(msg)
    return int(numeric_value)


def parse_review_date(value: str, row_number: int) -> str:
    """Parse an ISO date or an Excel serial date and return YYYY-MM-DD."""
    stripped = value.strip()
    try:
        return date.fromisoformat(stripped).isoformat()
    except ValueError:
        pass

    try:
        numeric_value = float(stripped)
    except ValueError as exc:
        msg = f"Source review row {row_number} has invalid review_date {value!r}."
        raise ValueError(msg) from exc
    if not numeric_value.is_integer() or numeric_value <= 0:
        msg = f"Source review row {row_number} has invalid review_date {value!r}."
        raise ValueError(msg)
    return (EXCEL_DATE_EPOCH + timedelta(days=int(numeric_value))).date().isoformat()


def extract_evidence_urls(legal_text: str) -> list[str]:
    """Extract stable, de-duplicated evidence URLs from reviewed legal text."""
    urls = [url.rstrip(".,);\"'") for url in URL_PATTERN.findall(legal_text)]
    return list(dict.fromkeys(urls))


def validate_unique_hosts(rules: list[dict[str, Any]]) -> None:
    """Reject duplicate normalized hosts in the reviewed workbook."""
    counts = Counter(rule["host"] for rule in rules)
    duplicate_hosts = sorted(host for host, count in counts.items() if count > 1)
    if duplicate_hosts:
        msg = f"Source review workbook contains duplicate hosts: {duplicate_hosts}."
        raise ValueError(msg)


def validate_news_source_policy(policy: dict[str, Any]) -> None:
    """Validate internal counts and ordering in one generated policy."""
    rules = policy["rules"]
    hosts = [rule["host"] for rule in rules]
    if hosts != sorted(hosts):
        msg = "News source policy rules must be sorted by host."
        raise ValueError(msg)
    validate_unique_hosts(rules)

    summary = policy["summary"]
    decision_counts = Counter(rule["decision"] for rule in rules)
    expected_summary = {
        "include_source_count": decision_counts["include"],
        "exclude_source_count": decision_counts["exclude"],
        "unreviewed_source_count": 0,
        "reviewed_article_occurrence_count": sum(
            rule["article_count"] for rule in rules
        ),
        "included_article_occurrence_count_before_deduplication": sum(
            rule["article_count"]
            for rule in rules
            if rule["decision"] == "include"
        ),
        "excluded_article_occurrence_count_before_deduplication": sum(
            rule["article_count"]
            for rule in rules
            if rule["decision"] == "exclude"
        ),
    }
    if summary != expected_summary:
        msg = "News source policy summary does not match its rules."
        raise ValueError(msg)


def require_text(row: dict[str, str], key: str, row_number: int) -> str:
    """Return one required, non-empty reviewed source value."""
    value = row.get(key, "").strip()
    if not value:
        msg = f"Source review row {row_number} needs non-empty field {key!r}."
        raise ValueError(msg)
    return value
