from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
INFO_PATH = ROOT / "submission" / "submission_info.json"
METRICS_PATH = ROOT / "results" / "metrics.json"
REPORT_PATH = ROOT / "results" / "classification_report.txt"
CONFUSION_MATRIX_PATH = ROOT / "results" / "confusion_matrix.png"
FEATURE_OVERVIEW_PATH = ROOT / "results" / "feature_overview.png"
PDF_OUTPUT_PATH = ROOT / "submission" / "FinalProject_GROUP18_Phase1_Report.pdf"
UNICODE_FONT_PATHS = [
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]


def register_fonts() -> tuple[str, str]:
    for path in UNICODE_FONT_PATHS:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ArialUnicode", str(path)))
            return "ArialUnicode", "ArialUnicode"
    return "Helvetica", "Helvetica-Bold"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    if not path.exists():
        return "Not generated yet."
    return path.read_text(encoding="utf-8").strip()


def fmt(value: Any, default: str = "TBD") -> str:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return str(value)


def build_styles():
    base_font, bold_font = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyTR",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10.5,
            leading=15,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleTR",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#17324d"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingTR",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
