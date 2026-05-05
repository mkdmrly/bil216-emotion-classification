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


def fmt(value: Any, default: str = "Belirlenecek") -> str:
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
    styles.add(
        ParagraphStyle(
            name="SmallTR",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#555555"),
            spaceAfter=6,
        )
    )
    return styles


def metadata_table(info: dict[str, Any], metrics: dict[str, Any], styles):
    student_numbers = info.get("student_numbers", {})
    numbered_members = []
    for member in info.get("team_members", []):
        number = student_numbers.get(member)
        numbered_members.append(f"{member} ({number})" if number else member)
    rows = [
        ["Course", "BIL216 Signals and Systems"],
        ["Project", "Sound Signal Analysis and Emotion Classification"],
        ["Phase", "Phase 1 - Beginning Model"],
        ["Group No", fmt(info.get("group_number"), "GROUP")],
        ["Team Members", fmt(", ".join(numbered_members), "To be added")],
        ["GitHub Link", fmt(info.get("github_url"))],
        ["Total Recordings", fmt(metrics.get("num_samples"))],
        ["Initial Accuracy", fmt(metrics.get("test_accuracy"))],
    ]
    table = Table(rows, colWidths=[4.2 * cm, 11.7 * cm])
    base_font = styles["BodyTR"].fontName
    bold_font = styles["HeadingTR"].fontName
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf1f7")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTNAME", (0, 0), (0, -1), bold_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c7d4")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def class_distribution_paragraph(metrics: dict[str, Any]) -> str:
    distribution = metrics.get("class_distribution")
    if not distribution:
        return (
            "Since the dataset has not yet been added to the project workspace, the class distribution "
            "could not be inserted automatically into this report. This section will be updated once "
            "the recordings are placed in the dataset folder."
        )
    items = ", ".join(f"{label}: {count}" for label, count in distribution.items())
    return f"The sample distribution in the dataset is as follows: {items}."


def contributions_paragraph(info: dict[str, Any]) -> str:
    contributions = info.get("member_contributions", [])
    if not contributions:
        return (
            "The team member contribution details have not been added yet. This section should briefly "
            "describe each member's work in coding, data collection, analysis, and reporting."
        )
    joined = "<br/>".join(f"- {entry}" for entry in contributions)
    return joined


def team_members_paragraph(info: dict[str, Any]) -> str:
    student_numbers = info.get("student_numbers", {})
    members = info.get("team_members", [])
    if not members:
        return "Team member details will be added."
    lines = []
    for member in members:
        number = student_numbers.get(member)
        if number:
            lines.append(f"- {member} - Student No: {number}")
        else:
            lines.append(f"- {member}")
    return "<br/>".join(lines)


def resources_paragraph(info: dict[str, Any]) -> str:
    resources = info.get("resources", [])
    if not resources:
        resources = [
            "Python",
            "librosa",
            "scikit-learn",
            "matplotlib",
            "GitHub",
            "ders notları ve proje yönergesi",
        ]
    return "<br/>".join(f"- {item}" for item in resources)


def prompts_paragraph(info: dict[str, Any]) -> str:
    prompts = info.get("ai_prompts", [])
    if not prompts:
        return (
            "If AI tools were used during the project, the related prompts can be listed in this "
            "section in table form. At this stage, a detailed prompt list has not been added."
        )
    return "<br/>".join(f"- {item}" for item in prompts)


def maybe_image(path: Path, width_cm: float):
    if not path.exists():
        return None
    image = Image(str(path))
    image.drawWidth = width_cm * cm
    image.drawHeight = image.drawWidth * 0.72
    return image


def build_document():
    info = read_json(INFO_PATH)
    metrics = read_json(METRICS_PATH)
    report_text = read_text(REPORT_PATH).replace("\n", "<br/>")
    styles = build_styles()

    PDF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )

    story = []
    story.append(Paragraph("2025-2026 Spring Semester BIL216 Final Project", styles["TitleTR"]))
    story.append(
        Paragraph(
            "Phase 1 Technical Report: Sound Signal Analysis and Emotion Classification",
            styles["HeadingTR"],
        )
    )
    story.append(Paragraph("This report summarizes the first working version of the proposed system.", styles["SmallTR"]))
    story.append(metadata_table(info, metrics, styles))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("1. Introduction", styles["HeadingTR"]))
    story.append(
        Paragraph(
            "The aim of this project is to automatically classify five different emotions from speech "
            "recordings: neutral, happy, angry, sad, and surprised. In Phase 1, the goal is to build "
            "an initial working model and establish a signal-processing and machine-learning pipeline "
            "that can be improved in later stages of the project.",
            styles["BodyTR"],
        )
    )

    story.append(Paragraph("2. Dataset Characterization", styles["HeadingTR"]))
    story.append(
        Paragraph(
            class_distribution_paragraph(metrics),
            styles["BodyTR"],
        )
    )
    story.append(
        Paragraph(
            "This initial version is designed to process the recordings collected during the midterm "
            "project together with their emotion labels. The system supports both a folder-based "
            "dataset structure and a CSV-based labeling structure.",
            styles["BodyTR"],
        )
    )

    story.append(Paragraph("3. Methodology", styles["HeadingTR"]))
    story.append(
        Paragraph(
            "The system was developed in Python. In the first stage, speech recordings are loaded as "
            "single-channel signals at a fixed sampling rate, and a set of acoustic features is "
            "extracted. The selected features include zero-crossing rate, RMS energy, pitch statistics, "
            "spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, spectral "
            "contrast, MFCC coefficients, and delta MFCC coefficients.",
            styles["BodyTR"],
        )
    )
    story.append(
        Paragraph(
            "In the classification stage, several classical machine-learning models are compared. "
            "Support Vector Machines (SVM), Random Forest, K-Nearest Neighbors, and Logistic "
            "Regression are evaluated by cross-validation, and the best-performing configuration "
            "is selected. This approach provides a fast, interpretable, and extensible baseline "
            "for Phase 1.",
            styles["BodyTR"],
        )
    )

    story.append(Paragraph("4. Statistical Findings", styles["HeadingTR"]))
    story.append(
        Paragraph(
            "After feature extraction, the system can generate selected feature distribution plots and "
            "a confusion matrix to observe separability between emotional classes. If the training "
            "pipeline has been executed, these visual outputs are automatically saved in the `results/` folder.",
            styles["BodyTR"],
        )
    )
    feature_image = maybe_image(FEATURE_OVERVIEW_PATH, width_cm=15.5)
    if feature_image is not None:
        story.append(feature_image)
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("5. Classification Performance", styles["HeadingTR"]))
    if metrics:
        best_params = metrics.get("best_params", {})
        story.append(
            Paragraph(
                f"Initial test accuracy: {fmt(metrics.get('test_accuracy'))}. "
                f"Best cross-validation score: {fmt(metrics.get('best_model_score_cv'))}. "
                f"Selected parameters: {best_params}.",
                styles["BodyTR"],
            )
        )
    else:
        story.append(
            Paragraph(
                "Since the real dataset has not yet been placed in this workspace, numerical performance "
                "values could not be generated automatically at this stage. However, the baseline pipeline "
                "is ready, and once the dataset is added, the accuracy, classification report, and "
                "confusion matrix can be produced with a single command.",
                styles["BodyTR"],
            )
        )

    cm_image = maybe_image(CONFUSION_MATRIX_PATH, width_cm=13.2)
    if cm_image is not None:
        story.append(cm_image)
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("6. Error Analysis and Discussion", styles["HeadingTR"]))
    story.append(
        Paragraph(
            "In speech-based emotion recognition, some emotions may be acoustically similar. In "
            "particular, neutral and sad may be confused because of lower energy levels and limited "
            "pitch variation. Although angry and happy can often be separated through increased energy, "
            "speaker differences and recording quality may still affect the results. In later phases, "
            "the system can be improved through feature selection, better class balance, and "
            "speaker-independent evaluation.",
            styles["BodyTR"],
        )
    )

    story.append(Paragraph("7. Classification Report Summary", styles["HeadingTR"]))
    story.append(Paragraph(report_text, styles["BodyTR"]))

    story.append(Paragraph("8. GitHub Link", styles["HeadingTR"]))
    story.append(Paragraph(fmt(info.get("github_url")), styles["BodyTR"]))

    story.append(Paragraph("9. Resources Used", styles["HeadingTR"]))
    story.append(Paragraph(resources_paragraph(info), styles["BodyTR"]))

    story.append(Paragraph("10. AI Prompts", styles["HeadingTR"]))
    story.append(Paragraph(prompts_paragraph(info), styles["BodyTR"]))

    story.append(Paragraph("11. Team Members", styles["HeadingTR"]))
    story.append(Paragraph(team_members_paragraph(info), styles["BodyTR"]))

    story.append(Paragraph("12. Team Member Contributions", styles["HeadingTR"]))
    story.append(Paragraph(contributions_paragraph(info), styles["BodyTR"]))

    doc.build(story)


if __name__ == "__main__":
    build_document()
    print(f"Created: {PDF_OUTPUT_PATH}")
