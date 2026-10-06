"""
reports.py
-----------
Milestone 4, Task 6: Meeting report generation and export (PDF + CSV).

Both formats are built from the same `report` dict (see
services.AppServices.build_report), so the CSV and PDF can never disagree.
Contents: meeting details, summary, key points, decisions, action items,
participants and deadlines - for exactly ONE meeting.
"""

import csv
import io
import logging
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["Section", "Item", "Assignee", "Deadline", "Priority", "Status", "Details"]
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection: a transcript-derived cell that
    starts with = + - @ would otherwise be executed by Excel/Sheets on open."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def report_filename(report: dict, extension: str) -> str:
    base = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", report["meeting"]["filename"])
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base).strip("_")[:60] or "meeting"
    return f"meeting_{report['meeting']['id']}_{base}.{extension}"


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def report_to_csv(report: dict) -> bytes:
    """UTF-8 with BOM so Excel opens non-English text correctly."""
    m = report["meeting"]
    rows = []

    def add(section, item="", assignee="", deadline="", priority="", status="", details=""):
        rows.append([section, item, assignee, deadline, priority, status, details])

    duration = m.get("audio_duration_seconds")
    add("Meeting Details", "Meeting ID", details=m["id"])
    add("Meeting Details", "File name", details=m["filename"])
    add("Meeting Details", "Date", details=m["created_at"])
    add("Meeting Details", "Language", details=m.get("language") or "")
    add("Meeting Details", "Duration (minutes)", details=round(duration / 60.0, 1) if duration else "")
    add("Meeting Details", "Source", details=m.get("source") or "upload")
    add("Summary", "Summary", details=report["summary_text"] or "")
    for i, point in enumerate(report["key_points"], 1):
        add("Key Points", f"Key point {i}", details=point)
    for i, decision in enumerate(report["decisions"], 1):
        add("Decisions", f"Decision {i}", details=decision)
    for item in report["action_items"]:
        add("Action Items", item.get("task", ""), item.get("participant_name") or "Unassigned",
            item.get("deadline") or "", item.get("priority") or "", item.get("status") or "")
    for name in report["participants"]:
        add("Participants", name)
    for deadline in report["deadlines"]:
        add("Deadlines", deadline)

    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([_csv_safe(cell) for cell in row])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


# ---------------------------------------------------------------------------
# PDF (reportlab)
# ---------------------------------------------------------------------------

_UNICODE_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def _register_fonts():
    """Returns (regular, bold) font names. Uses a system TrueType font when one
    is found (so accented / non-Latin text renders), else built-in Helvetica."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for regular, bold in _UNICODE_FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("ReportBody", regular))
                pdfmetrics.registerFont(TTFont("ReportBold", bold))
                return "ReportBody", "ReportBold"
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not load font %s: %s", regular, e)
    return "Helvetica", "Helvetica-Bold"


def report_to_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    body_font, bold_font = _register_fonts()
    teal, amber = colors.HexColor("#0F766E"), colors.HexColor("#B45309")

    title = ParagraphStyle("t", fontName=bold_font, fontSize=20, textColor=teal, spaceAfter=4, leading=24)
    sub = ParagraphStyle("s", fontName=body_font, fontSize=9, textColor=colors.HexColor("#6B7280"), spaceAfter=10)
    h = ParagraphStyle("h", fontName=bold_font, fontSize=12.5, textColor=teal, spaceBefore=12, spaceAfter=5)
    body = ParagraphStyle("b", fontName=body_font, fontSize=10, leading=14)
    bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2)
    cell = ParagraphStyle("c", fontName=body_font, fontSize=9, leading=12)
    cell_head = ParagraphStyle("ch", fontName=bold_font, fontSize=9, textColor=colors.white, leading=12)

    def p(text, style=body):
        return Paragraph(escape(str(text if text is not None else "")).replace("\n", "<br/>"), style)

    m = report["meeting"]
    duration = m.get("audio_duration_seconds")
    story = [
        p("Meeting Report", title),
        p(f"Generated {datetime.now():%Y-%m-%d %H:%M} · Career Intelligence Platform", sub),
    ]

    details = [
        ["Meeting ID", m["id"]], ["File", m["filename"]], ["Date", m["created_at"][:19].replace("T", " ")],
        ["Language", m.get("language") or "-"],
        ["Duration", f"{duration / 60.0:.1f} min" if duration else "-"],
        ["Source", m.get("source") or "upload"],
    ]
    t = Table([[p(k, cell), p(v, cell)] for k, v in details], colWidths=[35 * mm, 135 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0FDFA")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [p("Meeting Details", h), t]

    story += [p("Summary", h), p(report["summary_text"] or "No summary available.")]

    def bullet_section(heading, items, empty):
        story.append(p(heading, h))
        if items:
            for it in items:
                story.append(Paragraph(escape(str(it)), bullet, bulletText="•"))
        else:
            story.append(p(empty))

    bullet_section("Key Points", report["key_points"], "None recorded.")
    bullet_section("Decisions", report["decisions"], "No decisions recorded.")

    story.append(p("Action Items", h))
    if report["action_items"]:
        data = [[p(c, cell_head) for c in ("Task", "Assignee", "Deadline", "Priority", "Status")]]
        for i in report["action_items"]:
            data.append([p(i.get("task", ""), cell), p(i.get("participant_name") or "Unassigned", cell),
                         p(i.get("deadline") or "-", cell), p(i.get("priority") or "-", cell),
                         p(i.get("status") or "-", cell)])
        at = Table(data, colWidths=[62 * mm, 30 * mm, 27 * mm, 22 * mm, 29 * mm], repeatRows=1)
        at.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), teal),
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                                ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(at)
    else:
        story.append(p("No action items."))

    bullet_section("Participants", report["participants"], "No participants identified.")
    bullet_section("Deadlines", report["deadlines"], "No deadlines identified.")
    story.append(Spacer(1, 6 * mm))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=f"Meeting Report - {m['filename']}", author="Career Intelligence Platform")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(body_font, 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Meeting #{m['id']} · page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
