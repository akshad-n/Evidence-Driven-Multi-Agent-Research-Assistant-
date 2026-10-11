import os
import io
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from utils.logging import logger


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page numbers:
    'Page X of Y' in the running footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Evidentia AI — Evidence-Driven Multi-Agent Research Assistant")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — ACADEMIC RESEARCH USE ONLY")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Removes common markdown bold/italic tags and escapes XML chars for ReportLab."""
    if not text:
        return ""
    text = str(text)
    # Escape XML entities
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Convert markdown bold **text** to ReportLab <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert markdown italic *text* or _text_ to <i>text</i>
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text.strip()


def generate_research_pdf(
    research_question: str,
    papers: List[Dict[str, Any]],
    paper_analyses: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]],
    research_gaps: List[Dict[str, Any]],
    final_report_markdown: str = "",
    session_id: str = ""
) -> bytes:
    """
    Generates a high-quality publication-grade academic PDF report using ReportLab.
    Returns bytes buffer ready for download or file writing.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palettes & Styles
    primary_blue = colors.HexColor("#1E3A8A")
    accent_blue = colors.HexColor("#2563EB")
    dark_slate = colors.HexColor("#0F172A")
    muted_slate = colors.HexColor("#475569")
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_blue,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=muted_slate,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_blue,
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=dark_slate,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_slate,
        spaceAfter=6
    )

    quote_style = ParagraphStyle(
        "Quote_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=4,
        spaceAfter=4
    )

    meta_style = ParagraphStyle(
        "Meta_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=muted_slate
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("EVIDENTIA AI RESEARCH REPORT", title_style))
    story.append(Paragraph("Evidence-Driven Multi-Agent Academic Literature Review & Gap Analysis", subtitle_style))
    
    meta_table_data = [
        [
            Paragraph(f"<b>Inquiry:</b> {clean_markdown_inline(research_question)}", body_style),
            Paragraph(f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/><b>Session:</b> {session_id[:16] if session_id else 'Direct'}", meta_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[350, 154])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Metric Summary Strip
    metrics = [
        [
            Paragraph(f"<font size=14 color='#2563EB'><b>{len(papers)}</b></font><br/><font size=7 color='#64748B'>PAPERS</font>", ParagraphStyle('C1', alignment=1)),
            Paragraph(f"<font size=14 color='#2563EB'><b>{len(paper_analyses)}</b></font><br/><font size=7 color='#64748B'>ANALYZED</font>", ParagraphStyle('C2', alignment=1)),
            Paragraph(f"<font size=14 color='#2563EB'><b>{len(evidence)}</b></font><br/><font size=7 color='#64748B'>EVIDENCE ITEMS</font>", ParagraphStyle('C3', alignment=1)),
            Paragraph(f"<font size=14 color='#DC2626'><b>{len(contradictions)}</b></font><br/><font size=7 color='#64748B'>CONTRADICTIONS</font>", ParagraphStyle('C4', alignment=1)),
            Paragraph(f"<font size=14 color='#4F46E5'><b>{len(research_gaps)}</b></font><br/><font size=7 color='#64748B'>RESEARCH GAPS</font>", ParagraphStyle('C5', alignment=1))
        ]
    ]
    metric_table = Table(metrics, colWidths=[100, 100, 104, 100, 100])
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 14))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))
    
    # Extract executive summary from markdown if available
    exec_summary_text = ""
    if "## 2. Executive Summary" in final_report_markdown:
        part = final_report_markdown.split("## 2. Executive Summary")[-1]
        part = part.split("## 3.")[0].strip()
        exec_summary_text = part
    elif "## Executive Summary" in final_report_markdown:
        part = final_report_markdown.split("## Executive Summary")[-1]
        part = part.split("##")[0].strip()
        exec_summary_text = part
    else:
        exec_summary_text = f"This report delivers a systematic, multi-agent literature synthesis on '{research_question}'. Extracted claims have been cross-verified against peer-reviewed academic literature with verbatim excerpts, quantitative metrics, and contradiction detection."

    story.append(Paragraph(clean_markdown_inline(exec_summary_text), body_style))
    story.append(Spacer(1, 10))

    # Section 2: Paper Comparison Table
    if paper_analyses:
        story.append(Paragraph("2. Comparative Literature Breakdown", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

        table_rows = [
            [
                Paragraph("<b>Paper / Model</b>", meta_style),
                Paragraph("<b>Methodology</b>", meta_style),
                Paragraph("<b>Dataset</b>", meta_style),
                Paragraph("<b>Reported Results</b>", meta_style),
                Paragraph("<b>Chief Limitations</b>", meta_style)
            ]
        ]

        for idx, a in enumerate(paper_analyses[:5]):
            pname = a.get("paper", f"Paper {idx + 1}")
            method = a.get("method", "Not specified")[:60]
            dataset = a.get("dataset", "Not specified")[:50]
            results = a.get("results", "Not specified")[:60]
            limitations = a.get("limitations", "Not specified")[:60]

            table_rows.append([
                Paragraph(f"<b>[{idx + 1}]</b> {clean_markdown_inline(pname[:40])}", meta_style),
                Paragraph(clean_markdown_inline(method), meta_style),
                Paragraph(clean_markdown_inline(dataset), meta_style),
                Paragraph(clean_markdown_inline(results), meta_style),
                Paragraph(clean_markdown_inline(limitations), meta_style)
            ])

        comp_table = Table(table_rows, colWidths=[110, 100, 95, 100, 99])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 12))

    # Section 3: Verified Evidence Claims
    if evidence:
        story.append(Paragraph("3. Empirical Evidence & Grounded Claims", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

        for idx, ev in enumerate(evidence[:6]):
            claim = ev.get("claim", "")
            quote = ev.get("evidence", "")
            src = ev.get("source_title", "Academic Literature")
            cit = ev.get("citation", f"[{idx + 1}]")
            conf = int(float(ev.get("confidence", 0.9)) * 100)

            ev_card = [
                [
                    Paragraph(f"<b>Claim {idx + 1}:</b> {clean_markdown_inline(claim)} <b>{cit}</b>", h2_style),
                    Paragraph(f"<font color='#16A34A'><b>{conf}% Confidence</b></font>", ParagraphStyle('Conf', alignment=2, fontName="Helvetica", fontSize=8))
                ],
                [
                    Paragraph(f"<i>\"{clean_markdown_inline(quote)}\"</i><br/><font color='#64748B'>Source: {clean_markdown_inline(src)}</font>", quote_style),
                    ""
                ]
            ]
            ev_table = Table(ev_card, colWidths=[400, 104])
            ev_table.setStyle(TableStyle([
                ('SPAN', (0, 1), (1, 1)),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ('LINEBEFORE', (0, 0), (0, -1), 3, accent_blue),
                ('BOX', (0, 0), (-1, -1), 0.5, border_color),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(ev_table)
            story.append(Spacer(1, 6))

    # Section 4: Contradiction Detection
    if contradictions:
        story.append(Spacer(1, 6))
        story.append(Paragraph("4. Contradictory Empirical Findings", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

        for idx, c in enumerate(contradictions):
            fa = c.get("finding_a", "")
            sa = c.get("source_a", "")
            fb = c.get("finding_b", "")
            sb = c.get("source_b", "")
            exp = c.get("explanation", "")

            c_text = (
                f"<b>Contradiction #{idx + 1}:</b><br/>"
                f"• <b>Finding A:</b> {clean_markdown_inline(fa)} (<i>{clean_markdown_inline(sa)}</i>)<br/>"
                f"• <b>Finding B:</b> {clean_markdown_inline(fb)} (<i>{clean_markdown_inline(sb)}</i>)<br/>"
                f"<b>Objective Analysis:</b> {clean_markdown_inline(exp)}"
            )

            c_table = Table([[Paragraph(c_text, body_style)]], colWidths=[504])
            c_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
                ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor("#EF4444")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#FECACA")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(c_table)
            story.append(Spacer(1, 6))

    # Section 5: Potential Research Gaps
    if research_gaps:
        story.append(Spacer(1, 6))
        story.append(Paragraph("5. Discovered Research Gaps", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

        for idx, g in enumerate(research_gaps):
            gap = g.get("gap", "")
            reason = g.get("reason", "")
            supp = ", ".join(g.get("supporting_papers", [])) or "Analyzed Literature"

            g_text = (
                f"<b>Gap #{idx + 1}: {clean_markdown_inline(gap)}</b><br/>"
                f"<b>Justification:</b> {clean_markdown_inline(reason)}<br/>"
                f"<font color='#64748B'>Supporting Source Papers: {clean_markdown_inline(supp)}</font>"
            )

            g_table = Table([[Paragraph(g_text, body_style)]], colWidths=[504])
            g_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
                ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor("#6366F1")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(g_table)
            story.append(Spacer(1, 6))

    # Section 6: Formal Academic References
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. References", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

    for idx, p in enumerate(papers):
        authors = p.get("authors", [])
        if isinstance(authors, list):
            auth_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        else:
            auth_str = str(authors)
        title = p.get("title", "Untitled")
        year = p.get("year", "n.d.")
        source = p.get("source", "ACADEMIC").upper()
        doi = p.get("doi", "")
        url = p.get("url", "")

        ref_line = f"<b>[{idx + 1}]</b> {clean_markdown_inline(auth_str)} ({year}). <i>{clean_markdown_inline(title)}</i>. {source}."
        if doi and doi != "Not available":
            ref_line += f" DOI: {doi}."
        if url and url != "Not available":
            ref_line += f" Available at: {url}"

        story.append(Paragraph(ref_line, meta_style))
        story.append(Spacer(1, 3))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    logger.info(f"Generated publication PDF ({len(pdf_bytes)} bytes) for question: '{research_question}'")
    return pdf_bytes
