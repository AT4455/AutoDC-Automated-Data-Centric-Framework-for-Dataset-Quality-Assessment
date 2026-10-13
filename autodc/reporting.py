import json
from io import BytesIO
from datetime import datetime, timezone
import numpy as np

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, "isoformat"): return obj.isoformat()
    return obj

def make_pdf_report(report, audit):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleAuto", parent=styles["Title"], textColor=colors.HexColor("#17365D"),
                              fontSize=20, leading=25, spaceAfter=14))
    story = [Paragraph("AutoDC Dataset Quality Report", styles["TitleAuto"]),
             Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()}", styles["Normal"]),
             Spacer(1, 12),
             Paragraph(f"Rows: {report.get('rows','N/A')} | Columns: {report.get('columns','N/A')} | Quality score: {report.get('quality_score','N/A')}/100", styles["Heading2"]),
             Spacer(1, 8),
             Paragraph("Summary", styles["Heading1"])]
    summary = [
        ["Measure","Value"],
        ["Missing cells", str(report.get("missing_cells","N/A"))],
        ["Duplicate rows", str(report.get("duplicate_rows","N/A"))],
        ["Outlier candidate rows", str(report.get("outlier_rows","N/A"))],
        ["Constant columns", ", ".join(report.get("constant_columns",[])) or "None"],
    ]
    t = Table(summary, colWidths=[7*cm, 8*cm], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24476B")),
                           ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                           ("GRID",(0,0),(-1,-1),0.4,colors.grey),
                           ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F6FA")]),
                           ("PADDING",(0,0),(-1,-1),6)]))
    story += [t, Spacer(1, 12), Paragraph("Quality dimensions", styles["Heading1"])]
    dims = [["Dimension","Score / 100"]] + [[str(k), f"{float(v):.2f}"] for k,v in report.get("dimension_scores",{}).items()]
    t2 = Table(dims, colWidths=[7*cm,8*cm], repeatRows=1)
    t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24476B")),
                            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                            ("GRID",(0,0),(-1,-1),0.4,colors.grey),
                            ("PADDING",(0,0),(-1,-1),5)]))
    story += [t2, Spacer(1, 12), Paragraph("Audit trail", styles["Heading1"])]
    rows = [["Action","Details","Rows before","Rows after"]]
    for item in audit:
        rows.append([str(item.get("action","")), str(item.get("details",""))[:120],
                     str(item.get("rows_before","")), str(item.get("rows_after",""))])
    t3 = Table(rows, colWidths=[3.5*cm,7.5*cm,2*cm,2*cm], repeatRows=1)
    t3.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#24476B")),
                            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                            ("GRID",(0,0),(-1,-1),0.35,colors.grey),
                            ("VALIGN",(0,0),(-1,-1),"TOP"),
                            ("FONTSIZE",(0,0),(-1,-1),7),
                            ("PADDING",(0,0),(-1,-1),4)]))
    story.append(t3)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: this report is a data-quality aid, not a guarantee of model performance. Review flagged observations before removal.", styles["Italic"]))
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm)
    doc.build(story)
    return buffer.getvalue()
