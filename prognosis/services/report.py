
from fpdf import FPDF

from ..config import ROOT

FONT_DIR = ROOT / "assets" / "fonts"
BAND_RGB = {"low": (46, 125, 50), "moderate": (230, 140, 0), "high": (198, 40, 40)}


class _Pdf(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("dv", "", 8)
        self.set_text_color(110)
        self.cell(0, 6, f"Prognosis decision-support report. Page {self.page_no()}", align="C")


def build_pdf(patient: dict, pred: dict, summary: str, card: dict) -> bytes:
    pdf = _Pdf()
    # a real unicode font, so names and symbols are not silently stripped like the old latin-1 hack did
    pdf.add_font("dv", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("dv", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    pdf.set_font("dv", "B", 17)
    pdf.cell(0, 10, f"{pred['title']} risk report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("dv", "", 10)
    pdf.set_text_color(90)
    pdf.cell(0, 6, f"Patient: {patient['name']}   MRN: {patient['mrn']}   Sex: {patient['sex']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {pred['created_at']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    r, g, b = BAND_RGB[pred["band"]]
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255)
    pdf.set_font("dv", "B", 14)
    pdf.cell(0, 12, f"  Estimated probability {pred['probability']:.1%}  |  {pred['band'].upper()} risk band", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(4)

    pdf.set_font("dv", "B", 11)
    pdf.cell(0, 7, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("dv", "", 10)
    pdf.multi_cell(0, 5.5, summary, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # inputs that were left empty are not shown as drivers, their "effect" only reflects the fill-in value
    contribs = [c for c in pred.get("contributions", []) if not c["imputed"] and abs(c["effect"]) >= 0.0005][:8]
    n_empty = sum(1 for c in pred.get("contributions", []) if c["imputed"])
    if contribs:
        pdf.set_font("dv", "B", 11)
        pdf.cell(0, 7, "What moved the estimate", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("dv", "", 9)
        pdf.set_text_color(90)
        pdf.multi_cell(0, 4.5, f"Change in probability, in percentage points, compared with a typical training patient (average {pred.get('baseline', 0):.1%}).", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(1)
        top = max(abs(c["effect"]) for c in contribs)
        x0 = pdf.l_margin + 78
        for c in contribs:
            y = pdf.get_y()
            val = f"{c['value']:g}" if isinstance(c["value"], float) else str(c["value"])
            pdf.set_font("dv", "", 9)
            pdf.cell(78, 6, f"{c['label']}: {val} {c['unit']}".strip()[:44])
            w = 60 * abs(c["effect"]) / top
            pdf.set_fill_color(*((198, 40, 40) if c["effect"] > 0 else (30, 100, 180)))
            if c["effect"] > 0:
                pdf.rect(x0 + 30, y + 1, w, 4, style="F")
            else:
                pdf.rect(x0 + 30 - w, y + 1, w, 4, style="F")
            pdf.set_xy(x0 + 95, y)
            pdf.cell(20, 6, f"{c['effect'] * 100:+.1f}")
            pdf.ln(6)
        if n_empty:
            pdf.set_font("dv", "", 8)
            pdf.set_text_color(110)
            pdf.cell(0, 5, f"{n_empty} input(s) were not given and are filled with typical training values.", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0)
        pdf.ln(2)

    if pred.get("warnings"):
        pdf.set_font("dv", "B", 10)
        pdf.cell(0, 6, "Cautions on this estimate", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("dv", "", 9)
        for w in pred["warnings"]:
            pdf.multi_cell(0, 4.8, f"- {w}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    t = card["test"]["metrics"]
    lo, hi = card["test"]["ci95"]["roc_auc"]
    pdf.set_font("dv", "B", 10)
    pdf.cell(0, 6, "About the model", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("dv", "", 9)
    pdf.multi_cell(0, 4.8,
        f"{card['selected']}{'' if card['calibration'] == 'none' else ' with ' + card['calibration'] + ' calibration'}, trained {card['trained_at'][:10]} on {card['data']['train_rows']} records. "
        f"Held-out test AUC {t['roc_auc']:.3f} (95% CI {lo:.3f} to {hi:.3f}), Brier {t['brier']:.3f}. "
        f"Training data: {card['label']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("dv", "", 8)
    pdf.set_text_color(110)
    pdf.multi_cell(0, 4.2,
        "This report is decision support only and is not a diagnosis. The model was trained on public research datasets, "
        "not on this clinic's population, and has not been clinically validated. Probabilities describe the training data's "
        "case mix. A clinician should interpret them together with the whole clinical picture.", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
