"""Build the submission deck from the organisers' template (python-pptx, native shapes).

    python docs/build_deck.py --template Submission_Final_forGithub.pptx --out docs/CollegeName_codeCrafters.pptx
    (numbers come from docs/deck_numbers.json; team details from docs/team.json)
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
BG, PANEL, PANEL2, EDGE = "0A0E1A", "111A33", "16244A", "24365F"
CYAN, BLUE, AMBER, TEAL, GREEN, RED = "00B4D8", "1565C0", "F59E0B", "2EC4B6", "34C77B", "F26363"
WHITE, TEXT, MUTED = "FFFFFF", "E8EDF5", "9AA8C1"
FONT = "Calibri"


def rgb(h):
    return RGBColor.from_string(h)


def box(slide, x, y, w, h, fill=PANEL, line=EDGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    s.text_frame.text = ""
    return s


def text(slide, x, y, w, h, runs, size=14, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         font=FONT, name=None, space_after=4):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        bullet = False
        if isinstance(para, dict):
            bullet, para = para.get("bullet", False), para["t"]
        items = para if isinstance(para, list) else [(para, {})]
        for t, o in items:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color))
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.18)))); pPr.set("indent", str(-Emu(Inches(0.16))))
            for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
                for e in pPr.findall(qn(tag)):
                    pPr.remove(e)
            bc = pPr.makeelement(qn("a:buClr"), {}); sc = bc.makeelement(qn("a:srgbClr"), {"val": CYAN}); bc.append(sc)
            pPr.append(bc)
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
    return tb


def arrow(slide, x1, y1, x2, y2, color=CYAN, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def set_run_text(shape, new):
    """Replace a template text box's text, keeping its first run's formatting."""
    p = shape.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = new
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def stat(slide, x, y, w, big, label, color=CYAN):
    box(slide, x, y, w, 1.15, fill=PANEL)
    text(slide, x + 0.15, y + 0.1, w - 0.3, 0.6, big, size=30, bold=True, color=color)
    text(slide, x + 0.15, y + 0.72, w - 0.3, 0.4, label, size=11, color=MUTED)


def card(slide, x, y, w, h, title, body, accent=CYAN, num=None, body_size=12):
    box(slide, x, y, w, h, fill=PANEL)
    tx = x + 0.18
    if num is not None:
        c = box(slide, x + 0.16, y + 0.16, 0.38, 0.38, fill=accent, line=None, shape=MSO_SHAPE.OVAL)
        text(slide, x + 0.16, y + 0.16, 0.38, 0.38, str(num), size=13, bold=True, color=BG,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        tx = x + 0.66
    text(slide, tx, y + 0.17, x + w - tx - 0.12, 0.4, title, size=14, bold=True, color=WHITE)
    text(slide, x + 0.18, y + 0.62, w - 0.32, h - 0.7, body, size=body_size, color=TEXT, space_after=3)


def build(a):
    N = json.loads(Path(a.numbers).read_text())
    T = json.loads(Path(a.team).read_text())
    prs = Presentation(a.template)
    s1, s2, s3, s4, s5, s6, s7, s8, s9 = prs.slides

    # ---- 1. title
    for sh in s1.shapes:
        if sh.name == "Text 7":
            set_run_text(sh, "TRACK 2  |  EchoGesture")
        elif sh.name == "Text 8":
            set_run_text(sh, "EchoGesture")
        elif sh.name == "Text 10":
            tf = sh.text_frame
            ps = tf.paragraphs
            ps[0].runs[0].text = f"Team: {T['team']}"
            ps[1].runs[0].text = T["college"] if len(ps) > 1 and ps[1].runs else T["college"]
        elif sh.name == "Text Box 22":
            tf = sh.text_frame
            tf.paragraphs[0].runs[0].text = "Members"
            sh.left = Inches(6.55)
    text(s1, 0.55, 3.62, 4.6, 0.35, "On-device sign language captions for XR and edge devices",
         size=13, color=CYAN)
    mem = [[(m, {"size": 14, "color": TEXT})] for m in T["members"]]
    text(s1, 6.62, 2.95, 2.0, 1.2, mem, size=14, color=TEXT, space_after=4)
    text(s1, 0.55, 4.55, 7.8, 0.4, f"Samsung R&D Institute India  |  XR Hackathon 2026  |  {T['repo']}",
         size=10, color=MUTED)

    # ---- 2. problem
    text(s2, 0.45, 1.1, 5.0, 0.5, "Deaf signers and hearing people still need an interpreter to talk",
         size=20, bold=True, color=WHITE)
    text(s2, 0.45, 1.85, 5.0, 2.8, [
        {"t": "Captions must keep up with conversation: a sign has to become text within ~50 ms of ending, "
              "or turn-taking breaks.", "bullet": True},
        {"t": "Bystander video is private. It must never leave the phone or headset, so no cloud models.",
         "bullet": True},
        {"t": "Real cameras are exocentric: angled signers, moving heads, low light, hands leaving the frame "
              "or covered.", "bullet": True},
        {"t": "The device is small: an INT8 model under 20 MB that runs for minutes without draining the "
              "battery or throttling.", "bullet": True},
    ], size=13, space_after=8)
    stat(s2, 5.85, 1.1, 3.6, "<= 50 ms", "per-sign latency budget")
    stat(s2, 5.85, 2.4, 3.6, "0 frames", "may leave the device (zero cloud streaming)")
    stat(s2, 5.85, 3.7, 3.6, "<= 20 MB", "quantised INT8 model")

    # ---- 3. solution
    text(s3, 0.45, 1.05, 9.1, 0.5, "Landmarks, not pixels: a 2 MB INT8 model reads the body, never the picture",
         size=18, bold=True, color=WHITE)
    cw, gap, y = 2.12, 0.18, 1.7
    cards = [
        ("See", "MediaPipe Holistic on-device: 129 face, hand and pose points per frame. "
                "The frame is dropped right after.", CYAN),
        ("Normalise", "Centre and scale on the face, mirror to right-handed, fill short landmark gaps, "
                      "auto-exposure for dark frames.", TEAL),
        ("Recognise", "1D-CNN + Transformer (1.9 M params) x 3 seeds, INT8 LiteRT, 2 views each. "
                      f"{N['model_size']} MB in total.", AMBER),
        ("Caption", "Word on a high-contrast HUD with confidence and alternatives; optional offline speech; "
                    "predictions.jsonl.", GREEN),
    ]
    for i, (t, b, c) in enumerate(cards):
        card(s3, 0.45 + i * (cw + gap), y, cw, 1.95, t, b, accent=c, num=i + 1, body_size=12)
    box(s3, 0.45, 3.85, 9.1, 1.3, fill=PANEL2)
    text(s3, 0.65, 3.97, 8.7, 1.05, [
        [("Training recipe  ", {"bold": True, "color": CYAN, "size": 12}),
         ("Kaggle GISLR (94k clips, 250 signs) -> WLASL2000 -> WLASL300; AWP + late dropout; "
          "augmentation that imitates detector failures (covered side, missed hands, dark frames).", {"size": 12})],
        [("Measured honestly  ", {"bold": True, "color": CYAN, "size": 12}),
         ("official WLASL test split only, nothing tuned on it; every robustness test runs the full video "
          "pipeline.", {"size": 12})],
    ], size=12, space_after=4)

    # ---- 4. architecture
    text(s4, 0.45, 0.98, 9.1, 0.35, "ON THE DEVICE  (inference, everything local)", size=11, bold=True, color=CYAN)
    stages = [("Camera / video", "15 fps, any size"), ("MediaPipe Holistic", "543 landmarks / frame\nframe discarded"),
              ("Features", "129 pts x (xy, dxy, d2xy)\n64-frame window"),
              ("INT8 ensemble", f"3 x 2.1 MB LiteRT\n2-view TTA"),
              ("Caption HUD", "word + confidence\nspeech, jsonl")]
    bw, bh, gx, y0 = 1.6, 1.05, 0.275, 1.38
    for i, (t, sub) in enumerate(stages):
        x = 0.45 + i * (bw + gx)
        box(s4, x, y0, bw, bh, fill=PANEL2 if i not in (1, 3) else "163A5C", line=CYAN if i in (1, 3) else EDGE)
        text(s4, x + 0.1, y0 + 0.1, bw - 0.2, 0.35, t, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        text(s4, x + 0.1, y0 + 0.45, bw - 0.2, 0.55, sub.split("\n"), size=10, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            arrow(s4, x + bw + 0.02, y0 + bh / 2, x + bw + gx - 0.02, y0 + bh / 2)
    text(s4, 0.45, 2.52, 9.1, 0.3, f"Only text leaves the pipeline.  Model {N['latency_p95']} ms p95, "
         f"landmarks {N['landmark_ms']} ms/frame on a CPU.  Sign boundaries found automatically "
         "(hands enter, move, rest or leave).", size=10, color=MUTED)
    text(s4, 0.45, 2.98, 9.1, 0.35, "OFFLINE TRAINING  (Kaggle GPU, reproducible notebooks)", size=11, bold=True,
         color=AMBER)
    tstages = [("GISLR pretrain", "94k clips, 250 signs"), ("WLASL2000", "all glosses"),
               ("WLASL300 x 3 seeds", "train+val, 120 epochs"), ("INT8 PTQ", "calibrated, 99-100%\nagreement"),
               ("Evaluate", "official test, 5\nperturbations")]
    y1 = 3.38
    for i, (t, sub) in enumerate(tstages):
        x = 0.45 + i * (bw + gx)
        box(s4, x, y1, bw, 0.95, fill=PANEL, line=EDGE)
        text(s4, x + 0.1, y1 + 0.09, bw - 0.2, 0.33, t, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        text(s4, x + 0.1, y1 + 0.43, bw - 0.2, 0.5, sub.split("\n"), size=10, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(tstages) - 1:
            arrow(s4, x + bw + 0.02, y1 + 0.475, x + bw + gx - 0.02, y1 + 0.475, color=AMBER)
    text(s4, 0.45, 4.48, 9.1, 0.7, "One shared features.py is used by training and by predict(), so the model "
         "sees exactly the same numbers in both.  Stack: Python 3.11, MediaPipe Tasks, LiteRT (TFLite) INT8 + "
         "XNNPACK, Keras 3 / TensorFlow for training, OpenCV.", size=11, color=TEXT)

    # ---- 5. track output: results.csv table + robustness chart + predict() sample
    for sl, title in ((s5, "TRACK FINAL OUTPUT"), (s8, "BROWNIE: BEYOND THE BRIEF")):
        for sh in sl.shapes:
            if sh.name == "Text 2":
                set_run_text(sh, title)
    rows = [("Metric", "Target", "Measured", "Pass")] + [tuple(r) for r in N["csv"]]
    tbl = s5.shapes.add_table(len(rows), 4, Inches(0.45), Inches(1.05), Inches(4.75), Inches(0.3 * len(rows))).table
    widths = [1.95, 0.8, 1.25, 0.75]
    for j, wv in enumerate(widths):
        tbl.columns[j].width = Inches(wv)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            cell = tbl.cell(i, j)
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(PANEL2 if i == 0 else (PANEL if i % 2 else "0E1630"))
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            tf = cell.text_frame
            tf.text = ""
            run = tf.paragraphs[0].add_run(); run.text = str(v)
            col = WHITE if i == 0 else TEXT
            if j == 3 and i > 0:
                col = GREEN if v == "PASS" else RED if v == "FAIL" else MUTED
            run.font.size = Pt(11); run.font.bold = (i == 0 or j == 3); run.font.name = FONT
            run.font.color.rgb = rgb(col)
            tcPr = cell._tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = tcPr.makeelement(qn(side), {"w": "6350"})
                sf = ln.makeelement(qn("a:solidFill"), {}); c_ = sf.makeelement(qn("a:srgbClr"), {"val": EDGE})
                sf.append(c_); ln.append(sf); tcPr.insert(0, ln) if False else tcPr.append(ln)
    text(s5, 0.45, 1.1 + 0.3 * len(rows) + 0.08, 4.75, 0.5, N["csv_note"], size=10, color=MUTED)

    cd = CategoryChartData()
    cd.categories = list(N["robust"].keys())
    cd.add_series("Top-1 %", list(N["robust"].values()))
    gf = s5.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(5.45), Inches(1.0), Inches(4.1), Inches(2.6), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = "End-to-end top-1 under each condition (317 test videos)"
    tr = ch.chart_title.text_frame.paragraphs[0].runs[0].font
    tr.size = Pt(10); tr.color.rgb = rgb(TEXT); tr.bold = True; tr.name = FONT
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = '0.0'; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(9); dl.font.color.rgb = rgb(TEXT); dl.font.name = FONT
    ser = plot.series[0]
    for k, cat in enumerate(N["robust"].keys()):
        pt = ser.points[k]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(CYAN if k == 0 else (AMBER if cat == "occlusion" else BLUE))
    va = ch.value_axis
    va.maximum_scale = 100; va.minimum_scale = 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("1E2C52")
    va.tick_labels.font.size = Pt(9); va.tick_labels.font.color.rgb = rgb(MUTED)
    va.format.line.fill.background()
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(9); ca.tick_labels.font.color.rgb = rgb(TEXT)
    ca.format.line.color.rgb = rgb(EDGE)
    text(s5, 5.45, 3.62, 4.1, 0.3, f"Mean drop {N['drop']} pt (v7: 9.6), just inside the 5 pt target; "
         "trained on really occluded, blurred and dark video.", size=10, color=MUTED)
    box(s5, 5.45, 4.0, 4.1, 1.25, fill="0E1630")
    text(s5, 5.58, 4.07, 3.9, 0.25, "predict(\"17713.mp4\")  ->  one line of predictions.jsonl (real output)",
         size=10, bold=True, color=CYAN)
    text(s5, 5.58, 4.35, 3.9, 0.9, N["sample_json"], size=9, color=TEXT, font="Courier New")

    # ---- 6. differentiation
    diffs = [
        ("Private by construction", "Frames go straight to landmarks and are discarded; only 129 points per "
                                    "frame and the final word exist in memory. No network calls.", CYAN),
        ("Tiny and fast", f"{N['model_size']} MB INT8 for the whole ensemble, {N['latency_p95']} ms p95 "
                          "on a CPU. Runs on any LiteRT device.", TEAL),
        ("Honest numbers", "Official test split only. We caught and removed a signer-overlap leak "
                           "(92% -> real 67%) before building on it.", AMBER),
        ("Robustness engineered", "Auto-exposure, gap filling and training on really occluded video: "
                                  "low light 13.6 -> 2.8 pt, occlusion 22.4 -> 13.6 pt.", GREEN),
        ("Same code everywhere", "One feature pipeline for training, predict(), the live app and the demo "
                                 "renderer: no train/serve skew.", BLUE),
        ("Usable output", "Live caption HUD with confidence and alternatives, transcript, optional speech "
                          "for two-way conversation.", CYAN),
    ]
    cw, ch_, gx, gy = 2.92, 1.85, 0.17, 0.25
    for i, (t, b, c) in enumerate(diffs):
        r_, c_ = divmod(i, 3)
        card(s6, 0.45 + c_ * (cw + gx), 1.08 + r_ * (ch_ + gy), cw, ch_, t, b, accent=c, num=i + 1, body_size=12)

    # ---- 7. demo & local setup
    shot = HERE / "hud_frame.png"
    if shot.exists():
        s7.shapes.add_picture(str(shot), Inches(0.45), Inches(1.05), width=Inches(4.6))
        text(s7, 0.45, 3.7, 4.6, 0.3, "Caption HUD: same code as the live app and the demo video", size=10, color=MUTED)
    text(s7, 0.45, 4.05, 4.6, 1.2, [
        [("Demo video  ", {"bold": True, "color": CYAN, "size": 12}), (T["video"], {"size": 12})],
        [("Code  ", {"bold": True, "color": CYAN, "size": 12}), (T["repo"], {"size": 12})],
    ], size=12, space_after=6)
    box(s7, 5.3, 1.05, 4.25, 3.35, fill="0E1630")
    text(s7, 5.45, 1.15, 4.0, 0.3, "Run it locally (Python 3.11)", size=12, bold=True, color=WHITE)
    code = ["pip install -r src/requirements.txt",
            "",
            "# evaluator: predict() on a folder",
            "python src/main.py --videos test/ --out results",
            "",
            "# live captions from the webcam",
            "python src/demo/live.py --tts",
            "",
            "# Docker",
            "docker build -f src/Dockerfile -t eg .",
            "docker run -v $PWD/test:/videos eg \\",
            "    --videos /videos"]
    text(s7, 5.45, 1.55, 4.0, 3.6, code, size=10, color=TEXT, font="Courier New", space_after=0)

    # ---- 8. brownie + limitations
    text(s8, 0.45, 1.0, 4.5, 0.35, "Beyond the brief", size=14, bold=True, color=AMBER)
    text(s8, 0.45, 1.4, 4.4, 2.6, [
        {"t": "Live caption app: automatic sign segmentation, HUD with confidence and top-3, rolling "
              "transcript, offline text-to-speech.", "bullet": True},
        {"t": "End-to-end robustness suite: low light, motion blur, occlusion, camera tilt and frame "
              "drops, all through the real video pipeline.", "bullet": True},
        {"t": "Low-light auto-exposure (dark frames only, ~1 ms) and training on MediaPipe output from really occluded, blurred and dark videos.", "bullet": True},
        {"t": "Reproducible Kaggle notebooks and a Docker image for the evaluator.", "bullet": True},
    ], size=12, space_after=6)
    text(s8, 5.15, 1.0, 4.4, 0.35, "Limitations and next steps", size=14, bold=True, color=CYAN)
    text(s8, 5.15, 1.4, 4.4, 2.6, [
        {"t": f"Occlusion still costs {N['occ_drop']} pt: when the signing hand is hidden mid-sign the "
              "information is gone. Next: hand-region attention and a temporal language prior.",
         "bullet": True},
        {"t": "On a Redmi Note 12 Pro: classifier 4.7 ms per call on the CPU, 14.7% slower after 10 min of load. Power needs an Android app (the phone blocks battery reads over adb).",
         "bullet": True},
        {"t": "Isolated signs (300 words). Next: continuous signing with CTC over the same features.",
         "bullet": True},
        {"t": "Port the same .tflite to Android / Galaxy XR with the MediaPipe Tasks SDK.", "bullet": True},
    ], size=12, space_after=6)
    for i, (big, lab) in enumerate([(f"{N['top1']}%", "top-1, official test videos"),
                                    (f"{N['top5']}%", "top-5"),
                                    (f"{N['latency_p95']} ms", "model p95"),
                                    (f"{N['model_size']} MB", "INT8 ensemble")]):
        stat(s8, 0.45 + i * 2.32, 4.1, 2.15, big, lab, color=[CYAN, TEAL, AMBER, GREEN][i])

    # ---- 9. thank you
    for sh in s9.shapes:
        if sh.name == "Text 9":
            sh.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    for sh in s9.shapes:
        if sh.name == "Text 25":
            p = sh.text_frame.paragraphs[0]
            r = p.add_run(); r.text = f"{T['team']}  |  {T['college']}  |  {T['repo']}"
            r.font.size = Pt(10); r.font.color.rgb = rgb(WHITE); r.font.name = FONT
    text(s9, 2.0, 3.3, 5.55, 0.5, "EchoGesture: captions that never leave the device", size=14,
         color=CYAN, align=PP_ALIGN.CENTER)

    prs.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--numbers", default=str(HERE / "deck_numbers.json"))
    ap.add_argument("--team", default=str(HERE / "team.json"))
    build(ap.parse_args())
