"""Build the 50-sample human-evaluation workbook (plan Tier 2: "Human evaluation (50 samples)").

Samples: 50 of the 52 Qwen LLM drafts in experiments/results/generation_llm.json (both drafts where the LLM declined
are kept; 2 others are dropped with a fixed seed), shuffled so order carries no signal.
Sources: re-retrieved with the production retriever (same configuration as the run), so the [n] citations in each
draft can be checked; reproduction is verified against the stored `kb_retrieved` flag of every case.

Output: data/evaluation/human_eval_sheet.xlsx
  Instructions  how to rate, rubric, legend
  Ratings       one row per sample; yellow cells are the rater's inputs (dropdowns); row 2 is a worked example
  System        HIDDEN until rating is done: system decision, confidence, groundedness, auto metric, LLM-judge scores
  Summary       live formulas comparing human ratings with the system and the LLM judge
Analyse a filled copy with: python evaluation/human_eval_analysis.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import Workbook  # noqa: E402
from openpyxl.comments import Comment  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.database import dispose_engine, get_session_factory  # noqa: E402
from app.models.embeddings import get_embedding_service  # noqa: E402
from app.services.retrieval import HybridRetriever, RetrievalService  # noqa: E402
from evaluation.data import EVAL_DIR, RESULTS_DIR, load_eval  # noqa: E402

OUT = EVAL_DIR / "human_eval_sheet.xlsx"
N_SAMPLES, SEED = 50, 7
FIRST, LAST = 3, 3 + N_SAMPLES - 1  # data rows on Ratings/System (row 2 = example)

FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
INPUT_FILL = PatternFill("solid", fgColor="FFFF00")
EXAMPLE_FILL = PatternFill("solid", fgColor="EDEDED")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")


def excerpt(text: str, n: int = 230) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


async def retrieve_sources(complaints: list[str]) -> list[list]:
    sf = get_session_factory()
    service = RetrievalService(HybridRetriever(sf, get_embedding_service()), None, use_reranking=False)
    out = []
    for c in complaints:
        out.append((await service.retrieve(c, top_k=settings.RETRIEVAL_TOP_K)).items)
    await dispose_engine()
    return out


def style_header(ws, headers: list[str], widths: list[int]) -> None:
    for col, (h, w) in enumerate(zip(headers, widths), start=1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = BORDER
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 42


def main() -> None:
    import json

    rows = json.loads((RESULTS_DIR / "generation_llm.json").read_text(encoding="utf-8"))["rows"]
    cases = {c["id"]: c for c in load_eval("generation_eval")}
    rng = random.Random(SEED)
    empty = [r for r in rows if r["n_steps"] == 0]
    others = [r for r in rows if r["n_steps"] > 0]
    chosen = empty + rng.sample(others, N_SAMPLES - len(empty))
    rng.shuffle(chosen)

    sources = asyncio.run(retrieve_sources([cases[r["id"]]["complaint"] for r in chosen]))
    mismatches = [
        r["id"]
        for r, items in zip(chosen, sources)
        if (cases[r["id"]]["reference_kb_id"] in [i.document.id for i in items]) != bool(r["kb_retrieved"])
    ]
    if mismatches:
        raise SystemExit(f"retrieval did not reproduce the stored run for {mismatches}; refusing to build the sheet")

    wb = Workbook()
    # ------------------------------------------------------------------ Instructions
    ins = wb.active
    ins.title = "Instructions"
    ins.column_dimensions["A"].width = 120
    lines = [
        ("Human evaluation of draft resolutions (50 samples)", True),
        ("", False),
        (
            "Purpose: check whether the system's drafts are actually useful and safe, and whether its RESOLVE / REVIEW / "
            "ESCALATE decisions match what a support agent would do. Your ratings also calibrate the heuristic confidence "
            "score and validate the automatic LLM-judge scores.",
            False,
        ),
        ("", False),
        ("How to rate (Ratings sheet, about 1 minute per row):", True),
        (
            "1. Read the complaint (column B), then the draft (column D). Citations like [3] refer to the numbered sources "
            "in column C.",
            False,
        ),
        ("2. Use the reference fix (column E) as the ground truth for what actually resolves this issue.", False),
        ("3. Fill ONLY the yellow cells (F to K). Each has a dropdown, except Notes.", False),
        ("4. Row 2 is a worked EXAMPLE (grey) showing the expected format; it is not counted.", False),
        ("5. Rate independently: do not unhide the 'System' sheet or open 'Summary' until all rows are done.", False),
        ("", False),
        ("Rubric", True),
        (
            "Relevance (1-5): does the draft address this customer's actual problem? 1 = wrong problem, 5 = exactly it.",
            False,
        ),
        (
            "Completeness (1-5): are the key steps of the reference fix present? 1 = none, 5 = all essential steps.",
            False,
        ),
        ("Correctness (1-5): are the steps right and safe? 1 = wrong or harmful advice, 5 = fully correct.", False),
        ("Safe to use as-is (Yes/No): would you send these steps to the customer without editing?", False),
        (
            "Your decision: RESOLVE = use the draft; REVIEW = usable after an agent checks or edits it; "
            "ESCALATE = not usable, needs a specialist.",
            False,
        ),
        (
            "Empty drafts (the system declined): rate relevance/completeness/correctness as 1 and give your decision "
            "for the case (ESCALATE if declining was right).",
            False,
        ),
        ("Notes (optional): anything wrong or notable, e.g. 'step 2 contradicts the fix', 'wrong scenario'.", False),
        ("", False),
        ("Legend: yellow cell = your input. Grey row = example. Blue header = system-provided content.", False),
        ("When done: save the file and run  python evaluation/human_eval_analysis.py", False),
        (
            "Provenance: drafts from Groq qwen/qwen3.8-27b (eval run in experiments/results/generation_llm.json); "
            "sources re-retrieved with the production retriever and verified against that run; sample seed 7.",
            False,
        ),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ins.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, bold=bold, size=12 if i == 1 else 10)
        c.alignment = Alignment(wrap_text=True, vertical="top")

    # ------------------------------------------------------------------ Ratings
    rat = wb.create_sheet("Ratings")
    headers = [
        "Sample",
        "Customer complaint",
        "Sources the draft could cite",
        "Draft resolution (system)",
        "Reference fix (ground truth)",
        "Relevance\n(1-5)",
        "Completeness\n(1-5)",
        "Correctness\n(1-5)",
        "Safe to use\nas-is?",
        "Your decision",
        "Notes",
    ]
    style_header(rat, headers, [8, 45, 70, 60, 50, 11, 13, 12, 11, 13, 30])
    example = [
        "EX",
        "I can't set up my voicemail, callers just hear it ring out.",
        "[1] ticket TKT-001501: Complaint: my voicemail isn't working… Resolution: Activate voicemail on the line…\n"
        "[2] KB-0032 Setting up voicemail: 1. Activate voicemail… 2. Set call diversion on no answer…",
        "1. Activate voicemail on the line from the account system [1][2]\n"
        "2. Set call diversion on no answer to the voicemail number [2]",
        "- Activate voicemail on the line\n- Set call diversion on no answer\n- Reset the voicemail PIN and send it by SMS",
        4,
        3,
        5,
        "Yes",
        "RESOLVE",
        "Example only: missing the PIN reset step.",
    ]
    for col, v in enumerate(example, start=1):
        c = rat.cell(row=2, column=col, value=v)
        c.font = Font(name=FONT, size=9, italic=True, color="595959")
        c.fill = EXAMPLE_FILL
        c.alignment = WRAP_TOP
        c.border = BORDER
    rat.cell(row=2, column=1).comment = Comment("Worked example showing the expected format. Not counted.", "builder")

    sysw = wb.create_sheet("System")
    sys_headers = [
        "Sample",
        "Case ID",
        "Scenario",
        "Category",
        "System decision",
        "Heuristic confidence",
        "Groundedness",
        "Auto metric: reference-step recall",
        "LLM-judge relevance",
        "LLM-judge completeness",
        "LLM-judge specificity",
        "LLM-judge correctness",
        "Draft empty (declined)",
    ]
    style_header(sysw, sys_headers, [8, 11, 34, 20, 14, 12, 13, 16, 12, 13, 12, 12, 12])
    sysw.cell(row=2, column=1, value="EX").font = Font(name=FONT, size=9, italic=True, color="595959")

    for k, (r, items) in enumerate(zip(chosen, sources)):
        row = FIRST + k
        case = cases[r["id"]]
        src_text = "\n".join(
            f"[{n}] {it.document.doc_type.replace('_', ' ')} {it.document.id}: {excerpt(it.document.text)}"
            for n, it in enumerate(items, start=1)
        )
        draft = "\n".join(f"{n}. {s}" for n, s in enumerate(r["steps"], start=1)) or (
            "(no steps: the system declined) " + (r.get("summary") or "")
        )
        ref = "\n".join(f"- {s}" for s in case["reference_steps"])
        values = [f"H{k + 1:02d}", case["complaint"], src_text, draft, ref]
        for col, v in enumerate(values, start=1):
            c = rat.cell(row=row, column=col, value=v)
            c.font = Font(name=FONT, size=9)
            c.alignment = WRAP_TOP
            c.border = BORDER
        for col in range(6, 12):
            c = rat.cell(row=row, column=col)
            c.fill = INPUT_FILL
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)
            c.border = BORDER
        rat.row_dimensions[row].height = 230

        judge = r.get("judge") or {}
        sys_values = [
            f"H{k + 1:02d}",
            r["id"],
            case["scenario_id"],
            case["category"],
            r["decision"],
            r["confidence"],
            r["groundedness"],
            round(r["reference_step_recall"], 4),
            judge.get("relevance"),
            judge.get("completeness"),
            judge.get("specificity"),
            judge.get("correctness"),
            "Yes" if r["n_steps"] == 0 else "No",
        ]
        for col, v in enumerate(sys_values, start=1):
            c = sysw.cell(row=row, column=col, value=v)
            c.font = Font(name=FONT, size=9)
            c.border = BORDER
        sysw.cell(row=row, column=6).number_format = "0.000"
        sysw.cell(row=row, column=7).number_format = "0.000"

    rng_str = lambda col: f"{col}{FIRST}:{col}{LAST}"  # noqa: E731
    dv_score = DataValidation(
        type="list",
        formula1='"1,2,3,4,5"',
        allow_blank=True,
        error="Choose 1-5",
        errorTitle="Invalid rating",
        showErrorMessage=True,
    )
    dv_yes = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True, showErrorMessage=True)
    dv_dec = DataValidation(type="list", formula1='"RESOLVE,REVIEW,ESCALATE"', allow_blank=True, showErrorMessage=True)
    for dv in (dv_score, dv_yes, dv_dec):
        rat.add_data_validation(dv)
    for letter in "FGH":
        dv_score.add(rng_str(letter))
    dv_yes.add(rng_str("I"))
    dv_dec.add(rng_str("J"))
    rat.freeze_panes = "C2"
    sysw.freeze_panes = "B2"
    sysw.sheet_state = "hidden"

    # ------------------------------------------------------------------ Summary (formulas only)
    sm = wb.create_sheet("Summary")
    sm.column_dimensions["A"].width = 62
    sm.column_dimensions["B"].width = 14
    sm.column_dimensions["C"].width = 70
    R = lambda col: f"Ratings!{rng_str(col)}"  # noqa: E731
    S = lambda col: f"System!{rng_str(col)}"  # noqa: E731
    summary = [
        ("Open after rating is complete.", None, None),
        ("Metric", "Value", "How it is computed"),
        ("Samples rated (correctness filled)", f"=COUNT({R('H')})", "count of numeric correctness ratings"),
        ("Mean relevance (1-5)", f'=IFERROR(AVERAGE({R("F")}),"-")', "average of rated rows"),
        ("Mean completeness (1-5)", f'=IFERROR(AVERAGE({R("G")}),"-")', "average of rated rows"),
        ("Mean correctness (1-5)", f'=IFERROR(AVERAGE({R("H")}),"-")', "average of rated rows"),
        ("Share safe to use as-is", f'=IFERROR(COUNTIF({R("I")},"Yes")/COUNTA({R("I")}),"-")', "Yes / answered"),
        ("Human decisions: RESOLVE", f'=COUNTIF({R("J")},"RESOLVE")', ""),
        ("Human decisions: REVIEW", f'=COUNTIF({R("J")},"REVIEW")', ""),
        ("Human decisions: ESCALATE", f'=COUNTIF({R("J")},"ESCALATE")', ""),
        ("System decisions: RESOLVE", f'=COUNTIF({S("E")},"RESOLVE")', "from the hidden System sheet"),
        ("System decisions: REVIEW", f'=COUNTIF({S("E")},"REVIEW")', ""),
        ("System decisions: ESCALATE", f'=COUNTIF({S("E")},"ESCALATE")', ""),
        (
            "Decision agreement (human = system)",
            f'=IFERROR(SUMPRODUCT(({R("J")}<>"")*({R("J")}={S("E")}))/COUNTA({R("J")}),"-")',
            "share of answered rows where your decision equals the system's",
        ),
        (
            "Share safe, among system RESOLVE",
            f'=IFERROR(SUMPRODUCT(({S("E")}="RESOLVE")*({R("I")}="Yes"))/SUMPRODUCT(({S("E")}="RESOLVE")*({R("I")}<>"")),"-")',
            "precision of RESOLVE: how often an auto-resolved draft was actually safe",
        ),
        (
            "Mean human correctness | system RESOLVE",
            f'=IFERROR(AVERAGEIFS({R("H")},{S("E")},"RESOLVE"),"-")',
            "should be highest",
        ),
        ("Mean human correctness | system REVIEW", f'=IFERROR(AVERAGEIFS({R("H")},{S("E")},"REVIEW"),"-")', ""),
        (
            "Mean human correctness | system ESCALATE",
            f'=IFERROR(AVERAGEIFS({R("H")},{S("E")},"ESCALATE"),"-")',
            "should be lowest",
        ),
        (
            "Mean LLM-judge correctness (same 50 drafts)",
            f'=IFERROR(AVERAGE({S("L")}),"-")',
            "compare with mean human correctness",
        ),
        (
            "Correlation: human vs LLM-judge correctness",
            f'=IFERROR(CORREL({R("H")},{S("L")}),"-")',
            "Pearson over rated rows; validates the LLM judge",
        ),
        (
            "Correlation: human correctness vs heuristic confidence",
            f'=IFERROR(CORREL({R("H")},{S("F")}),"-")',
            "does the confidence score track quality?",
        ),
        (
            "Correlation: human completeness vs auto reference-step recall",
            f'=IFERROR(CORREL({R("G")},{S("H")}),"-")',
            "validates the automatic recall metric",
        ),
    ]
    for i, (a, b, c_) in enumerate(summary, start=1):
        ca = sm.cell(row=i, column=1, value=a)
        cb = sm.cell(row=i, column=2, value=b)
        cc = sm.cell(row=i, column=3, value=c_)
        bold = i <= 2
        for cell in (ca, cb, cc):
            cell.font = Font(name=FONT, size=10, bold=bold, italic=(i == 1))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if i == 2:
            for cell in (ca, cb, cc):
                cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
                cell.fill = HEADER_FILL
        if i >= 3:
            for cell in (ca, cb, cc):
                cell.border = BORDER
    for i in (7, 14, 15):
        sm.cell(row=i, column=2).number_format = "0.0%"
    for i in (4, 5, 6, 16, 17, 18, 19, 20, 21, 22):
        sm.cell(row=i, column=2).number_format = "0.00"

    wb.active = wb.sheetnames.index("Ratings")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT} ({N_SAMPLES} samples; retrieval reproduced the stored run for all {len(chosen)} cases)")
    print(
        "system decisions in sample:",
        {d: sum(r["decision"] == d for r in chosen) for d in ("RESOLVE", "REVIEW", "ESCALATE")},
    )


if __name__ == "__main__":
    main()
