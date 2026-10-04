# MiniLM baseline

Results of every evaluation as measured with the previous embedding model (sentence-transformers/all-MiniLM-L6-v2,
384-d), copied here before the switch to all-mpnet-base-v2 so before/after comparisons can be checked. The
corresponding commit on `main` before the switch is d6206e4.

`human_eval_sheet_minilm_drafts.xlsx`, `ai_eval_sheet_minilm_drafts.xlsx` and `ai_rater_eval.json` are the rating
sheet built from the pre-switch LLM drafts, its AI-rated copy and that copy's analysis (EVALUATION.md §6.4). The
sheets in `data/evaluation/` were rebuilt from the mpnet drafts.
