# Reports

The written report, the two walkthrough decks and the Loom talking points. The
documents are rendered with Quarto and LaTeX (xelatex) and run no Python. Numbers
are written by hand from the files in `../artifacts/` (`../artifacts/RESULTS.md`
traces each one to its source). Figures are native TikZ/pgfplots or PNGs written
by the pipeline.

| Source | Output | What it is |
|--------|--------|------------|
| `report.qmd` | `_output/report.pdf` | PDF report (KOMA `scrartcl`, A4, 8 pages): validation and split approach, model, results (incl. the TabPack check), December chart |
| `deck.qmd` | `_output/deck.pdf` | short Beamer deck, 16:9, 9 slides (one figure each, no section dividers) for the 3-minute video |
| `deck-long.qmd` | `_output/deck-long.pdf` | long Beamer deck, 16:9, 22 pages (17 frames + section dividers): the full walkthrough, kept for interviews |

Shared figures in `figures/` (each `\input` by one or more documents; every optional
`\def` switch defaults to the look the report uses):

| File | What | Data |
|------|------|------|
| `validation-timeline.tex` | the forward-time folds (`\def\tlKeyGap{4mm}` in both decks keeps the legend inside 16:9) | – |
| `month-bias.tex` | median bias, first vs second test month (`\def\mbShowFinal{0}` draws the plain panel only; `\mbBarW` sets the bar width) | numbers inline, from `backtest/cb_base.json` and `tuning/final_comparison.csv` |
| `f3-mae.tex` | F3 holdout MAE by model (`\def\fmaeShowTabPack{1}` adds the two TabPack rows, short deck) | numbers inline, from `tuning/final_comparison.csv`, `backtest/*summary.csv`, `final/ensemble_f3.json`; TabPack rows from `../artifacts/tabpack/table_a.csv` |
| `decomposition.tex` | block diagram of L(date) + f(x) | – |
| `december-neutral.tex` | December chart, Option A vs neutral market | reads `../artifacts/final/december_components.csv` at render time |
| `tabpack-f3.tex` | F3 clean and day-debiased MAE, CatBoost vs TabPack (long deck) | numbers inline, from `../artifacts/tabpack/table_a.csv` |

PNG figures come from `../scorer_results/candidate_december.png` (all three),
`../artifacts/final/level_continuity.png` (report),
`../artifacts/backtest/level_index.png` (both decks) and
`../artifacts/backtest/outlier_log_ratio_hist.png` (short deck).

## Render

Requires Quarto (tested with 1.7.30) and a TeX distribution with `xelatex`
(tested with TeX Live 2023). Render one document at a time:

```bash
cd reports
quarto render report.qmd   # -> _output/report.pdf
quarto render deck.qmd     # -> _output/deck.pdf
quarto render deck-long.qmd   # -> _output/deck-long.pdf
```

All three documents are in the `render` list of `_quarto.yml`, so a bare
`quarto render` builds them into `_output/`. `make deck` renders the short deck only.

To debug LaTeX, add `-M keep-tex:true`; the `.tex` file is then left next to the
`.qmd` (`deck.tex` and `report.tex` are git-ignored; delete `deck-long.tex` by hand).

## Inputs the documents need

All three documents include `../scorer_results/candidate_december.png` if it exists
(LaTeX `\IfFileExists`) and show a dashed placeholder box otherwise; the other
PNGs and `december_components.csv` must exist. `make all` writes all of them.
The `\tbd{...}` / `\tbdmark` placeholder macros are still defined in
`styles/article-pkms.tex`, but no document uses them any more.

## Styles

`styles/` holds the PKMS house style: the palette, TikZ/pgfplots setup and the
custom Beamer theme.

| File | Origin | Role |
|------|--------|------|
| `common.tex` | copied from the PKMS vault (`~/Documents/PKMS/reports/styles/`) | palette + TikZ, pgfplots, booktabs, listings |
| `header-pkms.tex` | copied from the PKMS vault | custom Beamer theme (deck only) |
| `report.mplstyle` | copied from the PKMS vault | matplotlib style matching the palette |
| `pkms_palette.py` | copied from the PKMS vault | palette constants + `use_style()` |
| `article-pkms.tex` | this project | tints the report's KOMA title and headings to match the deck |

The four copied files are verbatim apart from a one-line provenance comment at
the top; edit the originals in the vault, not these copies. Palette:
`pkmsink #1A2A3A`, `accentblue #1565C0`, `accentgreen #2E7D32`,
`accentred #C62828`, `accentamber #F9A825`, `accentgray #616161`,
`surface #F5F5F5`.
