---
name: tidy-notebook
description: Review an analysis notebook (especially report_*.ipynb) against the notebook rules in CLAUDE.md and trim it before it is shared. Use when a report notebook is finished or when I ask to clean up a notebook.
---

1. Read the notebook and the notebook rules in CLAUDE.md.
2. List the proposed changes, one line each: redundant cells or plots,
   markdown that is too long or uses internal jargon, missing or outdated
   summary at the top, leftover debug output, obsolete cells.
3. Wait for my approval, then apply the changes.
4. Run the notebook top to bottom and confirm it executes cleanly.
