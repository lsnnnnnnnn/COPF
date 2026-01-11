COPF fixed package (regenerated 2026-01-11T05:11:43)

What was fixed:
- runner.py: MultiCalibrator constructor keyword fixed (cfg=...), and multicalibration now updates from DR buffer each step.
- audit.py: adds update_from_dr(dr) and makes compute_violations join against DR buffer items.
- cross_fit.py: removes real-data filtering to d=1 only; fixes synthetic outcome model training to avoid length mismatches.
- eval.py: export_phase_results now reads hits@10 (fallback to hits10).

How to use:
- Copy these .py files over your project counterparts (same relative paths).
- Run your existing scripts as before.

Notes:
- Real-data semantics still follow your current runner: if d==0 then y=0. If you want a different causal semantics, change runner.
