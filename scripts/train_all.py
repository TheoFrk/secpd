#!/usr/bin/env python3
"""Trainiert alle kanonischen Bundles hintereinander oder parallel.

``--mode combined`` schreibt in jedem Lauf financial + combined mit derselben
``train_run_id`` — damit verschwindet die Kohärenz-Warnung.

Default-Universum: ``zenodo_full`` + full-Panels, falls vorhanden.
Rating-Jobs nur, wenn ``--ratings`` existiert.

Beispiele
---------
python scripts/train_all.py                      # hintereinander, Full-Universum
python scripts/train_all.py --jobs 2             # zwei Läufe parallel
python scripts/train_all.py --dry-run
python scripts/train_all.py --horizons 12,36 --skip-fraud
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
SECRETS_FILE = ROOT / ".secpd.env"

logger = logging.getLogger("train_all")

DEFAULT_HORIZONS = (12, 24, 36)


def _load_secrets() -> None:
    if not SECRETS_FILE.exists():
        return
    for raw in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _prefer(full: Path, small: Path) -> Path:
    return full if full.exists() else small


def default_paths() -> dict[str, Path]:
    return {
        "data": _prefer(
            ROOT / "data/processed/zenodo_full.csv.gz",
            ROOT / "data/processed/zenodo_labeled.csv.gz",
        ),
        "financials": _prefer(
            ROOT / "data/raw/financials_panel_full.csv",
            ROOT / "data/raw/financials_panel.csv",
        ),
        "events": _prefer(
            ROOT / "data/raw/edgar_8k_events_full.csv",
            ROOT / "data/raw/edgar_8k_events.csv",
        ),
        "ratings": ROOT / "data/raw/ratings_panel.csv",
        "out": ROOT / "models",
    }


def plan_jobs(
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    include_fraud: bool = True,
    include_rating: bool = False,
    rating_targets: tuple[str, ...] = ("ordinal",),
) -> list[dict[str, Any]]:
    """Liste der Sweep-Jobs (noch ohne CLI-Argv)."""
    jobs: list[dict[str, Any]] = []
    for h in horizons:
        jobs.append(
            {
                "name": f"default_h{int(h)}",
                "label_source": "default",
                "horizon": int(h),
            }
        )
    if include_fraud:
        jobs.append({"name": "fraud", "label_source": "fraud"})
    if include_rating:
        for tgt in rating_targets:
            job: dict[str, Any] = {
                "name": f"rating_{tgt}",
                "label_source": "rating",
                "rating_target": tgt,
            }
            if tgt == "downgrade":
                job["horizon"] = 12
                job["name"] = "rating_downgrade_h12"
            jobs.append(job)
    return jobs


def argv_for(
    job: dict[str, Any],
    *,
    data: Path,
    financials: Path | None,
    events: Path | None,
    ratings: Path | None,
    out: Path,
    mode: str = "combined",
    llm: str = "openai",
    llm_cache_only: bool = True,
    llm_refresh: bool = False,
    calibrate: bool = True,
    calibrate_method: str = "auto",
    require_financials: bool = True,
    min_fyear: int = 2009,
    extra: list[str] | None = None,
) -> list[str]:
    argv = [
        PY, str(ROOT / "train.py"),
        "--data", str(data),
        "--mode", mode,
        "--label-source", str(job["label_source"]),
        "--out", str(out),
        "--min-fyear", str(int(min_fyear)),
    ]
    if financials is not None:
        argv += ["--financials", str(financials)]
    if events is not None:
        argv += ["--events", str(events)]
    if require_financials:
        argv.append("--require-financials")
    if job["label_source"] == "default":
        argv += ["--default-horizon", str(int(job["horizon"]))]
    if job["label_source"] == "rating":
        if ratings is None:
            raise ValueError("Rating-Job braucht --ratings")
        argv += ["--ratings", str(ratings), "--rating-target", str(job["rating_target"])]
        if job.get("horizon") is not None:
            argv += ["--default-horizon", str(int(job["horizon"]))]
    if mode in {"combined", "ensemble"}:
        argv += ["--llm", llm]
        if llm_cache_only:
            argv.append("--llm-cache-only")
        if llm_refresh:
            argv.append("--llm-refresh")
    skip_calibrate = job.get("label_source") == "rating" and str(job.get("rating_target")) == "ordinal"
    if calibrate and not skip_calibrate:
        argv.append("--calibrate")
        argv += ["--calibrate-method", str(calibrate_method)]
    if extra:
        argv.extend(extra)
    return argv


def parse_horizons(raw: str) -> tuple[int, ...]:
    parts = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    if not parts or any(h < 1 for h in parts):
        raise argparse.ArgumentTypeError("Horizonte: positive Integers, kommagetrennt")
    return parts


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    p = argparse.ArgumentParser(description="Alle SEC-PD-Bundles trainieren (Sweep)")
    p.add_argument("--data", default=str(paths["data"]))
    p.add_argument("--financials", default=str(paths["financials"]))
    p.add_argument("--events", default=str(paths["events"]))
    p.add_argument("--ratings", default=str(paths["ratings"]))
    p.add_argument("--out", default=str(paths["out"]))
    p.add_argument("--mode", choices=["financial", "combined", "ensemble"], default="combined")
    p.add_argument("--horizons", type=parse_horizons, default=DEFAULT_HORIZONS,
                   help="Default-Label-Horizonte, kommagetrennt (Default: 12,24,36)")
    p.add_argument("--skip-fraud", action="store_true")
    p.add_argument("--skip-rating", action="store_true")
    p.add_argument("--jobs", type=int, default=1,
                   help="1 = hintereinander, >1 = parallel (Subprozesse)")
    p.add_argument("--llm", default=os.environ.get("SECPD_LLM_MODE") or "openai")
    p.add_argument("--llm-cache-only", action="store_true", default=True)
    p.add_argument("--no-llm-cache-only", action="store_true")
    p.add_argument("--llm-refresh", action="store_true")
    p.add_argument("--calibrate", action="store_true", default=True)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument(
        "--calibrate-method",
        choices=["auto", "sigmoid", "isotonic"],
        default="auto",
        help="an train.py durchreichen (auto: sigmoid bei <100 Positiven)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run_one(name: str, argv: list[str]) -> tuple[str, int]:
    logger.info("START %s", name)
    proc = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), check=False)
    rc = int(proc.returncode)
    if rc == 0:
        logger.info("OK    %s", name)
    else:
        logger.error("FAIL  %s (exit %d)", name, rc)
    return name, rc


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _load_secrets()
    args = parse_args()
    llm_cache_only = bool(args.llm_cache_only) and not args.no_llm_cache_only
    calibrate = bool(args.calibrate) and not args.no_calibrate

    data = Path(args.data)
    financials = Path(args.financials) if args.financials else None
    events = Path(args.events) if args.events else None
    ratings_path = Path(args.ratings) if args.ratings else None
    if financials is not None and not financials.exists():
        financials = None
    if events is not None and not events.exists():
        events = None
    have_ratings = ratings_path is not None and ratings_path.exists()
    include_rating = have_ratings and not args.skip_rating
    if not args.skip_rating and not have_ratings:
        logger.info("Kein Ratings-Panel (%s) — Rating-Jobs übersprungen.", args.ratings)

    jobs = plan_jobs(
        horizons=tuple(args.horizons),
        include_fraud=not args.skip_fraud,
        include_rating=include_rating,
    )
    planned = []
    for job in jobs:
        argv = argv_for(
            job,
            data=data,
            financials=financials,
            events=events,
            ratings=ratings_path if include_rating else None,
            out=Path(args.out),
            mode=args.mode,
            llm=args.llm,
            llm_cache_only=llm_cache_only,
            llm_refresh=bool(args.llm_refresh),
            calibrate=calibrate,
            calibrate_method=str(args.calibrate_method),
        )
        planned.append((job["name"], argv))

    logger.info(
        "%d Jobs · universe=%s · mode=%s · jobs=%s",
        len(planned), data.name, args.mode,
        "sequential" if args.jobs <= 1 else f"parallel:{args.jobs}",
    )
    for name, argv in planned:
        logger.info("  %s: %s", name, " ".join(argv[1:]))

    if args.dry_run or not planned:
        return 0

    results: list[tuple[str, int]] = []
    n_workers = max(1, int(args.jobs))
    if n_workers == 1:
        for name, argv in planned:
            results.append(_run_one(name, argv))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_run_one, name, argv): name for name, argv in planned}
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [n for n, rc in results if rc != 0]
    if failed:
        logger.error("Fehlgeschlagen: %s", ", ".join(failed))
        return 1
    logger.info("Alle %d Jobs ok.", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
