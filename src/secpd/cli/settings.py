"""Einstellungen: Status, LLM, Fetch, Training."""
from __future__ import annotations

import os
import subprocess

from secpd.cli import state
from secpd.cli.debug import configure_logging, format_debug_status
from secpd.cli.catalog import active_model_meta, invalidate_model_cache, list_model_catalog, warn_model_coherence
from secpd.cli.paths import (
    AAER,
    EVENTS,
    EVENTS_FULL,
    FIRM_YEARS,
    FIRM_YEARS_LABELS,
    LABELED,
    LABELED_FULL,
    MODEL_DIR,
    NATIVE_DEFAULT_HORIZONS,
    PANEL,
    PANEL_FULL,
    PY,
    RATINGS,
    ROOT,
    SECRETS_FILE,
    SUBMISSIONS_CACHE,
)
from secpd.cli.ui import (
    C,
    ask,
    ask_forecast_horizon,
    banner,
    clear,
    file_status,
    horizon_label,
    hr,
    pause,
    scale_pd,
)
from secpd.data.events import MIN_FYEAR_WITH_FINANCIALS
from secpd.llm.bank import DEFAULT_OPENAI_ENDPOINT, DEFAULT_OPENAI_MODEL


# --------------------------------------------------------------------------- #
# Secrets, Skript-Starter, SEC-UA
# --------------------------------------------------------------------------- #


def load_secrets_env() -> None:
    """Lädt lokale Secrets aus ``.secpd.env`` in os.environ (ohne Überschreiben)."""
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


def save_secrets_env(updates: dict[str, str]) -> None:
    """Schreibt/merged Key-Value in ``.secpd.env`` (gitignored)."""
    existing: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for raw in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip()
    existing.update({k: str(v) for k, v in updates.items() if v})
    lines = ["# Lokale SEC-PD Secrets — nicht committen"]
    lines.extend(f"{k}={v}" for k, v in sorted(existing.items()))
    lines.append("")
    SECRETS_FILE.write_text("\n".join(lines), encoding="utf-8")
    try:
        SECRETS_FILE.chmod(0o600)
    except OSError:
        pass


def run_script(argv: list[str], *, title: str) -> int:
    """Führt ein Skript/CLI im Projektroot aus und streamt die Ausgabe."""
    hr()
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"  {C.DIM}$ {' '.join(argv)}{C.RESET}")
    hr()
    print()
    try:
        proc = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), check=False)
    except KeyboardInterrupt:
        print(f"\n  {C.YELLOW}Abgebrochen.{C.RESET}")
        return 130
    code = int(proc.returncode)
    print()
    if code == 0:
        print(f"  {C.GREEN}Fertig (Exit {code}).{C.RESET}")
    else:
        print(f"  {C.RED}Fehlgeschlagen (Exit {code}).{C.RESET}")
    return code


def ensure_sec_ua() -> str | None:
    ua = os.environ.get("SECPD_SEC_UA", "").strip()
    if ua:
        return ua
    print(f"  {C.YELLOW}SECPD_SEC_UA ist nicht gesetzt (SEC-Pflicht).{C.RESET}")
    ua = ask(
        "User-Agent (Firma email@…)",
        "Commerzbank Praktikum vorname.nachname@example.com",
    ).strip()
    if not ua:
        print(f"  {C.RED}Ohne User-Agent kein EDGAR-Abruf.{C.RESET}")
        return None
    os.environ["SECPD_SEC_UA"] = ua
    save_secrets_env({"SECPD_SEC_UA": ua})
    return ua


# --------------------------------------------------------------------------- #
# Status / LLM / Fetch
# --------------------------------------------------------------------------- #


def show_settings_status() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Daten- & Modell-Status{C.RESET}")
    hr()
    rows = [
        ("AAER", AAER),
        ("Firm-Years Labels", FIRM_YEARS_LABELS),
        ("Firm-Years (groß)", FIRM_YEARS),
        ("Labeled Dataset", LABELED),
        ("Full Dataset", LABELED_FULL),
        ("Finanz-Panel", PANEL),
        ("Finanz-Panel full", PANEL_FULL),
        ("8-K Events", EVENTS),
        ("8-K Events full", EVENTS_FULL),
        ("Ratings-Panel", RATINGS),
        ("Submissions-Cache", SUBMISSIONS_CACHE),
        ("Models-Dir", MODEL_DIR),
    ]
    for label, path in rows:
        print(f"  {label:<20} {file_status(path)}")
        print(f"  {C.DIM}{'':20} {path.relative_to(ROOT)}{C.RESET}")

    print()
    print(f"  {C.BOLD}Modell-Bundles{C.RESET}")
    catalog = list_model_catalog()
    if not catalog:
        print(f"  {C.YELLOW}keine .joblib unter models/{C.RESET}")
    else:
        warn_model_coherence(catalog)
        for row in catalog:
            if row.get("legacy"):
                tag = "LEGACY"
            elif row.get("experimental"):
                tag = "experimentell"
            else:
                tag = "ok"
            print(
                f"  · {row['name']:<32} {C.DIM}{tag}  "
                f"{row.get('component')}/{row.get('label_source')}"
                f"{' h'+str(row['horizon_months']) if row.get('horizon_months') else ''}  "
                f"trained={row.get('trained_at') or '—'}{C.RESET}"
            )
    print()
    mode = os.environ.get("SECPD_LLM_MODE", "mock")
    endpoint = os.environ.get("SECPD_LLM_ENDPOINT", "") or "—"
    model = os.environ.get("SECPD_LLM_MODEL", "internal-default")
    ua = os.environ.get("SECPD_SEC_UA", "") or "—"
    key_set = "gesetzt" if os.environ.get("SECPD_LLM_API_KEY") else "nicht gesetzt"
    fmp_set = "gesetzt" if os.environ.get("SECPD_FMP_API_KEY") else "nicht gesetzt"
    print(f"  {C.BOLD}Umgebung{C.RESET}")
    print(f"    LLM-Modus     {C.CYAN}{mode}{C.RESET}")
    print(f"    LLM-Endpoint  {endpoint}")
    print(f"    LLM-Modell    {model}")
    print(f"    LLM-API-Key   {key_set}")
    print(f"    FMP-API-Key   {fmp_set}")
    print(f"    SEC-UA        {ua}")
    print()
    print(f"  {C.BOLD}Debug{C.RESET}")
    print(f"    {format_debug_status()}")
    pause()


def settings_llm() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}LLM anbinden{C.RESET}")
    hr()
    key_set = (
        "gesetzt"
        if (os.environ.get("SECPD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        else "nicht gesetzt"
    )
    print(f"  Aktuell: Modus={C.CYAN}{os.environ.get('SECPD_LLM_MODE', 'mock')}{C.RESET}")
    print(f"           Endpoint={os.environ.get('SECPD_LLM_ENDPOINT', '') or '—'}")
    print(f"           Modell={os.environ.get('SECPD_LLM_MODEL', 'internal-default')}")
    print(f"           API-Key={key_set}")
    print(f"  {C.DIM}Secrets: {SECRETS_FILE.name} (lokal, nicht committen){C.RESET}")
    print()
    print(f"  {C.CYAN}1{C.RESET}  Mock (offline-Heuristik)")
    print(f"  {C.CYAN}2{C.RESET}  OpenAI / ChatGPT — schnell (Default: {DEFAULT_OPENAI_MODEL})")
    print(f"  {C.CYAN}3{C.RESET}  LM Studio (lokal) — 172.16.3.164:1234")
    print(f"  {C.CYAN}4{C.RESET}  Bank-Gateway (OpenAI-kompatibel)")
    print(f"  {C.CYAN}5{C.RESET}  Ping / Test-Call")
    print(f"  {C.CYAN}0{C.RESET}  Zurück")
    print()
    choice = ask("Auswahl", "0")
    if choice == "1":
        from secpd.config import load_settings

        if not load_settings().llm_allow_mock:
            print(
                f"  {C.RED}Mock ist in Debug-Einstellungen verboten "
                f"(SECPD_LLM_ALLOW_MOCK=0).{C.RESET}"
            )
            pause()
            return
        os.environ["SECPD_LLM_MODE"] = "mock"
        save_secrets_env({"SECPD_LLM_MODE": "mock"})
        print(f"  {C.GREEN}LLM-Modus = mock{C.RESET}")
        pause()
        return
    if choice == "5":
        mode = os.environ.get("SECPD_LLM_MODE", "mock")
        if mode in {"openai", "chatgpt"}:
            ep = os.environ.get("SECPD_LLM_ENDPOINT") or DEFAULT_OPENAI_ENDPOINT
            model = os.environ.get("SECPD_LLM_MODEL") or DEFAULT_OPENAI_MODEL
        else:
            ep = os.environ.get("SECPD_LLM_ENDPOINT") or "http://172.16.3.164:1234"
            model = os.environ.get("SECPD_LLM_MODEL") or "auto"
        argv = [
            PY, str(ROOT / "scripts" / "ping_llm.py"),
            "--endpoint", ep, "--model", model, "--analyze",
            "--timeout", os.environ.get("SECPD_LLM_TIMEOUT", "120"),
        ]
        code = run_script(argv, title="ping_llm.py")
        print(f"  {C.GREEN if code == 0 else C.YELLOW}"
              f"{'OK' if code == 0 else 'Fehlgeschlagen'}{C.RESET}")
        pause()
        return
    if choice == "2":
        os.environ["SECPD_LLM_MODE"] = "openai"
        os.environ["SECPD_LLM_ENDPOINT"] = DEFAULT_OPENAI_ENDPOINT
        cur_model = os.environ.get("SECPD_LLM_MODEL", "")
        default_model = cur_model if cur_model.startswith("gpt-") else DEFAULT_OPENAI_MODEL
        model = ask("OpenAI-Modell", default_model)
        os.environ["SECPD_LLM_MODEL"] = model or DEFAULT_OPENAI_MODEL
        os.environ["SECPD_LLM_JSON_MODE"] = "1"
        os.environ["SECPD_LLM_TIMEOUT"] = os.environ.get("SECPD_LLM_TIMEOUT") or "120"
        key = ask("OpenAI API-Key (sk-…, leer = behalten)", "")
        if key:
            os.environ["SECPD_LLM_API_KEY"] = key
        if not (os.environ.get("SECPD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print(f"  {C.RED}Kein API-Key — bitte eingeben.{C.RESET}")
            pause()
            return
        save_secrets_env(
            {
                "SECPD_LLM_MODE": "openai",
                "SECPD_LLM_ENDPOINT": DEFAULT_OPENAI_ENDPOINT,
                "SECPD_LLM_MODEL": os.environ["SECPD_LLM_MODEL"],
                "SECPD_LLM_API_KEY": os.environ.get("SECPD_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY", ""),
                "SECPD_LLM_JSON_MODE": "1",
                "SECPD_LLM_TIMEOUT": os.environ["SECPD_LLM_TIMEOUT"],
            }
        )
        print(f"  {C.GREEN}OpenAI gespeichert ({os.environ['SECPD_LLM_MODEL']}).{C.RESET}")
        pause()
        return
    if choice == "3":
        os.environ["SECPD_LLM_MODE"] = "lmstudio"
        endpoint = ask(
            "LM-Studio-Host/URL",
            "http://172.16.3.164:1234",
        )
        os.environ["SECPD_LLM_ENDPOINT"] = endpoint
        model = ask("Modell (auto = erstes Chat-Modell)", "auto")
        os.environ["SECPD_LLM_MODEL"] = model or "auto"
        os.environ["SECPD_LLM_API_KEY"] = os.environ.get("SECPD_LLM_API_KEY") or "lm-studio"
        os.environ["SECPD_LLM_JSON_MODE"] = "0"
        save_secrets_env(
            {
                "SECPD_LLM_MODE": "lmstudio",
                "SECPD_LLM_ENDPOINT": os.environ["SECPD_LLM_ENDPOINT"],
                "SECPD_LLM_MODEL": os.environ["SECPD_LLM_MODEL"],
                "SECPD_LLM_API_KEY": os.environ["SECPD_LLM_API_KEY"],
                "SECPD_LLM_JSON_MODE": "0",
            }
        )
        print(f"  {C.GREEN}LM Studio gesetzt.{C.RESET}")
        pause()
        return
    if choice != "4":
        return

    os.environ["SECPD_LLM_MODE"] = "bank"
    endpoint = ask("Endpoint-URL", os.environ.get("SECPD_LLM_ENDPOINT") or "")
    if endpoint:
        os.environ["SECPD_LLM_ENDPOINT"] = endpoint
    model = ask("Modellname", os.environ.get("SECPD_LLM_MODEL") or "internal-default")
    if model:
        os.environ["SECPD_LLM_MODEL"] = model
    key = ask("API-Key (leer = behalten)", "")
    if key:
        os.environ["SECPD_LLM_API_KEY"] = key
    payload = {
        "SECPD_LLM_MODE": "bank",
        "SECPD_LLM_ENDPOINT": os.environ.get("SECPD_LLM_ENDPOINT", ""),
        "SECPD_LLM_MODEL": os.environ.get("SECPD_LLM_MODEL", ""),
    }
    if os.environ.get("SECPD_LLM_API_KEY"):
        payload["SECPD_LLM_API_KEY"] = os.environ["SECPD_LLM_API_KEY"]
    save_secrets_env(payload)
    print(f"  {C.GREEN}Bank-Gateway gesetzt.{C.RESET}")
    pause()


def settings_sec_ua() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}SEC User-Agent{C.RESET}")
    hr()
    current = os.environ.get("SECPD_SEC_UA", "") or "—"
    print(f"  Aktuell: {current}")
    print(f"  {C.DIM}Format: „Firma name@firma.de“ (SEC-Pflicht für EDGAR).{C.RESET}")
    print()
    ua = ask("Neuer User-Agent (leer = abbrechen)", "")
    if not ua:
        return
    os.environ["SECPD_SEC_UA"] = ua
    save_secrets_env({"SECPD_SEC_UA": ua})
    print(f"  {C.GREEN}SECPD_SEC_UA gesetzt.{C.RESET}")
    pause()


def settings_convert_zenodo() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Convert Zenodo{C.RESET}")
    hr()
    fy = FIRM_YEARS_LABELS if FIRM_YEARS_LABELS.exists() else FIRM_YEARS
    print(f"  Firm-Years: {file_status(fy)}")
    print(f"  AAER:       {file_status(AAER)}")
    print(f"  Ziel:       {LABELED.relative_to(ROOT)}")
    print()
    if not fy.exists() or not AAER.exists():
        print(f"  {C.RED}Rohdaten fehlen — zuerst Zenodo laden (Menü 4 → Fetch Zenodo).{C.RESET}")
        pause()
        return
    if LABELED.exists() and not ask("Vorhandenes labeled überschreiben? (j/n)", "j").lower().startswith("j"):
        return
    run_script(
        [
            PY, str(ROOT / "scripts" / "convert_zenodo.py"),
            "--firm-years", str(fy),
            "--aaer", str(AAER),
            "--out", str(LABELED),
        ],
        title="convert_zenodo.py",
    )
    pause()


def settings_fetch_zenodo() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch Zenodo{C.RESET}")
    hr()
    print(f"  Lädt aaer_mark5.csv + firm_years_labels.json (~683 MB).")
    print()
    if not ask("Jetzt herunterladen? (j/n)", "j").lower().startswith("j"):
        return
    run_script(
        [
            PY, str(ROOT / "scripts" / "fetch_zenodo.py"),
            "--files", "aaer_mark5.csv", "firm_years_labels.json",
            "--dest", str(ROOT / "data" / "raw"),
        ],
        title="fetch_zenodo.py",
    )
    pause()


def settings_fetch_edgar_financials() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch EDGAR Financials{C.RESET}")
    hr()
    print(f"  Dataset: {file_status(LABELED)}")
    print(f"  Ziel:    {PANEL.relative_to(ROOT)}")
    print(f"  {C.DIM}~5 Min für ~500 CIKs, Internet nötig.{C.RESET}")
    print()
    if not LABELED.exists():
        print(f"  {C.RED}Labeled Dataset fehlt — zuerst Convert Zenodo.{C.RESET}")
        pause()
        return
    if ensure_sec_ua() is None:
        pause()
        return
    if not ask("Abruf starten? (j/n)", "j").lower().startswith("j"):
        return
    run_script(
        [
            PY, str(ROOT / "scripts" / "fetch_edgar_financials.py"),
            "--dataset", str(LABELED),
            "--out", str(PANEL),
        ],
        title="fetch_edgar_financials.py",
    )
    pause()


def settings_fetch_edgar_events() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch EDGAR 8-K Events{C.RESET}")
    hr()
    print(f"  Dataset: {file_status(LABELED)}")
    print(f"  Cache:   {file_status(SUBMISSIONS_CACHE)}")
    print(f"  Ziel:    {EVENTS.relative_to(ROOT)}")
    print(f"  {C.DIM}~10–60 Min, resumefähig über den Cache.{C.RESET}")
    print()
    if not LABELED.exists():
        print(f"  {C.RED}Labeled Dataset fehlt — zuerst Convert Zenodo.{C.RESET}")
        pause()
        return
    if ensure_sec_ua() is None:
        pause()
        return
    if not ask("Abruf starten? (j/n)", "j").lower().startswith("j"):
        return
    run_script(
        [
            PY, str(ROOT / "scripts" / "fetch_edgar_events.py"),
            "--dataset", str(LABELED),
            "--out", str(EVENTS),
            "--cache-dir", str(SUBMISSIONS_CACHE),
        ],
        title="fetch_edgar_events.py",
    )
    pause()


def settings_fetch_ratings() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Fetch Ratings (NRSRO / FMP){C.RESET}")
    hr()
    print(f"  Dataset: {file_status(LABELED)}")
    print(f"  Ziel:    {RATINGS.relative_to(ROOT)}")
    print(
        f"  {C.DIM}Default: ratingshistory.info (Moody's, Fitch, Egan-Jones) — "
        f"echte Agency-Historien, kein API-Limit.{C.RESET}"
    )
    print()
    print(f"  {C.CYAN}1{C.RESET}  NRSRO (17g-7 Bulk, empfohlen)")
    print(f"  {C.CYAN}2{C.RESET}  FMP-Fundamentalnote (max. 250 Calls/Tag)")
    print(f"  {C.CYAN}3{C.RESET}  beide")
    print(f"  {C.CYAN}0{C.RESET}  Zurück")
    pick = ask("Quelle", "1")
    if pick in {"0", "q"}:
        return
    src = {"2": "fmp", "3": "both"}.get(pick, "nrsro")
    argv = [
        PY, str(ROOT / "scripts" / "fetch_ratings.py"),
        "--source", src,
        "--out", str(RATINGS),
    ]
    if LABELED.exists():
        argv += ["--dataset", str(LABELED)]
    if src in {"fmp", "both"}:
        key = os.environ.get("SECPD_FMP_API_KEY", "")
        if not key:
            print(f"  {C.YELLOW}Kein SECPD_FMP_API_KEY.{C.RESET}")
            print("  Kostenlos: https://site.financialmodelingprep.com/register")
            key = ask("FMP API-Key (leer = abbrechen FMP)", "")
            if key:
                os.environ["SECPD_FMP_API_KEY"] = key
                save_secrets_env({"SECPD_FMP_API_KEY": key})
                print(f"  {C.GREEN}FMP-Key in .secpd.env gespeichert.{C.RESET}")
            elif src == "fmp":
                pause()
                return
            else:
                argv[argv.index("--source") + 1] = "nrsro"
        argv += ["--max-requests", "250"]
    run_script(argv, title="fetch_ratings.py")
    pause()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def settings_train_all() -> None:
    """Sweep: combined schreibt financial+combined je Label mit derselben run-id."""
    clear()
    banner()
    print(f"  {C.BOLD}Alle Modelle trainieren (Sweep){C.RESET}")
    hr()
    data = LABELED_FULL if LABELED_FULL.exists() else LABELED
    panel = PANEL_FULL if PANEL_FULL.exists() else PANEL
    events = EVENTS_FULL if EVENTS_FULL.exists() else EVENTS
    print(f"  Data:       {file_status(data)}  {C.DIM}{data.relative_to(ROOT)}{C.RESET}")
    print(f"  Financials: {file_status(panel)}")
    print(f"  Events:     {file_status(events)}")
    print(f"  Ratings:    {file_status(RATINGS)}")
    print()
    print(f"  {C.DIM}combined schreibt financial + combined im selben Lauf.{C.RESET}")
    print(f"  {C.DIM}Default: h12, h24, h36 + fraud; Rating nur mit Panel.{C.RESET}")
    print()
    if not data.exists() or not panel.exists():
        print(f"  {C.RED}Dataset oder Finanz-Panel fehlt.{C.RESET}")
        pause()
        return

    print(f"  {C.BOLD}Ausführung{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}  hintereinander  (schonend, Default)")
    print(f"  {C.CYAN}2{C.RESET}  parallel (2 Jobs, mehr RAM)")
    jobs = "2" if ask("Auswahl", "1") == "2" else "1"

    llm = os.environ.get("SECPD_LLM_MODE") or "openai"
    argv = [
        PY, str(ROOT / "scripts" / "train_all.py"),
        "--data", str(data),
        "--financials", str(panel),
        "--events", str(events),
        "--jobs", jobs,
        "--mode", "combined",
        "--llm", llm,
        "--llm-cache-only",
        "--calibrate",
        "--calibrate-method", "auto",
    ]
    if RATINGS.exists() and ask("Rating-Jobs mitnehmen? (j/n)", "n").lower().startswith("j"):
        argv += ["--ratings", str(RATINGS)]
    else:
        argv.append("--skip-rating")
    if ask("Fraud-Label mittrainieren (experimentell)? (j/n)", "n").lower().startswith("j"):
        pass
    else:
        argv.append("--skip-fraud")

    print()
    if not ask("Sweep starten? (j/n)", "j").lower().startswith("j"):
        return
    run_script(argv, title="train_all.py")
    invalidate_model_cache()
    active_model_meta(refresh=True)
    pause()


def settings_train() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Modell trainieren{C.RESET}")
    hr()
    print(f"  Data:       {file_status(LABELED)}")
    print(f"  Financials: {file_status(PANEL)}")
    print(f"  Events:     {file_status(EVENTS)}")
    print(f"  Ratings:    {file_status(RATINGS)}")
    print()
    if not LABELED.exists():
        print(f"  {C.RED}Labeled Dataset fehlt.{C.RESET}")
        pause()
        return
    if not PANEL.exists():
        print(f"  {C.YELLOW}Finanz-Panel fehlt — Training ohne --financials ist schwach.{C.RESET}")
        if not ask("Trotzdem fortfahren? (j/n)", "n").lower().startswith("j"):
            return

    print(f"  {C.BOLD}Label{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}  fraud   — AAER/Misconduct")
    print(f"  {C.CYAN}2{C.RESET}  default — Ausfall-PD aus 8-K Insolvenz (Horizont wählbar)")
    print(f"  {C.CYAN}3{C.RESET}  rating  — Agency-Note (ordinal / HY / Downgrade)")
    print(f"  {C.CYAN}4{C.RESET}  alle    — Sweep: default h12/h24/h36 + fraud (+ rating)")
    label_choice = ask("Label-Quelle", "2")
    if label_choice == "4":
        settings_train_all()
        return
    label_source = {"2": "default", "3": "rating"}.get(label_choice, "fraud")
    if label_source == "default" and not EVENTS.exists():
        print(f"  {C.RED}default braucht Events — zuerst Fetch EDGAR 8-K.{C.RESET}")
        pause()
        return
    if label_source == "rating" and not RATINGS.exists():
        print(f"  {C.RED}rating braucht das Panel — zuerst Fetch Ratings.{C.RESET}")
        pause()
        return

    print()
    print(f"  {C.BOLD}Modus{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}  financial — nur Kennzahlen (+ Events)")
    print(f"  {C.CYAN}2{C.RESET}  combined  — Finanzen + Text + Events")
    print(f"  {C.CYAN}3{C.RESET}  ensemble  — Logit-Ensemble")
    mode_map = {"1": "financial", "2": "combined", "3": "ensemble"}
    mode = mode_map.get(ask("Modus", "2"), "combined")

    llm = os.environ.get("SECPD_LLM_MODE", "mock")
    llm_refresh = False
    if mode in {"combined", "ensemble"}:
        print()
        print(f"  {C.BOLD}LLM für Textfeatures{C.RESET}")
        print(f"  {C.CYAN}1{C.RESET}  mock")
        print(f"  {C.CYAN}2{C.RESET}  openai / ChatGPT ({DEFAULT_OPENAI_MODEL})")
        print(f"  {C.CYAN}3{C.RESET}  lmstudio")
        print(f"  {C.CYAN}4{C.RESET}  bank")
        pick = ask("Auswahl", "2" if llm in {"openai", "chatgpt"} else "1")
        llm = {"1": "mock", "2": "openai", "3": "lmstudio", "4": "bank"}.get(pick, "openai")
        from secpd.config import load_settings as _ls

        if llm == "mock" and not _ls().llm_allow_mock:
            print(f"  {C.RED}Mock ist in Debug-Einstellungen verboten.{C.RESET}")
            pause()
            return
        if llm == "openai" and not (
            os.environ.get("SECPD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        ):
            print(f"  {C.YELLOW}Kein OpenAI-Key — bitte unter Einstellungen → LLM setzen.{C.RESET}")
            key = ask("OpenAI API-Key jetzt eingeben (oder leer = abbrechen)", "")
            if not key:
                pause()
                return
            os.environ["SECPD_LLM_MODE"] = "openai"
            os.environ["SECPD_LLM_API_KEY"] = key
            os.environ["SECPD_LLM_ENDPOINT"] = DEFAULT_OPENAI_ENDPOINT
            os.environ["SECPD_LLM_MODEL"] = DEFAULT_OPENAI_MODEL
            save_secrets_env(
                {
                    "SECPD_LLM_MODE": "openai",
                    "SECPD_LLM_API_KEY": key,
                    "SECPD_LLM_ENDPOINT": DEFAULT_OPENAI_ENDPOINT,
                    "SECPD_LLM_MODEL": DEFAULT_OPENAI_MODEL,
                }
            )
        print()
        print(f"  {C.BOLD}Textfeatures{C.RESET}")
        print(f"  {C.CYAN}1{C.RESET}  Cache nutzen (schnell, nur Misses → LLM)")
        print(f"  {C.CYAN}2{C.RESET}  Neu bewerten (--llm-refresh, überschreibt Cache)")
        cache_pick = ask("Auswahl", "1")
        llm_refresh = cache_pick == "2"

    tgt = "ordinal"
    if label_source == "rating":
        print()
        print(f"  {C.BOLD}Rating-Target{C.RESET}")
        print(f"  {C.CYAN}1{C.RESET}  ordinal     — Notch 1–21 (Shadow-Rating, empfohlen)")
        print(f"  {C.CYAN}2{C.RESET}  speculative — HY-ähnlich zum Bilanzstichtag")
        print(f"  {C.CYAN}3{C.RESET}  downgrade   — Notch fällt im Horizont")
        tgt = {"2": "speculative", "3": "downgrade"}.get(ask("Target", "1"), "ordinal")

    calibrate = False
    if label_source == "rating" and tgt == "ordinal":
        print(f"  {C.DIM}Ordinal-Regression: Kalibrierung entfällt.{C.RESET}")
    else:
        calibrate = ask("Kalibrieren? (j/n)", "j").lower().startswith("j")

    argv = [
        PY, str(ROOT / "train.py"),
        "--data", str(LABELED),
        "--mode", mode,
        "--label-source", label_source,
        "--out", str(ROOT / "models"),
    ]
    if PANEL.exists():
        argv += ["--financials", str(PANEL)]
    if EVENTS.exists() and ask("Event-Features (--events) nutzen? (j/n)", "j").lower().startswith("j"):
        argv += ["--events", str(EVENTS)]
    if label_source == "default":
        if "--events" not in argv:
            argv += ["--events", str(EVENTS)]
        print()
        print(f"  {C.BOLD}Trainings-Horizont (Label){C.RESET}")
        print(f"  {C.DIM}12=1J, 60=5J, 120=10J — echte Labels, nicht nur Termstruktur.{C.RESET}")
        horizon = ask("Monate", "12")
        try:
            h = max(1, int(horizon))
        except ValueError:
            h = 12
        argv += ["--default-horizon", str(h)]
        argv += ["--min-fyear", str(MIN_FYEAR_WITH_FINANCIALS)]
        if ask(
            f"Nur Zeilen mit Finanzdaten (total_assets)? (j/n)",
            "j",
        ).lower().startswith("j"):
            argv.append("--require-financials")
        print(
            f"  {C.DIM}Policy: Legacy-Regime vor 2004-08-23 ignoriert; "
            f"min-fyear={MIN_FYEAR_WITH_FINANCIALS}.{C.RESET}"
        )
    elif label_source == "rating":
        argv += ["--ratings", str(RATINGS)]
        argv += ["--rating-target", tgt]
        if tgt == "downgrade":
            horizon = ask("Horizont Monate", "12")
            try:
                h = max(1, int(horizon))
            except ValueError:
                h = 12
            argv += ["--default-horizon", str(h)]
        argv += ["--min-fyear", str(MIN_FYEAR_WITH_FINANCIALS)]
        if PANEL.exists() and ask("Nur Zeilen mit Finanzdaten (total_assets)? (j/n)", "j").lower().startswith("j"):
            argv.append("--require-financials")
        print(
            f"  {C.DIM}Agency-Ratings (17g-7); ordinal = Regression ohne Kalibrierung.{C.RESET}"
        )
    if mode in {"combined", "ensemble"}:
        argv += ["--llm", llm]
        if llm_refresh:
            argv.append("--llm-refresh")
    if calibrate:
        argv.append("--calibrate")
        argv += ["--calibrate-method", "auto"]

    print()
    if not ask("Training starten? (j/n)", "j").lower().startswith("j"):
        return
    run_script(argv, title="train.py")
    invalidate_model_cache()
    active_model_meta(refresh=True)
    pause()


def settings_forecast_horizon() -> None:
    clear()
    banner()
    print(f"  {C.BOLD}Standard-Vorausschauhorizont{C.RESET}")
    hr()
    meta = active_model_meta()
    model_h = int(meta.get("default_horizon_months") or 12)
    cur = state.FORECAST_HORIZON_MONTHS or model_h
    print(f"  Session-Vorausschau: {cur} M")
    if meta.get("label_source") == "default":
        print(f"  Modellhorizont:      {model_h} M")
    else:
        print(f"  Native PD-Modelle:   {', '.join(f'{h} M' for h in NATIVE_DEFAULT_HORIZONS)}")
        print(
            f"  {C.DIM}Gilt für die sekundäre Ausfall-PD (Rating bleibt ein Punkt-Score).{C.RESET}"
        )
    print()
    state.FORECAST_HORIZON_MONTHS = ask_forecast_horizon(model_h)
    print(f"  {C.GREEN}Gesetzt: {state.FORECAST_HORIZON_MONTHS} Monate "
          f"({horizon_label(state.FORECAST_HORIZON_MONTHS)}).{C.RESET}")
    pause()


def settings_risk_bands() -> None:
    """Schwellen relativ zur Sample-Basisrate im jeweiligen Horizont."""
    clear()
    banner()
    print(f"  {C.BOLD}Risiko-Schwellen (Default-PD){C.RESET}")
    hr()
    meta = active_model_meta()
    metrics = meta.get("metrics") or {}
    model_br = metrics.get("base_rate")
    print(f"  {C.DIM}Bänder = Multiplikatoren × Basisrate im Horizont.{C.RESET}")
    print(f"  {C.DIM}unter Basisrate  <  mid × Basis{C.RESET}")
    print(f"  {C.DIM}um Basisrate     mid … high × Basis{C.RESET}")
    print(f"  {C.DIM}über Basisrate   ≥ high × Basis{C.RESET}")
    print()
    print(f"  Aktuell mid={state.RISK_BAND_MID_MULT:.2f}  high={state.RISK_BAND_HIGH_MULT:.2f}")
    print(f"  Fallback-Basisrate 12M: {100 * state.DEFAULT_BASE_RATE_12M:.2f} %")
    if model_br is not None:
        print(f"  Modell-Testset-Basisrate: {100 * float(model_br):.2f} % "
              f"(wird bevorzugt, falls vorhanden)")
    print()
    print(f"  {C.CYAN}1{C.RESET}  Schwellen anpassen")
    print(f"  {C.CYAN}2{C.RESET}  Defaults wiederherstellen (0.85 / 2.50)")
    print(f"  {C.CYAN}0{C.RESET}  Zurück")
    choice = ask("Auswahl", "0")
    if choice == "2":
        state.RISK_BAND_MID_MULT, state.RISK_BAND_HIGH_MULT = 0.85, 2.5
        state.DEFAULT_BASE_RATE_12M = 0.012
        print(f"  {C.GREEN}Defaults gesetzt.{C.RESET}")
        pause()
        return
    if choice != "1":
        return

    mid_raw = ask("mid-Multiplikator (um Basisrate ab)", f"{state.RISK_BAND_MID_MULT:.2f}")
    high_raw = ask("high-Multiplikator (über Basisrate ab)", f"{state.RISK_BAND_HIGH_MULT:.2f}")
    br_raw = ask("Fallback-Basisrate 12M (z. B. 0.012)", f"{state.DEFAULT_BASE_RATE_12M:.4f}")
    try:
        mid = float(mid_raw.replace(",", "."))
        high = float(high_raw.replace(",", "."))
        br = float(br_raw.replace(",", "."))
        if not (0 < mid < high) or not (0 < br < 1):
            raise ValueError
    except ValueError:
        print(f"  {C.RED}Ungültig — nichts geändert "
              f"(braucht 0 < mid < high und 0 < Basisrate < 1).{C.RESET}")
        pause()
        return
    state.RISK_BAND_MID_MULT, state.RISK_BAND_HIGH_MULT = mid, high
    state.DEFAULT_BASE_RATE_12M = br
    print()
    print(f"  {C.GREEN}Gesetzt:{C.RESET} mid={mid:.2f}  high={high:.2f}  "
          f"Fallback-BR={100 * br:.2f} %")
    # Beispiel für aktuellen Horizont
    h = state.FORECAST_HORIZON_MONTHS or int(meta.get("default_horizon_months") or 12)
    base_h = scale_pd(br, from_months=12, to_months=h)
    print(f"  {C.DIM}Beispiel {horizon_label(h)}: Basis≈{100 * base_h:.2f}% → "
          f"„um“ ab {100 * mid * base_h:.2f}%, "
          f"„über“ ab {100 * high * base_h:.2f}%{C.RESET}")
    pause()


def _persist_flag(key: str, value: str) -> None:
    os.environ[key] = value
    save_secrets_env({key: value})
    configure_logging()


def settings_debug() -> None:
    """Logs, Mock-Sperre, Cache-only, Bildschirm behalten."""
    from secpd.config import load_settings

    while True:
        s = load_settings()
        clear()
        banner()
        print(f"  {C.BOLD}Debug{C.RESET}")
        hr()
        print(f"  {C.DIM}{format_debug_status()}{C.RESET}")
        print(f"  {C.DIM}Logs laufen auf stderr. Bildschirm nicht löschen, sonst verschwinden sie.{C.RESET}")
        print()
        print(
            f"  {C.CYAN}1{C.RESET}  Logs im Terminal     "
            f"{C.GREEN if s.log_level != 'OFF' else C.DIM}{s.log_level}{C.RESET}"
        )
        print(
            f"  {C.CYAN}2{C.RESET}  Bildschirm behalten  "
            f"{C.GREEN if s.debug_keep_screen else C.DIM}"
            f"{'an' if s.debug_keep_screen else 'aus'}{C.RESET}"
            f"  {C.DIM}(Logs bleiben sichtbar){C.RESET}"
        )
        print(
            f"  {C.CYAN}3{C.RESET}  Mock-LLM             "
            f"{C.GREEN if not s.llm_allow_mock else C.YELLOW}"
            f"{'verboten' if not s.llm_allow_mock else 'erlaubt'}{C.RESET}"
        )
        print(
            f"  {C.CYAN}4{C.RESET}  Scoring Cache-only   "
            f"{C.GREEN if s.llm_cache_only else C.YELLOW}"
            f"{'an' if s.llm_cache_only else 'aus (API bei Miss)'}{C.RESET}"
        )
        print(
            f"  {C.CYAN}5{C.RESET}  Cache-Miss           "
            f"{C.GREEN if s.llm_fail_on_miss else C.DIM}"
            f"{'Abbruch' if s.llm_fail_on_miss else 'Fallback'}{C.RESET}"
        )
        print(f"  {C.CYAN}0{C.RESET}  Zurück")
        print()
        choice = ask("Auswahl", "0")
        if choice == "1":
            print()
            print(f"  {C.CYAN}0{C.RESET}  aus (still)")
            print(f"  {C.CYAN}1{C.RESET}  INFO  (Cache-Hits/Misses, Textanalyse)")
            print(f"  {C.CYAN}2{C.RESET}  DEBUG (ausführlich)")
            lvl = ask("Log-Level", "1" if s.log_level == "OFF" else {"INFO": "1", "DEBUG": "2"}.get(s.log_level, "0"))
            mapping = {"0": "OFF", "1": "INFO", "2": "DEBUG"}
            level = mapping.get(lvl, "INFO")
            os.environ["SECPD_LOG_LEVEL"] = level
            os.environ["SECPD_DEBUG"] = "1" if level != "OFF" else "0"
            payload = {"SECPD_LOG_LEVEL": level, "SECPD_DEBUG": os.environ["SECPD_DEBUG"]}
            if level != "OFF" and os.environ.get("SECPD_DEBUG_KEEP_SCREEN", "").strip() == "":
                os.environ["SECPD_DEBUG_KEEP_SCREEN"] = "1"
                payload["SECPD_DEBUG_KEEP_SCREEN"] = "1"
            save_secrets_env(payload)
            configure_logging()
            print(f"  {C.GREEN}Log-Level = {level}{C.RESET}")
            if level != "OFF":
                print(f"  {C.DIM}Beim nächsten Scoring erscheinen INFO-Zeilen im Terminal.{C.RESET}")
            pause()
        elif choice == "2":
            new = "0" if s.debug_keep_screen else "1"
            _persist_flag("SECPD_DEBUG_KEEP_SCREEN", new)
            print(f"  {C.GREEN}Bildschirm behalten = {'an' if new == '1' else 'aus'}{C.RESET}")
            pause()
        elif choice == "3":
            new = "0" if s.llm_allow_mock else "1"
            _persist_flag("SECPD_LLM_ALLOW_MOCK", new)
            if new == "0":
                mode = os.environ.get("SECPD_LLM_MODE", "mock")
                if mode == "mock":
                    print(
                        f"  {C.YELLOW}Aktueller LLM-Modus ist mock — "
                        f"bitte unter LLM auf openai stellen.{C.RESET}"
                    )
            print(f"  {C.GREEN}Mock-LLM = {'erlaubt' if new == '1' else 'verboten'}{C.RESET}")
            pause()
        elif choice == "4":
            new = "0" if s.llm_cache_only else "1"
            _persist_flag("SECPD_LLM_CACHE_ONLY", new)
            print(
                f"  {C.GREEN}Cache-only = {'an' if new == '1' else 'aus'}{C.RESET}"
            )
            if new == "0":
                print(f"  {C.DIM}Cache-Miss ruft GPT auf (Kosten/Key in .secpd.env).{C.RESET}")
            pause()
        elif choice == "5":
            new = "0" if s.llm_fail_on_miss else "1"
            _persist_flag("SECPD_LLM_FAIL_ON_MISS", new)
            print(
                f"  {C.GREEN}Cache-Miss = {'Abbruch' if new == '1' else 'Fallback'}{C.RESET}"
            )
            pause()
        elif choice in {"0", "q", "b", "back"}:
            return
        else:
            print(f"  {C.YELLOW}Bitte 0–5 wählen.{C.RESET}")
            pause()


def settings_menu() -> None:
    while True:
        clear()
        banner()
        print(f"  {C.BOLD}Einstellungen{C.RESET}")
        hr()
        fh = state.FORECAST_HORIZON_MONTHS
        fh_s = f"{fh} M" if fh else "Modelldefault"
        print(f"  {C.CYAN}1{C.RESET}  Status / Voraussetzungen")
        print(f"  {C.CYAN}2{C.RESET}  LLM anbinden")
        print(f"  {C.CYAN}3{C.RESET}  SEC User-Agent")
        print(f"  {C.CYAN}4{C.RESET}  Vorausschauhorizont  {C.DIM}({fh_s}){C.RESET}")
        print(
            f"  {C.CYAN}5{C.RESET}  Risiko-Schwellen  "
            f"{C.DIM}(mid={state.RISK_BAND_MID_MULT:.2f}, high={state.RISK_BAND_HIGH_MULT:.2f}){C.RESET}"
        )
        print(f"  {C.CYAN}6{C.RESET}  Fetch Zenodo")
        print(f"  {C.CYAN}7{C.RESET}  Convert Zenodo")
        print(f"  {C.CYAN}8{C.RESET}  Fetch EDGAR Financials")
        print(f"  {C.CYAN}9{C.RESET}  Fetch EDGAR 8-K Events")
        print(f"  {C.CYAN}10{C.RESET} Fetch Ratings (NRSRO)")
        print(f"  {C.CYAN}11{C.RESET} Modell trainieren")
        print(
            f"  {C.CYAN}12{C.RESET} Debug  "
            f"{C.DIM}({format_debug_status()}){C.RESET}"
        )
        print(f"  {C.CYAN}0{C.RESET}  Zurück")
        print()
        choice = ask("Auswahl", "0")

        if choice == "1":
            show_settings_status()
        elif choice == "2":
            settings_llm()
        elif choice == "3":
            settings_sec_ua()
        elif choice == "4":
            settings_forecast_horizon()
        elif choice == "5":
            settings_risk_bands()
        elif choice == "6":
            settings_fetch_zenodo()
        elif choice == "7":
            settings_convert_zenodo()
        elif choice == "8":
            settings_fetch_edgar_financials()
        elif choice == "9":
            settings_fetch_edgar_events()
        elif choice == "10":
            settings_fetch_ratings()
        elif choice == "11":
            settings_train()
        elif choice == "12":
            settings_debug()
        elif choice in {"0", "q", "b", "back"}:
            return
        else:
            print(f"  {C.YELLOW}Bitte 0–12 wählen.{C.RESET}")
            pause()
