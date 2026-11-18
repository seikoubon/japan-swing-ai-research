"""Run with uv run python research.py --help."""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import duckdb

import api
from inference import (
    VALID_END,
    bounds,
    decide,
    memory_view,
    profile,
    study,
    validate,
    winner,
)
from market import prepare
from report import render


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )
    tmp.replace(path)


def peak_mib():
    if os.name == "nt":
        import psutil

        return psutil.Process().memory_info().peak_wset / (1024**2)
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )


def claim_final(db, run):
    if os.environ.get("SWING_FINAL_PC") != "1":
        raise ValueError(
            "Set SWING_FINAL_PC=1 only on the designated final-evaluation PC"
        )
    try:
        with db:
            db.execute("INSERT INTO holdout VALUES(?,?)", (VALID_END, run))
    except sqlite3.IntegrityError as e:
        raise ValueError("Final period has already been opened") from e


def inputs():
    p = argparse.ArgumentParser(description="J-Quants 日本株AIイベント研究")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--clean", type=Path, help="同期済み data フォルダー（直下に clean）"
    )
    source.add_argument(
        "--cache",
        type=Path,
        help="2026-11-13版 Stage 6 の meta.json と bars.bin があるフォルダー",
    )
    p.add_argument("--spec", type=Path, help="保存済みJSONによる無課金の再検証")
    p.add_argument(
        "--previous",
        type=Path,
        help="別PCの初回レポートから保存した result.json。検証期間の要約だけを参照",
    )
    p.add_argument("--model", choices=list(api.RATES), default=next(iter(api.RATES)))
    p.add_argument(
        "--final",
        action="store_true",
        help="指定PCで検証通過の最上位1件に限り最終評価を開く",
    )
    p.add_argument("--output", type=Path, default=Path("reports"))
    return p.parse_args()


def run(args):
    if not args.spec and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is unset; set it locally or use --spec")
    protected = (args.clean or args.cache).resolve()
    if args.output.resolve().is_relative_to(protected):
        raise ValueError("Output must be outside input data")
    run_id = (
        datetime.now(api.JST).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    out = args.output / run_id
    out.mkdir(parents=True)
    state = Path.home() / ".japan-swing-ai-research" / "state.sqlite3"
    db = api.database(state)
    db.execute("CREATE TABLE IF NOT EXISTS lessons(run TEXT PRIMARY KEY,content TEXT)")
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with duckdb.connect(
                str(Path(tmp) / "work.duckdb"),
                config={"threads": 2, "memory_limit": "1500MB"},
            ) as con:
                features, version, universe, benchmark = prepare(con, args)
                periods = bounds(con)
                n, c = con.execute(
                    "SELECT count(*),count(distinct code) FROM events"
                ).fetchone()
                data = dict(
                    version=version,
                    universe=universe,
                    benchmark=benchmark,
                    rows=n,
                    codes=c,
                )
                historical = [
                    json.loads(s)
                    for (s,) in db.execute(
                        "SELECT content FROM lessons ORDER BY rowid DESC LIMIT 2"
                    )
                ]
                if args.previous:
                    prior = json.loads(args.previous.read_text(encoding="utf8"))
                    if prior["periods"]["validation"][1] > VALID_END:
                        raise ValueError(
                            "Previous results include dates after the validation cutoff"
                        )
                    historical.append({"validation": memory_view(prior["results"])})
                generated = (
                    json.loads(args.spec.read_text(encoding="utf8"))
                    if args.spec
                    else api.generate(
                        db,
                        run_id,
                        args.model,
                        features,
                        profile(con, features),
                        historical,
                    )
                )
                save(out / "hypotheses.json", generated)
                specs, invalid = validate(generated, features)
                span = {
                    name: [
                        str(d)
                        for (d,) in con.execute(
                            "SELECT d FROM days WHERE s IN (?,?) ORDER BY s", pair
                        ).fetchall()
                    ]
                    for name, pair in periods.items()
                }
                results = []
                model = "外部仕様・API未使用" if args.spec else args.model
                report = dict(
                    run_id=run_id,
                    origin="保存済み仮説を再生" if args.spec else "OpenAI API 自動生成",
                    model=model,
                    data=data,
                    features=features,
                    periods=span,
                    rejected=invalid,
                    results=results,
                    spec_sha256=hashlib.sha256(
                        (out / "hypotheses.json").read_bytes()
                    ).hexdigest(),
                    final_status="候補なし。最終期間は未評価。",
                )
                for spec in specs:
                    train = study(con, spec, periods["train"])
                    validation = study(con, spec, periods["validation"])
                    results.append(
                        dict(
                            spec=spec,
                            train=train,
                            validation=validation,
                            decision=decide(spec, validation, len(generated)),
                        )
                    )
                memory = memory_view(results)
                if not args.spec and results:
                    try:
                        report["interpretation"] = api.interpret(
                            db, run_id, args.model, memory
                        )
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        report["interpretation"] = {
                            "analysis": "APIによる解釈は未完了。" + type(exc).__name__,
                            "next": "未生成",
                        }
                with db:
                    db.execute(
                        "INSERT INTO lessons VALUES(?,?)",
                        (
                            run_id,
                            json.dumps(
                                {
                                    "validation": memory,
                                    "rejected": len(invalid),
                                    "interpretation": report.get("interpretation"),
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                best = winner(results)
                if best:
                    report["final_status"] = "検証通過。最終期間は未評価。"
                    if args.final:
                        try:
                            claim_final(db, run_id)
                        except ValueError as exc:
                            report["final_status"] = str(exc)
                        else:
                            best["final"] = study(con, best["spec"], periods["final"])
                            report["final_status"] = (
                                best["spec"]["id"]
                                + " の事前指定主期間を最終評価。詳細は下表。"
                            )
        report["measurement"] = dict(
            seconds=round(time.perf_counter() - started, 3),
            peak_mib=round(peak_mib(), 2),
        )
        report["api"] = api.usage(db, run_id)
        save(out / "result.json", report)
        (out / "report.html").write_text(render(report), encoding="utf8")
        print(out.resolve() / "report.html")
        return out
    finally:
        print(json.dumps(api.usage(db, run_id), ensure_ascii=False))
        db.close()


if __name__ == "__main__":
    run(inputs())
