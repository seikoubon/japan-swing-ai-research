"""Validated predicate grammar and session-clustered event studies."""

import math

import numpy as np

from market import HORIZONS

OPERATORS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}
COST = 0.003  # Gross equal-weight benchmark versus 15 bp per side on the idea.
TRAIN_END = "2021-12-31"
VALID_END = "2023-12-31"


def validate(items, features):
    if not isinstance(items, list) or not 1 <= len(items) <= 10:
        raise ValueError("Expected 1–10 hypotheses")
    valid, rejected, seen = [], [], set()
    for i, spec in enumerate(items, 1):
        try:
            if not isinstance(spec, dict) or set(spec) != {
                "title",
                "reason",
                "horizon",
                "conditions",
            }:
                raise ValueError("Invalid hypothesis fields")
            if any(
                not isinstance(spec[k], str) or not 1 <= len(spec[k].strip()) <= 800
                for k in ("title", "reason")
            ):
                raise ValueError("Invalid explanation")
            if type(spec["horizon"]) is not int or spec["horizon"] not in HORIZONS:
                raise ValueError("Invalid primary horizon")
            rules = spec["conditions"]
            if not isinstance(rules, list) or not 1 <= len(rules) <= 4:
                raise ValueError("Expected 1–4 AND conditions")
            for r in rules:
                if not isinstance(r, dict) or set(r) != {"feature", "op", "value"}:
                    raise ValueError("Invalid condition")
                if r["feature"] not in features or r["op"] not in OPERATORS:
                    raise ValueError("Unknown feature/operator")
                if type(r["value"]) not in (int, float) or not math.isfinite(
                    r["value"]
                ):
                    raise ValueError("Expected finite numeric threshold")
            signature = tuple(
                sorted((r["feature"], r["op"], r["value"]) for r in rules)
            )
            if signature in seen:
                raise ValueError("Duplicate conditions")
            seen.add(signature)
            valid.append(dict(spec, id=f"H{i:02d}"))
        except (TypeError, KeyError, ValueError, OverflowError) as error:
            rejected.append({"index": i, "spec": spec, "error": str(error)})
    return valid, rejected


def selection(con, spec, first, last):
    # Only validated feature and operator names reach SQL. Values are bound.
    rules = spec["conditions"]
    where = " AND ".join(f"{r['feature']} {OPERATORS[r['op']]} ?" for r in rules)
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE selected AS
        SELECT e.s,e.code,d.d,b.regime,e.y5,e.y10,e.y20,b.b5,b.b10,b.b20
        FROM events e JOIN bench b USING(s) JOIN days d USING(s)
        WHERE e.s BETWEEN ? AND ? AND {where}""",
        [first, last, *[r["value"] for r in rules]],
    )


def hac(daily, first, end, h):
    """Mean of active dates; inactive dates contribute zero influence at true calendar lag."""
    if len(daily) < 2:
        return None, None
    dates, values = zip(*daily)
    avg = float(np.mean(values))
    x = np.zeros(end - first + 1)
    x[np.asarray(dates) - first] = np.asarray(values) - avg
    variance = float(x @ x)
    # Bartlett HAC accounts for overlapping holds and common same-day market shocks.
    for lag in range(1, min(2 * h, len(x) - 1) + 1):
        variance += 2 * (1 - lag / (2 * h + 1)) * float(x[lag:] @ x[:-lag])
    se = math.sqrt(max(0, variance)) / len(values)
    if se <= 1e-12:
        return avg, None
    return avg, 0.5 * math.erfc(avg / se / math.sqrt(2))


def study(con, spec, bounds):
    first, last = bounds
    selection(con, spec, first, last)
    result = {}
    for h in HORIZONS:
        end = last - h - 1
        y, b = f"y{h}", f"b{h}"
        sql = f"FROM selected WHERE s<={int(end)}"
        row = con.execute(f"""SELECT count(*) signals,count(DISTINCT code) codes,count({y}) observed,
            avg({y}) mean_return,median({y}) median_return,avg(({y}>0)::int) win_rate,
            avg({y}-{b}) excess,count({y}-{b}) paired {sql}""").fetchone()
        row = dict(
            zip(
                (
                    "signals",
                    "codes",
                    "observed",
                    "mean_return",
                    "median_return",
                    "win_rate",
                    "excess",
                    "paired",
                ),
                row,
            )
        )
        if h == spec["horizon"]:
            days = con.execute(
                f"SELECT s,avg({y}-{b})-{COST} a {sql} GROUP BY s HAVING a IS NOT NULL ORDER BY s"
            ).fetchall()
            row["active_days"] = len(days)
            row["daily_net_excess"], row["p"] = (
                hac(days, first, end, h) if days else (None, None)
            )
            for key, group in [("years", "year(d)"), ("regimes", "regime")]:
                rows = con.execute(
                    f"SELECT {group} g,count(*),count({y}),avg({y}),avg({y}-{b}) {sql} GROUP BY 1 ORDER BY 1"
                ).fetchall()
                row[key] = [
                    dict(zip(("group", "signals", "observed", "mean", "excess"), v))
                    for v in rows
                ]
            top = con.execute(
                f"SELECT code,count(*) n {sql} GROUP BY code ORDER BY n DESC,code LIMIT 5"
            ).fetchall()
            row["top5_share"] = sum(n for _, n in top) / row["signals"] if top else None
            row["top_year_share"] = max(
                (t["signals"] / row["signals"] for t in row["years"]), default=None
            )
        result[str(h)] = row
    return result


def decide(spec, validation, attempts):
    primary = validation[str(spec["horizon"])]
    p = primary["p"]
    adjusted = min(1.0, attempts * p) if p is not None else None
    if (
        primary["codes"] < 20
        or primary["active_days"] < 100
        or primary["observed"] < 200
    ):
        status = "標本不足"
    elif primary["paired"] / primary["signals"] < 0.95 or primary["top5_share"] > 0.25:
        status = "観測率・集中度で保留"
    elif adjusted is not None and adjusted < 0.05 and primary["daily_net_excess"] > 0:
        status = "候補"
    else:
        status = "優位性を支持せず"
    return dict(status=status, p_adjusted=adjusted)


def bounds(con):
    ranges = {}
    for key, expr, args in [
        ("train", "d<=?", [TRAIN_END]),
        ("validation", "d>? AND d<=?", [TRAIN_END, VALID_END]),
        ("final", "d>?", [VALID_END]),
    ]:
        a, z = con.execute(
            "SELECT min(s),max(s) FROM days WHERE " + expr, args
        ).fetchone()
        if a is None or z - a < 22:
            raise ValueError(f"Insufficient {key} calendar")
        ranges[key] = (a, z)
    return ranges


def profile(con, features):
    expr = ",".join(f"approx_quantile({f},[.1,.5,.9])" for f in features)
    values = con.execute(
        f"SELECT {expr} FROM events JOIN days USING(s) WHERE d<=?", [TRAIN_END]
    ).fetchone()
    return {
        k: [round(v, 5) if v is not None else None for v in vs] if vs else None
        for k, vs in zip(features, values)
    }


def winner(results):
    passed = [r for r in results if r["decision"]["status"] == "候補"]
    return (
        min(
            passed,
            key=lambda r: (
                r["decision"]["p_adjusted"],
                -r["validation"][str(r["spec"]["horizon"])]["daily_net_excess"],
                r["spec"]["id"],
            ),
        )
        if passed
        else None
    )


def memory_view(results):
    # Explicit projection. Final returns and user notes never enter the next model prompt.
    rows = []
    for r in results:
        h = str(r["spec"]["horizon"])
        primary = r["validation"][h]
        rows.append(
            {
                **r["spec"],
                "decision": r["decision"],
                **{
                    k: primary[k]
                    for k in (
                        "signals",
                        "codes",
                        "daily_net_excess",
                        "win_rate",
                        "top5_share",
                        "years",
                        "regimes",
                    )
                },
            }
        )
    return rows
