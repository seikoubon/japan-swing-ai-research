"""Read J-Quants local snapshots. All generated relations live in a temporary DuckDB."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20)
MARKETS = ("0111", "0112", "0113", "0101", "0102", "0104", "0106", "0107")
BASE = {
    "return5": "5営業日の終値リターン",
    "return20": "20営業日の終値リターン",
    "return60": "60営業日の終値リターン",
    "volume_ratio": "当日出来高÷直前20日平均（倍率）",
    "body_ratio": "当日終値÷当日始値−1",
    "deviation25": "当日終値÷25日平均−1",
}
BROAD = {
    "breadth": "当日の上昇銘柄割合",
    "relative20": "20日騰落率−当日の対象銘柄の平均20日騰落率",
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def calendar(con, dates):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if dates.empty or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Invalid trading calendar")
    frame = pd.DataFrame({"d": dates.date, "s": np.arange(len(dates), dtype="int32")})
    con.from_df(frame).create("days")


def load_clean(con, root):
    clean = Path(root) / "clean"
    files = []
    for name in ("master", "prices", "calendar"):
        paths = sorted((clean / name).glob("*.parquet"))
        if not paths:
            raise ValueError(f"Missing {clean / name}")
        con.read_parquet([str(p) for p in paths], union_by_name=True).create_view(name)
        files.extend(paths)
    stamps = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    sha = hashlib.sha256()
    for p in files:
        sha.update((p.relative_to(clean).as_posix() + ":" + digest(p)).encode())
    for table, keys in [
        ("prices", "Date,Code"),
        ("master", "Date,Code"),
        ("calendar", "Date"),
    ]:
        if con.execute(
            f"SELECT 1 FROM {table} GROUP BY {keys} HAVING count(*)>1 LIMIT 1"
        ).fetchone():
            raise ValueError(f"Duplicate {table} key")
    dates = con.execute("""SELECT Date FROM calendar WHERE HolDiv IN ('1','2') AND Date BETWEEN
                       (SELECT min(Date) FROM master) AND (SELECT max(Date) FROM master) ORDER BY Date""").fetchall()
    calendar(con, [d for (d,) in dates])
    if con.execute("SELECT 1 FROM days ANTI JOIN master ON d=Date LIMIT 1").fetchone():
        raise ValueError("Missing daily master")
    cols = set(con.table("prices").columns)
    if "ExRT" not in cols:
        raise ValueError("Missing ExRT rights flag")
    con.execute(
        """CREATE TABLE bars AS SELECT m.Code code, s, p.O op, p.C cl, p.Vo vol,
                   p.AdjFactor factor, coalesce(cast(p.ExRT as varchar),'')='3' rights
                   FROM master m JOIN days ON m.Date=d LEFT JOIN prices p USING(Date,Code)
                   WHERE m.Mkt IN (SELECT unnest(?)) AND m.ProdCat='011' AND right(m.Code,1)='0'""",
        [list(MARKETS)],
    )
    if stamps != [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]:
        raise ValueError("Input changed during read. Retry after local sync.")
    return sha.hexdigest()


def load_cache(con, root):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf8"))
    codes = meta["codes"]
    dates = [x[0] for x in meta["days"]]
    if len(set(codes)) != len(codes) or meta["identity"]["cache_version"] != 2:
        raise ValueError("Unsupported prepared cache")
    sha = digest(root / "bars.bin")
    if sha != meta["files"]["bars.bin"]:
        raise ValueError("bars.bin checksum mismatch")
    # Legacy caches have no authenticated code/date mapping. Pin the known producer mapping.
    mapping = hashlib.sha256(
        json.dumps([codes, dates], separators=(",", ":")).encode()
    ).hexdigest()
    if (
        sha != "3fb87642b966170151ae3c1554479ac56deaefc6d7f2637c688eb38ea6d36cf4"
        or mapping != "0a4c94ec3279c266a87d9965389ea8e47d67803b238ce154883e8bfd5f9bdc1e"
    ):
        raise ValueError("Unknown legacy mapping. Use current clean Parquet.")
    calendar(con, dates)
    dtype = np.dtype(
        [
            ("op", "<f8"),
            ("cl", "<f8"),
            ("vol", "<f8"),
            ("factor", "<f8"),
            ("episode", "<i4"),
            ("rights", "u1"),
            ("eligible", "u1"),
        ]
    )
    if (root / "bars.bin").stat().st_size != len(dates) * len(codes) * dtype.itemsize:
        raise ValueError("Invalid cache length")
    arr = np.memmap(
        root / "bars.bin", dtype=dtype, mode="r", shape=(len(dates), len(codes))
    )
    labels = np.asarray(codes)
    con.execute(
        "CREATE TABLE bars(code VARCHAR,s INTEGER,op DOUBLE,cl DOUBLE,vol DOUBLE,factor DOUBLE,rights BOOLEAN)"
    )
    for start in range(0, len(dates), 64):
        chunk = arr[start : start + 64]
        day, stock = np.where(chunk["eligible"] == 1)
        values = chunk[day, stock]
        frame = pd.DataFrame(
            {
                "code": labels[stock],
                "s": day + start,
                **{k: values[k] for k in ("op", "cl", "vol", "factor", "rights")},
            }
        )
        con.from_df(frame).insert_into("bars")
    return sha + "/" + mapping


def panel(con):
    # Each gap, unknown adjustment or rights event breaks the series. Retain all signal rows;
    # an unobserved outcome is NULL and contributes to the reported coverage denominator.
    forward = ", ".join(
        f"""CASE WHEN lead(s,{h + 1}) OVER w=s+{h + 1}
        THEN lead(aop,{h + 1}) OVER w / nullif(lead(aop) OVER w,0)-1 END AS y{h}"""
        for h in HORIZONS
    )
    trailing = ", ".join(
        f"acl / lag(acl,{h}) OVER w - 1 AS return{h}" for h in (5, 20, 60)
    )
    con.execute(f"""CREATE TABLE events AS WITH valid AS (
        SELECT *, s-row_number() OVER(PARTITION BY code ORDER BY s) AS segment FROM bars
        WHERE cl>0 AND vol>0 AND factor>0 AND NOT rights AND isfinite(cl) AND isfinite(factor)
    ), scaled AS (SELECT *, exp(sum(ln(factor)) OVER w) mult FROM valid
        WINDOW w AS (PARTITION BY code,segment ORDER BY s ROWS UNBOUNDED PRECEDING)
    ), prices AS (SELECT *, cl/mult acl, CASE WHEN op>0 THEN op/mult END aop, vol*mult avol FROM scaled),
    calc AS (SELECT s,code, {trailing}, acl/lag(acl) OVER w-1 ret1,
        CASE WHEN count(*) OVER prev20=20 THEN avol/nullif(avg(avol) OVER prev20,0) END volume_ratio,
        CASE WHEN count(*) OVER days25=25 THEN acl/avg(acl) OVER days25-1 END deviation25,
        acl/nullif(aop,0)-1 body_ratio, {forward} FROM prices
        WINDOW w AS (PARTITION BY code,segment ORDER BY s),
        prev20 AS (w ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        days25 AS (w ROWS BETWEEN 24 PRECEDING AND CURRENT ROW))
        SELECT *, avg((ret1>0)::int) OVER(PARTITION BY s) breadth,
        return20-avg(return20) OVER(PARTITION BY s) relative20 FROM calc""")
    con.execute("DROP TABLE bars")
    # Equal-weight benchmark has exactly the same entry/exit clock as the stock.
    means = ", ".join(f"avg(y{h}) b{h}" for h in HORIZONS)
    con.execute(
        f"CREATE TABLE bench AS SELECT s, {means}, CASE WHEN avg(return20) IS NULL THEN '不明' WHEN avg(return20)>=0 THEN '上昇' ELSE '下落' END regime FROM events GROUP BY s"
    )


def prepare(con, args):
    version = load_clean(con, args.clean) if args.clean else load_cache(con, args.cache)
    panel(con)
    return (
        BASE | BROAD,
        version,
        "日別在籍の東証普通株近似・当日売買あり",
        "同日等ウェイト対象銘柄（観測可能な翌日始値から出口始値）",
    )
