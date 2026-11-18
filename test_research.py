import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

import api
from inference import memory_view, validate
from market import BASE, load_clean, panel
from research import claim_final


class MarketTest(unittest.TestCase):
    def make_input(self, root):
        dates = pd.bdate_range("2020-01-01", periods=100).date
        master, prices = [], []
        for t, day in enumerate(dates):
            for code in ("10000", "10010", "10020", "ETF00"):
                if code == "10010" and t == 75:
                    continue
                master.append([day, code, "0111", "999" if code == "ETF00" else "011"])
                price = (100 + t) * (0.5 if t >= 65 else 1)
                prices.append(
                    [
                        day,
                        code,
                        price,
                        price,
                        100 * (2 if t >= 65 else 1),
                        0.5 if t == 65 else 1,
                        "3" if code == "10020" and t == 70 else "",
                    ]
                )
        con = duckdb.connect()
        for name, frame in [
            ("calendar", pd.DataFrame({"Date": dates, "HolDiv": "1"})),
            (
                "master",
                pd.DataFrame(master, columns=["Date", "Code", "Mkt", "ProdCat"]),
            ),
            (
                "prices",
                pd.DataFrame(
                    prices,
                    columns=["Date", "Code", "O", "C", "Vo", "AdjFactor", "ExRT"],
                ),
            ),
        ]:
            folder = root / "clean" / name
            folder.mkdir(parents=True)
            con.from_df(frame).write_parquet(str(folder / "2020.parquet"))
        con.close()

    def test_clean_pit_panel(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con:
            self.make_input(Path(tmp))
            version = load_clean(con, tmp)
            self.assertEqual(len(version), 64)
            panel(con)
            r = con.execute(
                "SELECT return5,volume_ratio,y5 FROM events WHERE code='10000' AND s=62"
            ).fetchone()
            self.assertAlmostEqual(r[0], 162 / 157 - 1)
            self.assertAlmostEqual(r[1], 1)
            self.assertAlmostEqual(r[2], 168 / 163 - 1)
            self.assertEqual(
                con.execute(
                    "SELECT count(*) FROM events WHERE code='ETF00'"
                ).fetchone()[0],
                0,
            )
            for code, day in [("10010", 72), ("10020", 67)]:
                self.assertIsNone(
                    con.execute(
                        "SELECT y5 FROM events WHERE code=? AND s=?", [code, day]
                    ).fetchone()[0]
                )
            self.assertIsNone(
                con.execute(
                    "SELECT return5 FROM events WHERE code='10010' AND s=76"
                ).fetchone()[0]
            )
            self.assertAlmostEqual(
                con.execute(
                    "SELECT return5 FROM events WHERE code='10000' AND s=67"
                ).fetchone()[0],
                167 / 162 - 1,
            )
            self.assertAlmostEqual(
                con.execute(
                    "SELECT volume_ratio FROM events WHERE code='10000' AND s=67"
                ).fetchone()[0],
                1,
            )

    def test_partition_purge_and_concentration(self):
        from inference import study

        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con:
            self.make_input(Path(tmp))
            load_clean(con, tmp)
            panel(con)
            spec = dict(
                horizon=5, conditions=[dict(feature="return5", op="gt", value=0)]
            )
            v = study(con, spec, (0, 50))["5"]
            self.assertEqual(
                v["signals"], 40 * 3
            )  # signals on sessions 5..44, exits no later than 50
            self.assertEqual(v["active_days"], 40)
            self.assertAlmostEqual(v["excess"], 0)
            self.assertAlmostEqual(v["daily_net_excess"], -0.003)
            self.assertEqual(v["top5_share"], 1)

    def test_truncation_invariance(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con:
            self.make_input(Path(tmp))
            features = "return5,return20,return60,deviation25,volume_ratio,body_ratio,breadth,relative20"
            load_clean(con, tmp)
            con.execute("CREATE TABLE prefix AS SELECT * FROM bars WHERE s<=62")
            panel(con)
            first = con.execute(
                f"SELECT {features} FROM events WHERE s=62 ORDER BY code"
            ).fetchall()
            con.execute("DROP TABLE events")
            con.execute("DROP TABLE bench")
            con.execute("ALTER TABLE prefix RENAME TO bars")
            panel(con)
            second = con.execute(
                f"SELECT {features} FROM events WHERE s=62 ORDER BY code"
            ).fetchall()
            np.testing.assert_allclose(first, second, atol=1e-12)

    def test_specs_and_memory(self):
        s = dict(
            title="x",
            reason="y",
            horizon=5,
            conditions=[dict(feature="return5", op="lt", value=0.1)],
        )
        valid, rejected = validate(
            [s, s, {**s, "conditions": [dict(feature="y5", op="gt", value=0)]}], BASE
        )
        self.assertEqual((len(valid), len(rejected)), (1, 2))
        with self.assertRaises(ValueError):
            validate([s] * 11, BASE)
        row = {
            k: 0
            for k in (
                "signals",
                "codes",
                "daily_net_excess",
                "win_rate",
                "top5_share",
                "years",
                "regimes",
            )
        }
        m = memory_view(
            [
                {
                    "spec": valid[0],
                    "validation": {"5": row},
                    "final": {"secret_final": 40},
                    "decision": {},
                }
            ]
        )
        self.assertNotIn("secret_final", json.dumps(m))


class ApiTest(unittest.TestCase):
    def test_atomic_caps_and_lost_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = api.database(Path(tmp) / "state.db")
            api.reserve(db, "a", 1.6, "month")
            with self.assertRaises(ValueError):
                api.reserve(db, "a", 0.41, "month")
            for i in range(4):
                api.reserve(db, str(i), 2, "month")
            with self.assertRaises(ValueError):
                api.reserve(db, "other", 0.41, "month")
            self.assertEqual(api.usage(db, "a")["unconfirmed_reserved_usd"], 1.6)
            db.close()

    def test_api_generation_and_interpretation(self):
        def reply(request, timeout):
            body = json.loads(request.data)
            self.assertFalse(body["store"])
            self.assertNotIn("tools", body)
            x = (
                {"hypotheses": []}
                if "hypotheses" in body["input"]
                else {"analysis": "検証", "next": "次"}
            )
            return io.BytesIO(
                json.dumps(
                    dict(
                        status="completed",
                        usage=dict(input_tokens=100, output_tokens=25),
                        output=[
                            dict(content=[dict(type="output_text", text=json.dumps(x))])
                        ],
                    )
                ).encode()
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test"}),
        ):
            db = api.database(Path(tmp) / "state.db")
            original = api.call
            with patch.object(
                api, "call", side_effect=lambda *a: original(*a, transport=reply)
            ):
                self.assertEqual(
                    api.generate(db, "r", next(iter(api.RATES)), {}, {}, []), []
                )
                self.assertEqual(
                    api.interpret(db, "r", next(iter(api.RATES)), [])["next"], "次"
                )
            u = api.usage(db, "r")
            self.assertEqual(u["input_tokens"], 200)
            self.assertAlmostEqual(u["estimated_usd"], 0.00016)
            self.assertEqual(u["unconfirmed_reserved_usd"], 0)
            db.close()

    def test_holdout_once(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"SWING_FINAL_PC": "1"}),
        ):
            db = api.database(Path(tmp) / "state.db")
            claim_final(db, "a")
            with self.assertRaises(ValueError):
                claim_final(db, "b")
            db.close()


class AutomaticRunTest(unittest.TestCase):
    def test_generation_scoring_review_and_history(self):
        from argparse import Namespace
        from research import run

        def fixture(con, args):
            dates = pd.bdate_range("2021-01-04", "2024-04-30")
            from market import calendar

            calendar(con, dates)
            con.execute("""CREATE TABLE events AS SELECT s, 'TEST' code, .01::double return5,
                           .02::double y5, .03::double y10, .04::double y20 FROM days""")
            con.execute(
                """CREATE TABLE bench AS SELECT s,0.0 b5,0.0 b10,0.0 b20,'上昇' regime FROM days"""
            )
            return {"return5": "5日騰落"}, "synthetic", "synthetic", "synthetic"

        idea = {
            "title": "合成検査",
            "reason": "既知値",
            "horizon": 5,
            "conditions": [{"feature": "return5", "op": "gt", "value": 0}],
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test"}),
        ):
            root = Path(tmp)
            args = Namespace(
                clean=root / "data",
                cache=None,
                spec=None,
                output=root / "reports",
                previous=None,
                final=False,
                model=next(iter(api.RATES)),
            )
            with (
                patch("research.Path.home", return_value=root),
                patch("research.prepare", side_effect=fixture),
                patch("research.api.generate", return_value=[idea]) as generate,
                patch(
                    "research.api.interpret",
                    return_value={"analysis": "検証のみ", "next": "次の案"},
                ),
            ):
                out = run(args)
                first = json.loads((out / "result.json").read_text(encoding="utf8"))
                self.assertEqual(first["results"][0]["decision"]["status"], "標本不足")
                self.assertIn("最終期間は未評価", first["final_status"])
                self.assertTrue((out / "report.html").is_file())
                run(args)
                history = generate.call_args.args[-1]
                self.assertEqual(history[0]["validation"][0]["title"], "合成検査")
                self.assertNotIn('"final"', json.dumps(history))
