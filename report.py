import html
import json
from urllib.parse import quote


def esc(v):
    return html.escape(str(v), quote=True)


def pct(v):
    return "—" if v is None else f"{v * 100:+.2f}%"


def share(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def table(head, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(x)}</th>" for x in head)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def render(r):
    head = [
        "仮説",
        "主期間",
        "成立件数",
        "銘柄数",
        "成立日数",
        "平均騰落",
        "費用後超過¹",
        "判定",
    ]
    rows, details = [], []
    for item in r["results"]:
        s = item["spec"]
        h = str(s["horizon"])
        v = item["validation"][h]
        rows.append(
            [
                s["id"] + " " + s["title"],
                h + "日",
                f"{v['signals']:,}",
                f"{v['codes']:,}",
                v["active_days"],
                pct(v["mean_return"]),
                pct(v["daily_net_excess"]),
                item["decision"]["status"],
            ]
        )
        op = {"gt": ">", "ge": "≥", "lt": "<", "le": "≤"}
        conditions = " かつ ".join(
            f"{r['features'][x['feature']]} {op[x['op']]} {x['value']:g}"
            for x in s["conditions"]
        )
        sections = []
        for part, label in [
            ("train", "学習期間"),
            ("validation", "検証期間"),
            ("final", "最終評価期間"),
        ]:
            if part not in item:
                continue
            horizons = item[part]
            sections.append(
                f"<h4>{label}</h4>"
                + table(
                    [
                        "保有",
                        "成立",
                        "観測済",
                        "銘柄",
                        "上昇率",
                        "平均",
                        "中央値",
                        "市場超過（費用前）",
                    ],
                    [
                        [
                            k + "日",
                            f"{v['signals']:,}",
                            f"{v['observed']:,}",
                            v["codes"],
                            share(v["win_rate"]),
                            pct(v["mean_return"]),
                            pct(v["median_return"]),
                            pct(v["excess"]),
                        ]
                        for k, v in horizons.items()
                    ],
                )
            )
            p = horizons[h]
            sections.append(
                table(
                    ["主期間の内訳", "件数", "観測済", "平均", "市場超過（費用前）"],
                    [
                        [
                            y["group"],
                            y["signals"],
                            y["observed"],
                            pct(y["mean"]),
                            pct(y["excess"]),
                        ]
                        for y in p["years"] + p["regimes"]
                    ],
                )
            )
            sections.append(
                f"<p>上位5銘柄への件数集中 {share(p['top5_share'])} ／ 最大年の件数割合 {share(p['top_year_share'])} ／ 費用後の日付均等超過 {pct(p['daily_net_excess'])}</p>"
            )
        adj = item["decision"]["p_adjusted"]
        details.append(
            f"<details><summary>{esc(s['id'] + ' ' + s['title'])}</summary><p>{esc(s['reason'])}</p><p><b>成立条件</b>　{esc(conditions)}</p><p>主期間 {h}日 ／ 多重検定補正後の片側p値 {esc(round(adj, 4) if adj is not None else '—')}</p>"
            + "".join(sections)
            + "</details>"
        )
    m = r["measurement"]
    data = r["data"]
    api = r["api"]
    note = r.get("interpretation", {})

    def download(label, filename, value):
        uri = "data:application/json;charset=utf-8," + quote(
            json.dumps(value, ensure_ascii=False, indent=2)
        )
        return f'<a download="{filename}" href="{uri}">{label}</a>'

    specs = [
        {k: v for k, v in item["spec"].items() if k != "id"} for item in r["results"]
    ]
    downloads = (
        download("仮説仕様を保存", "hypotheses.json", specs)
        + "　"
        + download("研究結果を保存", "result.json", r)
    )
    spans = " ／ ".join(
        f"{label} {span[0]}〜{span[1]}"
        for label, span in zip(("学習", "検証", "最終評価"), r["periods"].values())
    )
    return f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>日本株 AI 仮説研究</title>
<style>*{{box-sizing:border-box}}body{{margin:auto;max-width:1120px;padding:36px 22px 80px;font:16px/1.65 system-ui,sans-serif;color:#1c2c35;background:#f7faf9}}h1{{font-size:clamp(26px,4vw,42px);margin:0}}h2{{margin-top:36px}}header{{border-bottom:4px solid #167163;padding-bottom:18px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}}th,td{{padding:7px 10px;border-bottom:1px solid #dbe4e1;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:#164e46;background:#e8f2ef}}p{{white-space:pre-line}}details{{background:white;padding:10px 14px;border-top:1px solid #dbe4e1}}summary{{font-weight:650;cursor:pointer}}code{{overflow-wrap:anywhere}}small{{color:#49616b}}</style>
<header><h1>日本株 AI 仮説研究</h1><p>{esc(data["universe"])} ｜ {data["codes"]:,}銘柄・{data["rows"]:,}銘柄日</p>{esc(spans)}</header>
<h2>検証期間の成績</h2>{table(head, rows)}
<p>¹ 各日の平均から同日市場平均と往復0.30%を引き、成立日を均等に平均。平均騰落は観測できた銘柄日を均等に平均。成立日が特定日に集中すると二つの集計の重みが変わる。</p>
<p><b>最終評価</b>　{esc(r["final_status"])}</p>
<h2>結果の解釈</h2><p>{esc(note.get("analysis", "未実施"))}</p><h3>次の研究課題</h3><p>{esc(note.get("next", "未設定"))}</p>
<h2>各仮説の条件と結果</h2>{"".join(details)}
<h2>検証方法と制約</h2><p>シグナル日引けまでの条件で翌営業日始値に買い、5・10・20営業日後の始値で評価。各区間の境界をまたぐラベルは除外。市場基準は{esc(data["benchmark"])}。地合いは同日の市場平均20日リターンの符号。</p>
<p>主期間は各仮説につき事前に一つ。成立日で集約し、保有期間の2倍のラグを用いたHAC標準誤差と片側検定、試行数のBonferroni補正を使用。補正後p&lt;0.05、費用後超過が正、20銘柄・100日・200観測以上、観測率95%以上、上位5銘柄の件数比率25%以下を候補とする。</p>
<p>推定普通株の当日在籍を使用。欠けた価格、権利・在籍の切れ目をまたぐ結果は観測不能。分割を調整した価格リターンで、配当・廃止対価・売買単位・値幅制限・流動性による約定可否は未反映。費用は仮定値。重複シグナルを含むイベント研究のため資金500万円の実現損益は未計算。</p>
<h2>実行記録</h2><p>{esc(r["origin"])} ／ {esc(r["model"])}<br>計算・API待ち {m["seconds"]:.2f}秒 ／ プロセスピーク {m["peak_mib"]:.1f} MiB ／ API {api["calls"]}回、入力 {api["input_tokens"]:,}・出力 {api["output_tokens"]:,}トークン、推定 ${api["estimated_usd"]:.5f}、応答未確認の予約 ${api["unconfirmed_reserved_usd"]:.5f}。</p>
<p>データ版 <code>{esc(data["version"])}</code></p><p>仮説仕様 SHA-256 <code>{esc(r["spec_sha256"])}</code></p><small>各PCの研究記録には不成立・棄却も含まれる。APIクレジットの残高・期限は取得していない。</small><p>{downloads}</p></html>"""
