#!/usr/bin/env python3
"""Simulate decide_signal + new fallback on the latest snapshot per ticker.

Answers: how many BUY signals emerge from rules after the moat-fix?
"""
import sys, sqlite3, pandas as pd
sys.path.insert(0, '.')
from buffett.scorer import compute_quant_score, decide_signal
from buffett.moat_llm import MoatLLMJudge

con = sqlite3.connect('/home/shalu/buffett-monitor/data/buffett.db')
latest = con.execute("""
  SELECT s.ticker, s.snapshot_date, s.quant_score, s.moat_strength,
         f.price, f.intrinsic_value, f.margin_of_safety,
         f.roe_latest, f.de_ratio, f.current_ratio
  FROM buffett_scores s
  JOIN buffett_fundamentals f ON s.ticker=f.ticker AND s.snapshot_date=f.snapshot_date
  WHERE s.snapshot_date = (SELECT MAX(snapshot_date) FROM buffett_scores WHERE ticker = s.ticker)
""").fetchall()
cols = ["ticker","snapshot_date","quant_score","moat_strength",
        "price","intrinsic_value","margin_of_safety",
        "roe_latest","de_ratio","current_ratio"]
df = pd.DataFrame(latest, columns=cols)
print(f"total latest snapshots: {len(df)}")

judge = MoatLLMJudge()
fresh = []
for _, row in df.iterrows():
    fund = {"roe": row.get("roe_latest") or 0.0,
            "debt_to_equity": row.get("de_ratio") or float('inf')}
    cr = row.get("current_ratio")
    fund["current_ratio"] = cr if pd.notna(cr) else 0.0
    j = judge._fallback_judgment(fund)
    price = row["price"] if pd.notna(row["price"]) else 0.0
    iv = row["intrinsic_value"] if pd.notna(row["intrinsic_value"]) else 0.0
    qs = row["quant_score"] if pd.notna(row["quant_score"]) else 0.0
    sig = decide_signal(
        quant_score=qs, moat_strength=j["moat_strength"],
        fundamentals_flag="NORMAL", price=price, intrinsic_value=iv)
    fresh.append((row["ticker"], row["snapshot_date"], qs,
                  j["moat_strength"], j["pillar1"], j["pillar2"],
                  row["margin_of_safety"], sig))

result = pd.DataFrame(fresh, columns=["ticker","sd","qs","moat","p1","p2","mos","signal"])
print()
print("=== signal distribution (re-decided) ===")
print(result['signal'].value_counts())
print()
print("=== moat distribution ===")
print(result['moat'].value_counts())
print()
buys = result[result['signal']=='BUY']
print(f"=== {len(buys)} BUY rows ===")
if len(buys) > 0:
    print(buys.head(40).to_string())
print()

quant_60 = result[result['qs'] >= 60]
moat_strong = result[result['moat'] == 'STRONG']
mos_20 = result[result['mos'].fillna(-1) >= 0.20]
print("=== BUY funnels ===")
print(f"  total snapshots:        {len(result)}")
print(f"  quant_score >= 60:     {len(quant_60)}")
print(f"  moat == STRONG:        {len(moat_strong)}")
print(f"  MoS >= 20%:            {len(mos_20)}")
n_buy = ((result['qs']>=60) & (result['moat']=='STRONG') & (result['mos'].fillna(-1)>=0.20)).sum()
print(f"  BUY = qty AND moat AND mos: {n_buy}")
con.close()
