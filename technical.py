"""Price action + ICT + Volume Profile (POC/VAH/VAL) + VWAP + moteur de signal."""
import io, time, datetime as dt
import numpy as np, pandas as pd, requests, yfinance as yf
import fundamentals

PLAYBOOK = [
    "1. Biais fondamental d'abord : ne trader que dans le sens du biais macro (ou rester à l'écart).",
    "2. Structure : BOS = continuation, CHoCH = premier signe de retournement. Suivre la structure du timeframe.",
    "3. Premium/Discount : acheter en discount (<50% du range), vendre en premium (>50%). Zone idéale OTE 62-79%.",
    "4. Liquidité : le prix cherche les equal highs/lows et PDH/PDL. Un sweep suivi d'un rejet = entrée de qualité.",
    "5. PD Arrays : entrer sur FVG / Order Block non mitigé aligné avec le sens du trade.",
    "6. Volume Profile : POC = aimant, VAH/VAL = bords de la value area. Sortie de value + acceptation = tendance ; rejet = retour vers POC.",
    "7. VWAP : au-dessus = acheteurs dominants, en dessous = vendeurs. Éviter d'acheter loin sous le VWAP en tendance haussière intraday.",
    "8. Timing : privilégier les killzones London (07-10 UTC) et New York (12-15 UTC).",
    "9. Risque : SL derrière la zone + 0.5 ATR, viser RR >= 2, risque <= 1% du capital par trade.",
]

def load(symbol, interval):
    per = {"5m": "30d", "15m": "30d", "1h": "180d", "4h": "360d", "1d": "2y"}
    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=per[interval], interval="1h" if interval == "4h" else interval,
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.index.tz is not None: df.index = df.index.tz_convert("UTC").tz_localize(None)
        if interval == "4h":
            df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    except Exception:
        df = pd.DataFrame()
    if df.empty:  # fallback Stooq (daily)
        s = symbol.lower()
        s = s.replace("=x", "") if s.endswith("=x") else (s if ("." in s or s.startswith("^")) else s + ".us")
        r = requests.get(f"https://stooq.com/q/d/l/?s={s}&i=d", timeout=15)
        df = pd.read_csv(io.StringIO(r.text), index_col="Date", parse_dates=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().tail(500)
    if len(df) < 60: raise ValueError("Pas assez de données pour ce symbole/timeframe")
    return df

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def swings(df, k=3):
    H, L, hi, lo = df.High.values, df.Low.values, [], []
    for i in range(k, len(df) - k):
        if H[i] == H[i - k:i + k + 1].max(): hi.append(i)
        if L[i] == L[i - k:i + k + 1].min(): lo.append(i)
    return hi, lo

def vwap(df, daily):
    tp = (df.High + df.Low + df.Close) / 3
    v = df.Volume.replace(0, np.nan).fillna(1)
    key = df.index.to_period("M") if daily else df.index.date
    return (tp * v).groupby(key).cumsum() / v.groupby(key).cumsum()

def profile(df, bins=60):
    lo, hi = df.Low.min(), df.High.max()
    edges = np.linspace(lo, hi, bins + 1); vol = np.zeros(bins)
    v = df.Volume.replace(0, np.nan).fillna(1).values
    for l, h, x in zip(df.Low.values, df.High.values, v):
        a = max(np.searchsorted(edges, l, side="right") - 1, 0)
        b = min(np.searchsorted(edges, h, side="right") - 1, bins - 1)
        vol[a:b + 1] += x / (b - a + 1)
    p = int(vol.argmax()); acc = vol[p]; i = j = p
    while acc < 0.7 * vol.sum() and (i > 0 or j < bins - 1):
        up = vol[j + 1] if j < bins - 1 else -1
        dn = vol[i - 1] if i > 0 else -1
        if up >= dn: j += 1; acc += vol[j]
        else: i -= 1; acc += vol[i]
    return {"poc": (edges[p] + edges[p + 1]) / 2, "vah": edges[j + 1], "val": edges[i]}

def fvgs(df, a):
    H, L, out = df.High.values, df.Low.values, []
    for i in range(2, len(df)):
        if np.isnan(a[i]): continue
        if L[i] > H[i - 2] and L[i] - H[i - 2] > 0.2 * a[i] and L[i + 1:].min(initial=np.inf) > H[i - 2]:
            out.append(("bull", L[i], H[i - 2], i - 1))
        if H[i] < L[i - 2] and L[i - 2] - H[i] > 0.2 * a[i] and H[i + 1:].max(initial=-np.inf) < L[i - 2]:
            out.append(("bear", L[i - 2], H[i], i - 1))
    return out

def order_blocks(df, a):
    O, H, L, C, out = (df[c].values for c in ("Open", "High", "Low", "Close")), None, None, None, []
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    for i in range(1, len(df)):
        if np.isnan(a[i]): continue
        body = C[i] - O[i]
        if body > 1.5 * a[i] and C[i - 1] < O[i - 1] and C[i + 1:].min(initial=np.inf) > L[i - 1]:
            out.append(("bull", H[i - 1], L[i - 1], i - 1))
        if body < -1.5 * a[i] and C[i - 1] > O[i - 1] and C[i + 1:].max(initial=-np.inf) < H[i - 1]:
            out.append(("bear", H[i - 1], L[i - 1], i - 1))
    return out

def f(x): return None if x is None or pd.isna(x) else round(float(x), 5)

def run(symbol, interval):
    df = load(symbol, interval); daily = interval == "1d"
    a = atr(df).values; A = a[-1]; C = df.Close.values; H = df.High.values; L = df.Low.values
    price = C[-1]; ts = lambda i: int(df.index[i].timestamp())
    hi, lo = swings(df)
    if len(hi) < 2 or len(lo) < 2: raise ValueError("Structure insuffisante")

    # Structure
    trend = "range"
    if H[hi[-1]] > H[hi[-2]] and L[lo[-1]] > L[lo[-2]]: trend = "haussier"
    elif H[hi[-1]] < H[hi[-2]] and L[lo[-1]] < L[lo[-2]]: trend = "baissier"
    event = None
    if C[hi[-1] + 1:].max(initial=-np.inf) > H[hi[-1]]:
        event = "BOS haussier" if trend == "haussier" else "CHoCH haussier"
    elif C[lo[-1] + 1:].min(initial=np.inf) < L[lo[-1]]:
        event = "BOS baissier" if trend == "baissier" else "CHoCH baissier"

    # Premium / Discount
    rh, rl = H[hi[-1]], L[lo[-1]]; eq = (rh + rl) / 2
    zone = "premium" if price > eq else "discount"

    # Liquidité
    sweep = None
    for i in range(max(len(df) - 8, 0), len(df)):
        for h in hi[-6:]:
            if h < i - 1 and H[i] > H[h] and C[i] < H[h]: sweep = "bear"
        for l in lo[-6:]:
            if l < i - 1 and L[i] < L[l] and C[i] > L[l]: sweep = "bull"
    eqh = [float(H[x]) for x in hi[-6:] if any(abs(H[x] - H[y]) < 0.15 * A for y in hi[-6:] if y != x)]
    eql = [float(L[x]) for x in lo[-6:] if any(abs(L[x] - L[y]) < 0.15 * A for y in lo[-6:] if y != x)]

    # PD arrays
    arrays = [{"kind": k, "type": t, "top": f(t_), "bottom": f(b_), "time": ts(i)}
              for k, lst in (("FVG", fvgs(df, a)), ("OB", order_blocks(df, a)))
              for t, t_, b_, i in lst[-4:]]
    vp = profile(df.tail(300)); vw = vwap(df, daily)

    # Niveaux jour / semaine précédents
    d1 = df.resample("1D").agg({"High": "max", "Low": "min"}).dropna()
    w1 = df.resample("W").agg({"High": "max", "Low": "min"}).dropna()
    levels = {"PDH": f(d1.High.iloc[-2]) if len(d1) > 1 else None, "PDL": f(d1.Low.iloc[-2]) if len(d1) > 1 else None,
              "PWH": f(w1.High.iloc[-2]) if len(w1) > 1 else None, "PWL": f(w1.Low.iloc[-2]) if len(w1) > 1 else None,
              "POC": f(vp["poc"]), "VAH": f(vp["vah"]), "VAL": f(vp["val"]), "VWAP": f(vw.iloc[-1]), "EQ": f(eq)}

    # Fondamental
    fund = fundamentals.analyse(symbol); b = fund["bias_score"]

    # Score & règles
    score, why = b * 0.30, [f"Biais fondamental {fund['bias']} ({b:+.0f}) : {b*0.30:+.0f} pts"]
    def add(p, txt): 
        nonlocal score; score += p; why.append(f"{txt} : {p:+d} pts")
    if trend == "haussier": add(20, "Structure haussière (HH/HL)")
    elif trend == "baissier": add(-20, "Structure baissière (LH/LL)")
    if event: add(10 if "haussier" in event else -10, event)
    add(10 if zone == "discount" else -10, f"Prix en {zone}")
    if sweep == "bull": add(15, "Sweep de liquidité sell-side + rejet")
    elif sweep == "bear": add(-15, "Sweep de liquidité buy-side + rejet")
    add(5 if price > vw.iloc[-1] else -5, "Prix vs VWAP")
    add(5 if price > vp["poc"] else -5, "Prix vs POC")
    bull_z = [x for x in arrays if x["type"] == "bull" and x["top"] <= price + 0.3 * A and price - x["top"] < 1.5 * A]
    bear_z = [x for x in arrays if x["type"] == "bear" and x["bottom"] >= price - 0.3 * A and x["bottom"] - price < 1.5 * A]
    if bull_z: add(10, "PD Array haussier proche (FVG/OB)")
    if bear_z: add(-10, "PD Array baissier proche (FVG/OB)")
    score = round(score)
    direction = "LONG" if score >= 35 else "SHORT" if score <= -35 else "ATTENDRE"

    setup = None
    if direction != "ATTENDRE":
        long = direction == "LONG"; zs = bull_z if long else bear_z
        if zs:
            z = zs[-1]; entry = (z["top"] + z["bottom"]) / 2
            sl = z["bottom"] - 0.5 * A if long else z["top"] + 0.5 * A
        else:
            entry = price; sl = (L[lo[-1]] - 0.5 * A) if long else (H[hi[-1]] + 0.5 * A)
        risk = abs(entry - sl)
        targets = [H[h] for h in hi if H[h] > entry] if long else [L[l] for l in lo if L[l] < entry]
        tp1 = (min(targets) if long else max(targets)) if targets else (entry + 2 * risk if long else entry - 2 * risk)
        if abs(tp1 - entry) < 1.5 * risk: tp1 = entry + 2 * risk if long else entry - 2 * risk
        tp2 = entry + 3 * risk if long else entry - 3 * risk
        setup = {"side": direction, "entry": f(entry), "sl": f(sl), "tp1": f(tp1), "tp2": f(tp2),
                 "rr_tp1": round(abs(tp1 - entry) / risk, 2) if risk else None}

    h = dt.datetime.utcnow().hour
    kz = "London" if 7 <= h < 10 else "New York" if 12 <= h < 15 else "Hors killzone"
    n = 300; sub = df.tail(n); idx = range(len(df) - len(sub), len(df))
    return {"symbol": symbol, "interval": interval, "price": f(price), "atr": f(A),
            "fundamental": fund,
            "technical": {"trend": trend, "event": event, "zone": zone, "sweep": sweep,
                          "equal_highs": [f(x) for x in eqh], "equal_lows": [f(x) for x in eql], "killzone": kz},
            "levels": levels, "pd_arrays": arrays,
            "signal": {"direction": direction, "score": score, "reasons": why, "setup": setup},
            "playbook": PLAYBOOK,
            "candles": [[ts(i), f(df.Open.iloc[i]), f(H[i]), f(L[i]), f(C[i])] for i in idx]}
