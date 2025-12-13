import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Kalite Kontrol Risk Paneli", layout="wide")

st.title("Yapay Zekâ Destekli Kalite Kontrol – Risk Paneli (Pilot)")

st.caption("Excel'den LOT / TARIH / CEKME_DAYANIMI verisini okur; lot bazlı özet, risk skoru ve öneri üretir.")

# -------------------------
# Helpers
# -------------------------
REQUIRED_COLS = ["LOT", "TARIH", "CEKME_DAYANIMI"]

def load_data(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    # normalize columns
    df.columns = [str(c).strip().upper() for c in df.columns]
    # common Turkish/variant mappings
    col_map = {
        "ÇEKME_DAYANIMI": "CEKME_DAYANIMI",
        "CEKME DAYANIMI": "CEKME_DAYANIMI",
        "TARİH": "TARIH",
        "LOT NO": "LOT",
        "PARTI": "LOT",
        "PARTİ": "LOT",
    }
    df = df.rename(columns=col_map)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}. Kolonlar şu şekilde olmalı: {REQUIRED_COLS}")

    # parse date + numeric
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce", dayfirst=True)
    df["CEKME_DAYANIMI"] = pd.to_numeric(df["CEKME_DAYANIMI"], errors="coerce")

    df = df.dropna(subset=["LOT", "TARIH", "CEKME_DAYANIMI"]).copy()
    df["LOT"] = df["LOT"].astype(str).str.strip()

    return df.sort_values(["LOT", "TARIH"]).reset_index(drop=True)

def slope_days(d: pd.DataFrame) -> float:
    # slope of CEKME_DAYANIMI vs time in days (requires >=2 points)
    if len(d) < 2:
        return np.nan
    x = (d["TARIH"] - d["TARIH"].min()).dt.total_seconds() / 86400.0
    y = d["CEKME_DAYANIMI"].values
    if np.allclose(x.values, 0):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])

def build_lot_report(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("LOT", sort=False)

    rep = g["CEKME_DAYANIMI"].agg(
        N="count",
        ORT="mean",
        STD="std",
        MIN="min",
        MAX="max"
    ).reset_index()

    # date range
    dr = g["TARIH"].agg(BASLANGIC="min", BITIS="max").reset_index()
    rep = rep.merge(dr, on="LOT", how="left")

    # trend
    trends = []
    for lot, d in g:
        trends.append((lot, slope_days(d)))
    trend_df = pd.DataFrame(trends, columns=["LOT", "TREND_MPA_GUN"])
    rep = rep.merge(trend_df, on="LOT", how="left")

    # fill STD NaN for N=1 with 0
    rep["STD"] = rep["STD"].fillna(0.0)

    return rep

def score_row(r, ref_mean, ref_std, target_low):
    """
    Pilot risk score (0-100):
    - Level penalty: how much below target_low (MPa) -> up to 50
    - Variability penalty (STD) vs ref_std -> up to 25
    - Negative trend penalty (downward) -> up to 15
    - Low sample penalty (N=1) -> 10
    """
    # A) level
    level_gap = max(0.0, target_low - r["ORT"])
    level_score = min(50.0, (level_gap / max(1.0, ref_std)) * 15.0)  # scale with std

    # B) variability
    var_score = min(25.0, (r["STD"] / max(1e-6, ref_std)) * 10.0)

    # C) trend (negative only)
    tr = r["TREND_MPA_GUN"]
    trend_score = 0.0 if np.isnan(tr) else min(15.0, max(0.0, -tr) * 2.0)

    # D) sample penalty
    n_score = 10.0 if r["N"] == 1 else (5.0 if r["N"] == 2 else 0.0)

    score = level_score + var_score + trend_score + n_score
    return float(max(0.0, min(100.0, score)))

def label(score):
    if score >= 60:
        return "RİSKLİ"
    if score >= 30:
        return "İZLENMELİ"
    return "GÜVENLİ"

def suggestion(n, score, trend):
    if score >= 60:
        if n == 1:
            return "Tek ölçüm: yeniden ölçüm + proses kontrol önerilir."
        if not np.isnan(trend) and trend < -2:
            return "Düşüş trendi: proses/ısıl işlem parametrelerini kontrol et, yeniden ölçüm yap."
        return "Yeniden ölçüm + proses kontrol önerilir."
    if score >= 30:
        return "Takip önerilir (ek ölçüm planla, sapma artarsa aksiyon al)."
    return "Normal üretime devam."

# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("Veri Yükle")
    up = st.file_uploader("Excel (.xlsx) veya CSV", type=["xlsx", "csv"])
    st.divider()
    st.subheader("Pilot Ayarları")
    target_low = st.number_input("Alt hedef/eşik (MPa)", value=900, step=1)
    st.caption("Pilot için varsayılan 900 MPa. İstersen daha sonra ISO 898-1 sınıfına göre güncelleriz.")

if up is None:
    st.info("Başlamak için soldan Excel/CSV yükle. Kolonlar: LOT, TARIH, CEKME_DAYANIMI")
    st.stop()

try:
    df = load_data(up)
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

st.success(f"Yüklendi: {len(df)} satır, {df['LOT'].nunique()} LOT")

# Reference stats for scoring
ref_mean = float(df["CEKME_DAYANIMI"].mean())
ref_std = float(df["CEKME_DAYANIMI"].std(ddof=1) if len(df) > 1 else 1.0)

lot_rep = build_lot_report(df)
lot_rep["RISK_SKORU"] = lot_rep.apply(lambda r: score_row(r, ref_mean, ref_std, target_low), axis=1).round(0).astype(int)
lot_rep["DURUM"] = lot_rep["RISK_SKORU"].apply(label)
lot_rep["ONERI"] = lot_rep.apply(lambda r: suggestion(int(r["N"]), int(r["RISK_SKORU"]), float(r["TREND_MPA_GUN"])), axis=1)

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam LOT", int(lot_rep["LOT"].nunique()))
c2.metric("RİSKLİ", int((lot_rep["DURUM"]=="RİSKLİ").sum()))
c3.metric("İZLENMELİ", int((lot_rep["DURUM"]=="İZLENMELİ").sum()))
c4.metric("GÜVENLİ", int((lot_rep["DURUM"]=="GÜVENLİ").sum()))

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Riskli LOT Listesi")
    risk_only = st.toggle("Sadece RİSKLİ göster", value=True)
    view = lot_rep.copy()
    if risk_only:
        view = view[view["DURUM"]=="RİSKLİ"]
    view = view.sort_values(["RISK_SKORU","ORT"], ascending=[False, True])
    st.dataframe(
        view[["LOT","N","ORT","STD","MIN","MAX","TREND_MPA_GUN","RISK_SKORU","DURUM","ONERI"]],
        use_container_width=True,
        height=420
    )

    # Export buttons
    csv_bytes = lot_rep.to_csv(index=False).encode("utf-8-sig")
    st.download_button("LOT_RAPORU CSV indir", data=csv_bytes, file_name="lot_raporu.csv", mime="text/csv")

with right:
    st.subheader("LOT Detay")
    sel = st.selectbox("LOT seç", options=lot_rep["LOT"].tolist())
    d = df[df["LOT"]==sel].sort_values("TARIH")
    r = lot_rep[lot_rep["LOT"]==sel].iloc[0]

    st.write(f"**Durum:** {r['DURUM']}  |  **Risk Skoru:** {int(r['RISK_SKORU'])}/100")
    st.write(f"**Öneri:** {r['ONERI']}")

    st.caption("Ölçüm Zaman Serisi")
    st.line_chart(d.set_index("TARIH")["CEKME_DAYANIMI"])

    st.caption("Ham Ölçümler")
    st.dataframe(d, use_container_width=True, height=240)
