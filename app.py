import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
import warnings
import io
import re
import json

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataForge · Analyst & Cleaner",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
#  CSS — Clean Industrial Dark Theme
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root{
  --bg:#080c14;
  --s1:#0d1117;
  --s2:#161b27;
  --s3:#1e2535;
  --cyan:#22d3ee;
  --green:#4ade80;
  --amber:#fbbf24;
  --rose:#f87171;
  --violet:#a78bfa;
  --blue:#60a5fa;
  --text:#f0f4f8;
  --muted:#64748b;
  --border:rgba(34,211,238,0.12);
  --border2:rgba(255,255,255,0.06);
}

*{box-sizing:border-box;}
html,body,[class*="css"]{
  font-family:'Syne',sans-serif;
  color:var(--text);
  background:var(--bg)!important;
}
.stApp{background:var(--bg)!important;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:var(--s1)!important;
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label{
  font-family:'JetBrains Mono',monospace!important;
  font-size:0.78rem!important;
}

/* ── Typography ── */
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;letter-spacing:-0.02em;}

/* ── Phase Nav Cards ── */
.phase-card{
  display:flex;align-items:center;gap:12px;
  background:var(--s2);border:1px solid var(--border2);
  border-radius:10px;padding:12px 16px;margin-bottom:8px;
  cursor:pointer;transition:border-color 0.2s;
}
.phase-card.active{border-color:var(--cyan);background:var(--s3);}
.phase-num{
  width:28px;height:28px;border-radius:50%;
  background:var(--s3);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--cyan);
  flex-shrink:0;
}
.phase-label{font-size:0.82rem;font-weight:600;color:var(--text);}
.phase-sub{font-size:0.7rem;color:var(--muted);}

/* ── Section Headers ── */
.sec-head{
  display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--border);
  padding-bottom:10px;margin:24px 0 16px;
}
.sec-icon{font-size:1.1rem;}
.sec-title{
  font-family:'JetBrains Mono',monospace;
  font-size:0.7rem;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--cyan);
}

/* ── Metric Grid ── */
.mgrid{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 20px;}
.mbox{
  flex:1 1 120px;background:var(--s2);
  border:1px solid var(--border2);border-radius:10px;
  padding:14px 16px;text-align:center;
  border-top:2px solid var(--cyan);
}
.mbox.green{border-top-color:var(--green);}
.mbox.amber{border-top-color:var(--amber);}
.mbox.rose {border-top-color:var(--rose);}
.mbox.violet{border-top-color:var(--violet);}
.mval{font-family:'JetBrains Mono',monospace;font-size:1.55rem;color:var(--cyan);font-weight:700;line-height:1;}
.mval.green{color:var(--green);}
.mval.amber{color:var(--amber);}
.mval.rose {color:var(--rose);}
.mlbl{font-size:0.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-top:5px;}

/* ── Info / Alert Boxes ── */
.ibox{border-radius:8px;padding:12px 16px;margin:8px 0;font-size:0.85rem;border-left:3px solid;}
.ibox.info  {background:rgba(34,211,238,0.08);border-color:var(--cyan);}
.ibox.good  {background:rgba(74,222,128,0.08);border-color:var(--green);}
.ibox.warn  {background:rgba(251,191,36,0.08);border-color:var(--amber);}
.ibox.bad   {background:rgba(248,113,113,0.08);border-color:var(--rose);}

/* ── Badge ── */
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.68rem;font-weight:700;margin:2px;}
.badge.cyan  {background:rgba(34,211,238,0.12);color:var(--cyan);}
.badge.green {background:rgba(74,222,128,0.12);color:var(--green);}
.badge.amber {background:rgba(251,191,36,0.12);color:var(--amber);}
.badge.rose  {background:rgba(248,113,113,0.12);color:var(--rose);}

/* ── Tables ── */
.dataframe{background:var(--s2)!important;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--s2)!important;border-radius:10px;padding:4px;gap:4px;
}
.stTabs [data-baseweb="tab"]{
  font-family:'JetBrains Mono',monospace!important;font-size:0.73rem!important;
  color:var(--muted)!important;background:transparent!important;border-radius:8px!important;
}
.stTabs [aria-selected="true"]{color:var(--cyan)!important;background:var(--s3)!important;}

/* ── Buttons ── */
.stButton>button{
  background:linear-gradient(135deg,#0e7490,#1d4ed8)!important;
  color:white!important;border:none!important;border-radius:8px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:0.78rem!important;
  letter-spacing:0.04em!important;padding:8px 20px!important;
}
.stButton>button:hover{opacity:0.88!important;}

/* ── Expander ── */
details summary{
  font-family:'JetBrains Mono',monospace!important;font-size:0.78rem!important;
  color:var(--cyan)!important;
}

/* ── Progress ── */
.stProgress>div>div{background:var(--cyan)!important;}

/* ── Step card ── */
.step-card{
  background:var(--s2);border:1px solid var(--border2);
  border-left:3px solid var(--green);
  border-radius:10px;padding:14px 18px;margin:8px 0;
}
.step-title{font-weight:700;font-size:0.9rem;margin-bottom:4px;}
.step-desc{font-size:0.8rem;color:var(--muted);}

/* ── Download btn ── */
.stDownloadButton>button{
  background:linear-gradient(135deg,#064e3b,#065f46)!important;
  color:white!important;border:none!important;border-radius:8px!important;
  font-family:'JetBrains Mono',monospace!important;font-size:0.78rem!important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  MATPLOTLIB THEME
# ════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b27",
    "axes.edgecolor":"#1e2535","axes.labelcolor":"#94a3b8",
    "xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#f0f4f8","grid.color":"#1e2535",
    "grid.linestyle":"--","grid.alpha":0.5,
    "figure.dpi":130,"font.family":"monospace","font.size":8,
})
PAL = ["#22d3ee","#a78bfa","#4ade80","#fbbf24","#f87171",
       "#60a5fa","#34d399","#fb923c","#e879f9","#94a3b8"]


def fig_show(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def sec(icon, title):
    st.markdown(f'<div class="sec-head"><span class="sec-icon">{icon}</span>'
                f'<span class="sec-title">{title}</span></div>', unsafe_allow_html=True)


def ibox(msg, kind="info"):
    st.markdown(f'<div class="ibox {kind}">{msg}</div>', unsafe_allow_html=True)


def mbox(val, lbl, color=""):
    return f'<div class="mbox {color}"><div class="mval {color}">{val}</div><div class="mlbl">{lbl}</div></div>'


def detect_types(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    dt  = df.select_dtypes(include=["datetime"]).columns.tolist()
    return num, cat, dt


# ════════════════════════════════════════════════════════
#  PHASE 1 — RAW ANALYSIS
# ════════════════════════════════════════════════════════

def phase_analysis(df):
    num, cat, dt = detect_types(df)
    miss = df.isnull().sum()
    miss_pct = miss / len(df) * 100
    dup = df.duplicated().sum()

    # ── Health Score ──
    health = 100
    if miss_pct.mean() > 20: health -= 25
    elif miss_pct.mean() > 5: health -= 10
    if dup / len(df) > 0.05: health -= 15
    health = max(0, health)
    health_color = "green" if health >= 80 else "amber" if health >= 50 else "rose"

    sec("🔍","Dataset Health Report")
    st.markdown(f"""
    <div class="mgrid">
      {mbox(f"{df.shape[0]:,}","Rows")}
      {mbox(f"{df.shape[1]}","Columns")}
      {mbox(f"{len(num)}","Numeric","green")}
      {mbox(f"{len(cat)}","Categorical","violet")}
      {mbox(f"{miss_pct.mean():.1f}%","Avg Missing","amber" if miss_pct.mean()>5 else "green")}
      {mbox(f"{dup:,}","Duplicates","rose" if dup>0 else "green")}
      {mbox(f"{df.memory_usage(deep=True).sum()/1e6:.2f} MB","Memory")}
      {mbox(f"{health}","Health Score",health_color)}
    </div>
    """, unsafe_allow_html=True)

    # Health bar
    color_map = {"green":"#4ade80","amber":"#fbbf24","rose":"#f87171"}
    st.markdown(f"**Overall Data Health: {health}/100**")
    st.progress(health/100)

    # ── Column Info Table ──
    sec("📋","Column-by-Column Summary")
    rows = []
    for col in df.columns:
        ser = df[col]
        is_num = col in num
        rows.append({
            "Column": col,
            "Type": str(ser.dtype),
            "Missing": int(ser.isnull().sum()),
            "Missing %": round(ser.isnull().mean()*100, 1),
            "Unique": int(ser.nunique()),
            "Zeros": int((ser==0).sum()) if is_num else "—",
            "Mean/Mode": round(ser.mean(),2) if is_num else str(ser.mode()[0]) if ser.notna().any() else "—",
            "Std": round(ser.std(),2) if is_num else "—",
            "Min": round(ser.min(),2) if is_num else "—",
            "Max": round(ser.max(),2) if is_num else "—",
        })
    info_df = pd.DataFrame(rows)

    def color_m(v):
        if isinstance(v, (int,float)):
            if v > 30: return "background-color:rgba(248,113,113,0.2)"
            if v > 10: return "background-color:rgba(251,191,36,0.2)"
        return ""

    st.dataframe(info_df.style.applymap(color_m, subset=["Missing %"]),
                 use_container_width=True, height=280)

    # ── Missing Values ──
    sec("🕳️","Missing Values Deep Dive")
    if miss[miss > 0].empty:
        ibox("✅ No missing values detected!", "good")
    else:
        mc1, mc2 = st.columns([3,2])
        with mc1:
            m_data = miss[miss > 0].sort_values(ascending=False)
            pct_data = miss_pct[miss_pct > 0].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, max(3, len(m_data)*0.45)))
            colors = ["#f87171" if p>30 else "#fbbf24" if p>10 else "#22d3ee" for p in pct_data]
            bars = ax.barh(m_data.index, pct_data.values, color=colors, height=0.6, edgecolor="none")
            for bar, pv in zip(bars, pct_data.values):
                ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        f"{pv:.1f}%", va="center", fontsize=8)
            ax.axvline(30, color="#f87171", linestyle="--", alpha=0.5, lw=1)
            ax.axvline(10, color="#fbbf24", linestyle="--", alpha=0.5, lw=1)
            ax.set_xlabel("Missing %"); ax.set_title("Missing % per Column", fontsize=10)
            fig.tight_layout(); fig_show(fig)
        with mc2:
            fig, ax = plt.subplots(figsize=(4,4))
            subset = df[m_data.index].head(100)
            ax.imshow(subset.isnull().astype(int).T, aspect="auto",
                      cmap="RdYlGn_r", interpolation="nearest")
            ax.set_yticks(range(len(m_data.index)))
            ax.set_yticklabels(m_data.index, fontsize=7)
            ax.set_xlabel("Rows (first 100)", fontsize=8)
            ax.set_title("Missing Pattern", fontsize=9)
            fig.tight_layout(); fig_show(fig)

    # ── Duplicates ──
    sec("♊","Duplicate Rows")
    if dup == 0:
        ibox("✅ No duplicate rows found.", "good")
    else:
        ibox(f"⚠️ Found <b>{dup} duplicate rows</b> ({dup/len(df)*100:.1f}% of data). These will be removed during cleaning.", "warn")
        with st.expander("View duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)

    # ── Numeric Analysis ──
    if num:
        sec("📊","Numeric Features Analysis")
        desc = df[num].describe().T
        desc["skewness"] = df[num].skew()
        desc["kurtosis"] = df[num].kurt()
        desc["outliers_iqr"] = [
            int(((df[c] < df[c].quantile(0.25)-1.5*(df[c].quantile(0.75)-df[c].quantile(0.25))) |
                 (df[c] > df[c].quantile(0.75)+1.5*(df[c].quantile(0.75)-df[c].quantile(0.25)))).sum())
            for c in num
        ]
        st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["mean","std"])
                               .background_gradient(cmap="Reds", subset=["outliers_iqr"]),
                     use_container_width=True)

        # Distributions
        st.markdown("**📈 Distributions + KDE**")
        per_row = 3
        for i in range(0, len(num), per_row):
            batch = num[i:i+per_row]
            fig, axes = plt.subplots(1, len(batch), figsize=(5*len(batch), 4))
            if len(batch)==1: axes=[axes]
            for ax, col in zip(axes, batch):
                data = df[col].dropna()
                c = PAL[num.index(col)%len(PAL)]
                ax.hist(data, bins=30, color=c, edgecolor="none", alpha=0.75)
                try:
                    ax2 = ax.twinx()
                    kx = np.linspace(data.min(), data.max(), 200)
                    ax2.plot(kx, stats.gaussian_kde(data)(kx), color="white", lw=1.5, alpha=0.8)
                    ax2.set_yticks([])
                except: pass
                ax.set_title(col, fontsize=9, pad=6)
                sk = skew(data)
                flag = " ⚠️" if abs(sk)>1 else ""
                ax.text(0.97,0.95,f"skew={sk:.2f}{flag}", transform=ax.transAxes,
                        ha="right",va="top",fontsize=7,color="#94a3b8")
            fig.tight_layout(); fig_show(fig)

        # Outlier Box Plots
        st.markdown("**📦 Box Plots — Outlier View**")
        fig, axes = plt.subplots(1, len(num), figsize=(max(12, 3*len(num)), 4))
        if len(num)==1: axes=[axes]
        for ax, col in zip(axes, num):
            c = PAL[num.index(col)%len(PAL)]
            ax.boxplot(df[col].dropna(), patch_artist=True, widths=0.55,
                       boxprops=dict(facecolor=c, alpha=0.6),
                       medianprops=dict(color="white", linewidth=2.5),
                       whiskerprops=dict(color="#64748b"),
                       capprops=dict(color="#64748b"),
                       flierprops=dict(marker="o", color="#f87171", markersize=3, alpha=0.6))
            ax.set_title(col, fontsize=8); ax.set_xticklabels([])
        fig.tight_layout(); fig_show(fig)

        # Outlier detail
        st.markdown("**🎯 Outlier Summary (IQR Method)**")
        out_rows = []
        for col in num:
            d = df[col].dropna()
            Q1,Q3 = d.quantile(0.25), d.quantile(0.75)
            IQR = Q3-Q1
            lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
            n_out = ((d<lo)|(d>hi)).sum()
            sev = "🔴 High" if n_out/len(d)>0.1 else "🟡 Medium" if n_out/len(d)>0.03 else "🟢 Low"
            out_rows.append({"Column":col,"Q1":round(Q1,2),"Q3":round(Q3,2),
                             "IQR":round(IQR,2),"Lower Fence":round(lo,2),
                             "Upper Fence":round(hi,2),"Outliers":n_out,
                             "Outlier %":round(n_out/len(d)*100,2),"Severity":sev})
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

        # Normality test
        st.markdown("**🧪 Normality Tests (Shapiro-Wilk, n≤5000)**")
        norm_rows = []
        for col in num:
            d = df[col].dropna()
            try:
                stat, p = shapiro(d.sample(min(len(d),5000), random_state=42))
                normal = "✅ Normal" if p>0.05 else "❌ Not Normal"
                norm_rows.append({"Column":col,"W-stat":round(stat,4),
                                  "p-value":round(p,4),"Verdict":normal})
            except: pass
        if norm_rows:
            st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)

    # ── Categorical Analysis ──
    if cat:
        sec("🏷️","Categorical Features Analysis")
        for col in cat:
            vc = df[col].value_counts().head(15)
            with st.expander(f"📌  {col}  —  {df[col].nunique()} unique  |  {df[col].isnull().sum()} missing"):
                c1,c2 = st.columns([3,2])
                with c1:
                    fig, ax = plt.subplots(figsize=(7, max(2.5, len(vc)*0.38)))
                    colors_b = [PAL[i%len(PAL)] for i in range(len(vc))]
                    ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                            color=colors_b[::-1], edgecolor="none", height=0.65)
                    for i,(v,c) in enumerate(zip(vc.index.astype(str)[::-1], vc.values[::-1])):
                        ax.text(c+max(vc.values)*0.01, i, f"{c:,}", va="center", fontsize=7)
                    ax.set_xlabel("Count"); ax.set_title(f"Value Counts: {col}", fontsize=9)
                    fig.tight_layout(); fig_show(fig)
                with c2:
                    top = min(7, len(vc))
                    fig, ax = plt.subplots(figsize=(4,4))
                    ax.pie(vc.values[:top], labels=vc.index.astype(str)[:top],
                           autopct="%1.1f%%", colors=PAL[:top],
                           startangle=140, pctdistance=0.75,
                           wedgeprops=dict(edgecolor="#0d1117", linewidth=1.5))
                    ax.set_title("Top Share", fontsize=9)
                    fig.tight_layout(); fig_show(fig)
                # High cardinality warning
                if df[col].nunique() > 50:
                    ibox(f"⚠️ High cardinality ({df[col].nunique()} unique values). Consider grouping or target encoding.", "warn")

    # ── Correlation ──
    if len(num) >= 2:
        sec("🔗","Correlation Matrix")
        corr = df[num].corr()
        c1,c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8,6))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220,10,as_cmap=True),
                        center=0, annot=len(num)<=12, fmt=".2f",
                        linewidths=0.5, linecolor="#080c14", ax=ax, annot_kws={"size":7})
            ax.set_title("Correlation Heatmap", fontsize=10); fig.tight_layout(); fig_show(fig)
        with c2:
            mask2 = np.triu(np.ones_like(corr, dtype=bool))
            pairs = (corr.where(~mask2).stack()
                     .reset_index().rename(columns={"level_0":"A","level_1":"B",0:"r"}))
            pairs["abs"] = pairs["r"].abs()
            pairs = pairs.sort_values("abs", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.barh(range(len(pairs)), pairs["r"].values,
                    color=["#4ade80" if v>0 else "#f87171" for v in pairs["r"]], height=0.65)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{r.A} × {r.B}" for r in pairs.itertuples()], fontsize=7)
            ax.axvline(0, color="#64748b"); ax.set_title("Top Correlated Pairs", fontsize=10)
            fig.tight_layout(); fig_show(fig)

        # Highly correlated feature warning
        high_corr = pairs[pairs["abs"] > 0.85]
        if not high_corr.empty:
            ibox(f"⚠️ <b>{len(high_corr)} highly correlated pairs</b> (|r|>0.85) detected — may cause multicollinearity.", "warn")

    sec("📝","Analysis Summary & Recommendations")
    issues = []
    if miss_pct.mean() > 0:   issues.append(("🕳️","Missing Values",f"Average {miss_pct.mean():.1f}% missing across columns","Handle with imputation"))
    if dup > 0:                issues.append(("♊","Duplicates",f"{dup} duplicate rows ({dup/len(df)*100:.1f}%)","Remove all duplicates"))
    if num:
        high_skew = [c for c in num if abs(skew(df[c].dropna()))>1]
        if high_skew: issues.append(("📈","Skewed Features",f"{', '.join(high_skew)}","Apply log/sqrt transform"))
    if cat:
        high_card = [c for c in cat if df[c].nunique()>20]
        if high_card: issues.append(("🏷️","High Cardinality",f"{', '.join(high_card)}","Target/frequency encoding recommended"))

    if not issues:
        ibox("✅ Data looks relatively clean! Proceed to the cleaning phase.", "good")
    else:
        for icon, title, detail, fix in issues:
            st.markdown(f"""
            <div class="step-card">
              <div class="step-title">{icon} {title}</div>
              <div class="step-desc"><b>Issue:</b> {detail}<br><b>Fix:</b> {fix}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PHASE 2 — CLEANING
# ════════════════════════════════════════════════════════

def phase_cleaning(df, opts):
    report = []
    cleaned = df.copy()

    # 1. Fix column names
    cleaned.columns = [re.sub(r"[^a-zA-Z0-9_]","_",c).strip("_").lower() for c in cleaned.columns]
    report.append(("🔤","Column Names Normalized","All names lowercased, special chars replaced","done"))

    # 2. Replace placeholder nulls
    placeholders = ["NA","N/A","na","n/a","NULL","null","None","none",
                    "NaN","nan","-","--","?","undefined","missing","MISSING",""]
    cleaned.replace(placeholders, np.nan, inplace=True)
    report.append(("🔄","Placeholder Values","Replaced NA/NULL/? etc. with NaN","done"))

    # 3. Strip whitespace
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].str.strip()
    report.append(("✂️","Whitespace Stripped","Leading/trailing spaces removed from text","done"))

    # 4. Type coercion
    coerced = 0
    for col in cleaned.select_dtypes(include="object").columns:
        c = pd.to_numeric(cleaned[col], errors="coerce")
        if c.notna().sum() / max(cleaned[col].notna().sum(),1) > 0.8:
            cleaned[col] = c; coerced += 1
    if coerced:
        report.append(("🔢",f"Type Coercion",f"{coerced} columns converted to numeric","done"))

    # 5. Datetime parsing
    for col in cleaned.select_dtypes(include="object").columns:
        try:
            p = pd.to_datetime(cleaned[col], errors="coerce", infer_datetime_format=True)
            if p.notna().sum() / max(cleaned[col].notna().sum(),1) > 0.7:
                cleaned[col] = p
        except: pass

    # 6. Remove duplicates
    n_dup = cleaned.duplicated().sum()
    if n_dup:
        cleaned = cleaned.drop_duplicates()
        report.append(("♊","Duplicates Removed",f"Removed {n_dup} duplicate rows","done"))

    # 7. Drop high-missing columns
    num_c, cat_c, _ = detect_types(cleaned)
    thresh = opts["drop_col_thresh"] / 100
    high_miss_cols = [c for c in cleaned.columns if cleaned[c].isnull().mean() > thresh]
    if high_miss_cols and opts["drop_high_miss_cols"]:
        cleaned.drop(columns=high_miss_cols, inplace=True)
        report.append(("🗑️","High-Missing Cols Dropped",
                       f"Dropped {len(high_miss_cols)} cols with >{thresh*100:.0f}% missing: {high_miss_cols}","done"))

    # 8. Drop high-missing rows
    num_c2, cat_c2, _ = detect_types(cleaned)
    row_thresh = opts["drop_row_thresh"] / 100
    if opts["drop_high_miss_rows"]:
        n_before = len(cleaned)
        cleaned = cleaned[cleaned.isnull().mean(axis=1) <= row_thresh]
        n_dropped = n_before - len(cleaned)
        if n_dropped:
            report.append(("🗑️","High-Missing Rows Dropped",
                           f"Dropped {n_dropped} rows with >{row_thresh*100:.0f}% missing","done"))

    # 9. Impute missing values
    num_c, cat_c, _ = detect_types(cleaned)
    impute_method = opts["impute_method"]
    if impute_method == "Median":
        for col in num_c:
            if cleaned[col].isnull().any():
                cleaned[col].fillna(cleaned[col].median(), inplace=True)
        report.append(("🩹","Numeric Imputation","Missing numerics filled with median","done"))
    elif impute_method == "Mean":
        for col in num_c:
            if cleaned[col].isnull().any():
                cleaned[col].fillna(cleaned[col].mean(), inplace=True)
        report.append(("🩹","Numeric Imputation","Missing numerics filled with mean","done"))
    elif impute_method == "KNN":
        miss_num = [c for c in num_c if cleaned[c].isnull().any()]
        if miss_num:
            imp = KNNImputer(n_neighbors=5)
            cleaned[miss_num] = imp.fit_transform(cleaned[miss_num])
            report.append(("🩹","KNN Imputation",f"KNN(k=5) applied to {len(miss_num)} numeric cols","done"))

    for col in cat_c:
        if cleaned[col].isnull().any():
            m = cleaned[col].mode()
            if len(m): cleaned[col].fillna(m[0], inplace=True)
    if cat_c:
        report.append(("🩹","Categorical Imputation","Missing categoricals filled with mode","done"))

    # 10. Outlier handling
    num_c, _, _ = detect_types(cleaned)
    out_method = opts["outlier_method"]
    if out_method == "IQR Cap (1.5×)":
        n_cap = 0
        for col in num_c:
            Q1,Q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            IQR = Q3-Q1
            lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
            n_cap += ((cleaned[col]<lo)|(cleaned[col]>hi)).sum()
            cleaned[col] = cleaned[col].clip(lo,hi)
        report.append(("📐","Outliers Capped (IQR 1.5×)",f"Capped {n_cap} values","done"))
    elif out_method == "IQR Cap (3×)":
        n_cap = 0
        for col in num_c:
            Q1,Q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            IQR = Q3-Q1
            lo,hi = Q1-3*IQR, Q3+3*IQR
            n_cap += ((cleaned[col]<lo)|(cleaned[col]>hi)).sum()
            cleaned[col] = cleaned[col].clip(lo,hi)
        report.append(("📐","Outliers Capped (IQR 3×)",f"Capped {n_cap} values","done"))
    elif out_method == "Z-score Remove (>3σ)":
        n_before = len(cleaned)
        z_mask = pd.Series([True]*len(cleaned), index=cleaned.index)
        for col in num_c:
            z = np.abs(stats.zscore(cleaned[col].dropna()))
            bad_idx = cleaned[col].dropna().index[z>3]
            z_mask[bad_idx] = False
        cleaned = cleaned[z_mask]
        report.append(("📐","Z-score Outliers Removed",f"Removed {n_before-len(cleaned)} rows","done"))
    elif out_method == "Winsorize (5%-95%)":
        for col in num_c:
            lo,hi = cleaned[col].quantile(0.05), cleaned[col].quantile(0.95)
            cleaned[col] = cleaned[col].clip(lo,hi)
        report.append(("📐","Winsorized (5%-95%)","Clipped extremes at 5th/95th percentile","done"))

    return cleaned, report


# ════════════════════════════════════════════════════════
#  PHASE 3 — MODEL READY
# ════════════════════════════════════════════════════════

def phase_model_ready(df, opts, target_col=None):
    steps = []
    result = df.copy()
    num_c, cat_c, dt_c = detect_types(result)

    # 1. Drop datetime columns (not usable directly)
    if dt_c:
        # Extract useful features from datetime
        for col in dt_c:
            try:
                result[f"{col}_year"]  = result[col].dt.year
                result[f"{col}_month"] = result[col].dt.month
                result[f"{col}_day"]   = result[col].dt.day
                result[f"{col}_dow"]   = result[col].dt.dayofweek
            except: pass
            result.drop(columns=[col], inplace=True)
        steps.append(("📅","Datetime Features Extracted",
                      f"Extracted year/month/day/dow from {dt_c}","done"))

    num_c, cat_c, _ = detect_types(result)

    # 2. Skewness correction
    if opts["fix_skewness"]:
        fixed = []
        for col in num_c:
            if col == target_col: continue
            d = result[col].dropna()
            sk = skew(d)
            if sk > 1 and (d > 0).all():
                result[col] = np.log1p(result[col])
                fixed.append(f"{col}(log1p)")
            elif abs(sk) > 1:
                result[col] = np.cbrt(result[col])
                fixed.append(f"{col}(cbrt)")
        if fixed:
            steps.append(("📈","Skewness Corrected",
                          f"Applied log1p/cbrt to: {', '.join(fixed)}","done"))

    # 3. Encode categoricals
    enc_method = opts["encoding"]
    le_map = {}
    if enc_method == "Label Encoding":
        for col in cat_c:
            if col == target_col: continue
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))
            le_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        steps.append(("🏷️","Label Encoding",
                      f"Applied to {len(cat_c)} categorical columns","done"))
    elif enc_method == "One-Hot Encoding":
        before = result.shape[1]
        to_encode = [c for c in cat_c if c != target_col]
        result = pd.get_dummies(result, columns=to_encode, drop_first=False)
        added = result.shape[1] - before
        steps.append(("🏷️","One-Hot Encoding",
                      f"Created {added} new binary columns","done"))
    elif enc_method == "Frequency Encoding":
        for col in cat_c:
            if col == target_col: continue
            freq = result[col].value_counts(normalize=True)
            result[col] = result[col].map(freq)
        steps.append(("🏷️","Frequency Encoding",
                      f"Replaced categories with their frequencies","done"))

    # 4. Encode target if categorical
    if target_col and target_col in result.columns:
        if result[target_col].dtype == object or str(result[target_col].dtype) in ["category","bool"]:
            le = LabelEncoder()
            result[target_col] = le.fit_transform(result[target_col].astype(str))
            steps.append(("🎯","Target Encoded",
                          f"'{target_col}' label-encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}","done"))

    # 5. Scaling
    num_c2, _, _ = detect_types(result)
    scale_cols = [c for c in num_c2 if c != target_col]
    scale_method = opts["scaling"]
    if scale_method == "StandardScaler (Z-score)":
        sc = StandardScaler()
        result[scale_cols] = sc.fit_transform(result[scale_cols])
        steps.append(("📏","StandardScaler Applied",
                      f"Mean=0, Std=1 for {len(scale_cols)} columns","done"))
    elif scale_method == "MinMaxScaler [0,1]":
        sc = MinMaxScaler()
        result[scale_cols] = sc.fit_transform(result[scale_cols])
        steps.append(("📏","MinMaxScaler Applied",
                      f"Range [0,1] for {len(scale_cols)} columns","done"))
    elif scale_method == "RobustScaler (IQR)":
        sc = RobustScaler()
        result[scale_cols] = sc.fit_transform(result[scale_cols])
        steps.append(("📏","RobustScaler Applied",
                      f"IQR-based scaling for {len(scale_cols)} columns","done"))

    # 6. Remove low-variance features
    if opts["remove_low_var"]:
        num_c3, _, _ = detect_types(result)
        low_var = [c for c in num_c3 if c != target_col and result[c].std() < 0.01]
        if low_var:
            result.drop(columns=low_var, inplace=True)
            steps.append(("🧹","Low-Variance Dropped",
                          f"Removed {len(low_var)} near-zero variance columns: {low_var}","done"))

    # 7. Remove highly correlated features
    if opts["remove_high_corr"]:
        num_c4, _, _ = detect_types(result)
        feat_cols = [c for c in num_c4 if c != target_col]
        if len(feat_cols) > 1:
            corr_m = result[feat_cols].corr().abs()
            upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
            if to_drop:
                result.drop(columns=to_drop, inplace=True)
                steps.append(("🔗","High-Corr Features Dropped",
                              f"Removed {len(to_drop)} cols with |r|>0.95: {to_drop}","done"))

    # 8. Final NaN cleanup
    remaining_null = result.isnull().sum().sum()
    if remaining_null:
        result = result.fillna(0)
        steps.append(("🩹","Remaining NaNs → 0",
                      f"Filled {remaining_null} remaining nulls with 0","done"))

    steps.append(("✅","Final Shape",f"Model-ready dataset: {result.shape[0]} rows × {result.shape[1]} columns","done"))

    return result, steps


# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 12px'>
      <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#22d3ee'>⚗️ DataForge</div>
      <div style='font-size:0.65rem;color:#475569;letter-spacing:0.18em;text-transform:uppercase;margin-top:3px'>Analyst & Model Prep</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("📂 Upload Dataset", type=["csv","xlsx","xls","json"],
                                 help="CSV, Excel or JSON")
    if uploaded:
        st.success(f"✅ {uploaded.name}")

    st.markdown("---")
    st.markdown("**🎯 Target Column**")
    target_inp = st.text_input("", placeholder="e.g. price, survived, label", label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**📋 Navigation**")
    phase = st.radio("", ["🔍 Phase 1 · Analysis", "🧹 Phase 2 · Cleaning", "🚀 Phase 3 · Model Ready"],
                     label_visibility="collapsed")

    if "Cleaning" in phase or "Model" in phase:
        st.markdown("---")
        st.markdown("**⚙️ Cleaning Options**")
        imp_method  = st.selectbox("Imputation",["Median","Mean","KNN","None"])
        out_method  = st.selectbox("Outlier Handling",
                                   ["IQR Cap (1.5×)","IQR Cap (3×)","Z-score Remove (>3σ)","Winsorize (5%-95%)","None"])
        drop_cols   = st.checkbox("Drop cols with >X% missing", value=True)
        drop_thresh = st.slider("Col missing threshold %", 30, 90, 60) if drop_cols else 60
        drop_rows   = st.checkbox("Drop rows with >Y% missing", value=False)
        row_thresh  = st.slider("Row missing threshold %", 30, 90, 50) if drop_rows else 50

    if "Model" in phase:
        st.markdown("---")
        st.markdown("**🚀 Model Prep Options**")
        encoding   = st.selectbox("Encoding",["Label Encoding","One-Hot Encoding","Frequency Encoding"])
        scaling    = st.selectbox("Scaling",["StandardScaler (Z-score)","MinMaxScaler [0,1]","RobustScaler (IQR)","None"])
        fix_skew   = st.checkbox("Fix Skewness (log/cbrt)", value=True)
        rem_lowvar = st.checkbox("Remove Low-Variance Features", value=True)
        rem_hicorr = st.checkbox("Remove Highly Correlated (>0.95)", value=True)

    st.markdown("---")
    st.caption("DataForge · Data Science Pipeline")


# ════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:24px 0 8px'>
  <h1 style='font-size:2.1rem;color:#f0f4f8;margin:0;font-weight:800'>
    ⚗️ DataForge
    <span style='color:#22d3ee;font-size:1.2rem;font-weight:400;margin-left:10px'>· Analyst & Model Prep Pipeline</span>
  </h1>
  <p style='color:#64748b;font-size:0.88rem;margin-top:6px'>
    Upload dirty data → Full analysis → Smart cleaning → Model-ready dataset
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  PIPELINE PROGRESS INDICATOR
# ════════════════════════════════════════════════════════
p1_active = "active" if "Analysis" in phase else ""
p2_active = "active" if "Cleaning" in phase else ""
p3_active = "active" if "Model"    in phase else ""

st.markdown(f"""
<div style='display:flex;gap:10px;margin:12px 0 24px;flex-wrap:wrap'>
  <div class='phase-card {p1_active}' style='flex:1;min-width:150px'>
    <div class='phase-num'>01</div>
    <div><div class='phase-label'>🔍 Analysis</div><div class='phase-sub'>Explore dirty data</div></div>
  </div>
  <div style='display:flex;align-items:center;color:#1e2535;font-size:1.5rem'>›</div>
  <div class='phase-card {p2_active}' style='flex:1;min-width:150px'>
    <div class='phase-num'>02</div>
    <div><div class='phase-label'>🧹 Cleaning</div><div class='phase-sub'>Fix & handle issues</div></div>
  </div>
  <div style='display:flex;align-items:center;color:#1e2535;font-size:1.5rem'>›</div>
  <div class='phase-card {p3_active}' style='flex:1;min-width:150px'>
    <div class='phase-num'>03</div>
    <div><div class='phase-label'>🚀 Model Ready</div><div class='phase-sub'>Encode, scale, finalize</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  NO FILE STATE
# ════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div style='background:#0d1117;border:1px dashed rgba(34,211,238,0.25);border-radius:16px;
                padding:60px 40px;text-align:center;margin-top:20px'>
      <div style='font-size:3.5rem;margin-bottom:12px'>📂</div>
      <h3 style='font-family:Syne,sans-serif;color:#f0f4f8;margin:0 0 8px'>Upload Your Dataset</h3>
      <p style='color:#64748b;max-width:420px;margin:auto;font-size:0.88rem'>
        Upload a <b>CSV</b>, <b>Excel</b>, or <b>JSON</b> file from the sidebar to begin the 3-phase pipeline.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    cards = [
        ("🔍","Phase 1 · Analysis","Deep-dive into your raw/dirty data — missing values, outliers, distributions, correlations, data types","cyan"),
        ("🧹","Phase 2 · Cleaning","Smart cleaning — remove duplicates, impute missing values, cap outliers, fix types, normalize names","green"),
        ("🚀","Phase 3 · Model Ready","Encode categoricals, scale features, fix skewness, remove low-variance & correlated columns","amber"),
    ]
    for col, (icon, title, desc, color) in zip([c1,c2,c3], cards):
        with col:
            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid rgba(255,255,255,0.07);
                        border-top:2px solid var(--{color},#22d3ee);
                        border-radius:12px;padding:20px 18px;height:170px'>
              <div style='font-size:1.5rem'>{icon}</div>
              <div style='font-weight:700;margin:8px 0 5px;font-size:0.9rem'>{title}</div>
              <div style='color:#64748b;font-size:0.78rem'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════
@st.cache_data
def load_file(file):
    name = file.name
    if name.endswith(".csv"):           return pd.read_csv(file)
    elif name.endswith((".xlsx",".xls")):return pd.read_excel(file)
    elif name.endswith(".json"):         return pd.read_json(file)
    return None

try:
    df_raw = load_file(uploaded)
    if df_raw is None: st.error("Unsupported format."); st.stop()
except Exception as e:
    st.error(f"Load error: {e}"); st.stop()

target_col = target_inp.strip() if target_inp else None
if target_col:
    matches = [c for c in df_raw.columns if c.lower()==target_col.lower()]
    if matches: target_col = matches[0]
    else:
        st.sidebar.warning(f"'{target_col}' not found"); target_col = None

# Preview
with st.expander("🗂️ Raw Data Preview", expanded=False):
    st.dataframe(df_raw.head(100), use_container_width=True)
    st.caption(f"Shape: {df_raw.shape}  |  Showing first 100 rows")

# ════════════════════════════════════════════════════════
#  PHASE DISPATCH
# ════════════════════════════════════════════════════════

if "Analysis" in phase:
    phase_analysis(df_raw)

elif "Cleaning" in phase:
    clean_opts = {
        "impute_method": imp_method,
        "outlier_method": out_method,
        "drop_high_miss_cols": drop_cols,
        "drop_col_thresh": drop_thresh,
        "drop_high_miss_rows": drop_rows,
        "drop_row_thresh": row_thresh,
    }

    with st.spinner("🧹 Cleaning data..."):
        df_clean, report = phase_cleaning(df_raw, clean_opts)

    sec("📋","Cleaning Steps Performed")
    for icon, title, detail, status in report:
        color = "#4ade80" if status=="done" else "#fbbf24"
        st.markdown(f"""
        <div class='step-card'>
          <div class='step-title' style='color:{color}'>{icon} {title}</div>
          <div class='step-desc'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    sec("📊","Before vs After")
    b1,b2,b3,b4 = st.columns(4)
    with b1: st.metric("Rows Before", f"{df_raw.shape[0]:,}")
    with b2: st.metric("Rows After",  f"{df_clean.shape[0]:,}", delta=f"{df_clean.shape[0]-df_raw.shape[0]}")
    with b3: st.metric("Cols Before", df_raw.shape[1])
    with b4: st.metric("Cols After",  df_clean.shape[1], delta=df_clean.shape[1]-df_raw.shape[1])

    miss_before = df_raw.isnull().sum().sum()
    miss_after  = df_clean.isnull().sum().sum()
    st.metric("Missing Values Remaining", f"{miss_after:,}",
              delta=f"{miss_after-miss_before}", delta_color="inverse")

    sec("👀","Cleaned Data Preview")
    st.dataframe(df_clean.head(100), use_container_width=True)

    # Compare distributions
    num_c, _, _ = detect_types(df_raw)
    common_num  = [c for c in num_c if re.sub(r"[^a-zA-Z0-9_]","_",c).strip("_").lower() in df_clean.columns]
    if common_num:
        sec("📈","Distribution: Before vs After Cleaning")
        show_cols = common_num[:6]
        for i in range(0, len(show_cols), 3):
            batch = show_cols[i:i+3]
            fig, axes = plt.subplots(2, len(batch), figsize=(5*len(batch), 6))
            if len(batch)==1: axes = [[axes[0]], [axes[1]]]
            for j, col in enumerate(batch):
                clean_name = re.sub(r"[^a-zA-Z0-9_]","_",col).strip("_").lower()
                raw_data   = df_raw[col].dropna()
                clean_data = df_clean[clean_name].dropna() if clean_name in df_clean else pd.Series([])
                axes[0][j].hist(raw_data, bins=25, color="#f87171", edgecolor="none", alpha=0.8)
                axes[0][j].set_title(f"{col}\n(Before)", fontsize=8)
                if len(clean_data):
                    axes[1][j].hist(clean_data, bins=25, color="#4ade80", edgecolor="none", alpha=0.8)
                    axes[1][j].set_title(f"{clean_name}\n(After)", fontsize=8)
            fig.tight_layout(); fig_show(fig)

    # Download
    st.markdown("---")
    buf = io.StringIO(); df_clean.to_csv(buf, index=False)
    stem = re.sub(r"\.[^.]+$","",uploaded.name)
    st.download_button("⬇️ Download Cleaned Dataset (CSV)", buf.getvalue(),
                       file_name=f"cleaned_{stem}.csv", mime="text/csv")

elif "Model" in phase:
    clean_opts = {
        "impute_method": imp_method,
        "outlier_method": out_method,
        "drop_high_miss_cols": drop_cols,
        "drop_col_thresh": drop_thresh,
        "drop_high_miss_rows": drop_rows,
        "drop_row_thresh": row_thresh,
    }
    model_opts = {
        "encoding": encoding,
        "scaling": scaling,
        "fix_skewness": fix_skew,
        "remove_low_var": rem_lowvar,
        "remove_high_corr": rem_hicorr,
    }

    with st.spinner("🧹 Cleaning..."):
        df_clean, _ = phase_cleaning(df_raw, clean_opts)

    # Resolve target after cleaning
    tc_clean = None
    if target_col:
        tc_clean_name = re.sub(r"[^a-zA-Z0-9_]","_",target_col).strip("_").lower()
        if tc_clean_name in df_clean.columns:
            tc_clean = tc_clean_name

    with st.spinner("🚀 Preparing for model..."):
        df_model, steps = phase_model_ready(df_clean, model_opts, tc_clean)

    sec("🚀","Model Preparation Steps")
    for icon, title, detail, status in steps:
        color = "#4ade80"
        st.markdown(f"""
        <div class='step-card'>
          <div class='step-title' style='color:{color}'>{icon} {title}</div>
          <div class='step-desc'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    sec("📊","Final Dataset Summary")
    f1,f2,f3,f4 = st.columns(4)
    with f1: st.metric("Original Shape", f"{df_raw.shape[0]} × {df_raw.shape[1]}")
    with f2: st.metric("Model-Ready Shape", f"{df_model.shape[0]} × {df_model.shape[1]}")
    with f3: st.metric("Missing Values", f"{df_model.isnull().sum().sum()}")
    with f4: st.metric("Features", df_model.shape[1]-(1 if tc_clean else 0))

    if df_model.isnull().sum().sum() == 0:
        ibox("✅ <b>Zero missing values</b> — dataset is clean!", "good")
    else:
        ibox(f"⚠️ {df_model.isnull().sum().sum()} missing values remain", "warn")

    num_f, cat_f, _ = detect_types(df_model)
    if cat_f:
        ibox(f"ℹ️ {len(cat_f)} non-numeric columns remain: {cat_f}. Consider encoding them.", "info")
    else:
        ibox("✅ All columns are numeric — ready for ML models!", "good")

    sec("👀","Model-Ready Data Preview")
    st.dataframe(df_model.head(100), use_container_width=True)

    # Feature distributions after scaling
    num_m, _, _ = detect_types(df_model)
    feat_cols_show = [c for c in num_m if c != tc_clean][:9]
    if feat_cols_show:
        sec("📈","Feature Distributions (Post Processing)")
        per_row = 3
        for i in range(0, len(feat_cols_show), per_row):
            batch = feat_cols_show[i:i+per_row]
            fig, axes = plt.subplots(1, len(batch), figsize=(5*len(batch), 3.5))
            if len(batch)==1: axes=[axes]
            for ax, col in zip(axes, batch):
                d = df_model[col].dropna()
                ax.hist(d, bins=30, color=PAL[feat_cols_show.index(col)%len(PAL)],
                        edgecolor="none", alpha=0.8)
                ax.set_title(col, fontsize=8)
                ax.text(0.97,0.95,f"skew={skew(d):.2f}",
                        transform=ax.transAxes,ha="right",va="top",fontsize=7,color="#94a3b8")
            fig.tight_layout(); fig_show(fig)

    # Correlation of final features
    if len(num_m) >= 2:
        sec("🔗","Final Feature Correlation")
        corr = df_model[num_m[:20]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220,10,as_cmap=True),
                    center=0, annot=len(num_m)<=15, fmt=".2f",
                    linewidths=0.5, linecolor="#080c14", ax=ax, annot_kws={"size":6})
        ax.set_title("Final Feature Correlation (Post Processing)", fontsize=10)
        fig.tight_layout(); fig_show(fig)

    # Target analysis
    if tc_clean and tc_clean in df_model.columns:
        sec("🎯","Target Column Analysis")
        target_data = df_model[tc_clean]
        num_m2, _, _ = detect_types(df_model)
        if tc_clean in num_m2:
            c1,c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(target_data.dropna(), bins=35, color="#22d3ee", edgecolor="none", alpha=0.85)
                ax.set_title(f"Target Distribution: {tc_clean}", fontsize=10)
                fig.tight_layout(); fig_show(fig)
            with c2:
                other = [c for c in num_m2 if c!=tc_clean]
                if other:
                    corr_t = df_model[other+[tc_clean]].corr()[tc_clean].drop(tc_clean).sort_values(key=abs, ascending=False).head(10)
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.barh(corr_t.index, corr_t.values,
                            color=["#4ade80" if v>0 else "#f87171" for v in corr_t], height=0.6)
                    ax.axvline(0,color="#64748b")
                    ax.set_title(f"Top Feature Correlations with {tc_clean}", fontsize=9)
                    fig.tight_layout(); fig_show(fig)
        else:
            vc = target_data.value_counts()
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(vc.index.astype(str), vc.values, color=PAL[:len(vc)], edgecolor="none")
            for i,v in enumerate(vc.values):
                ax.text(i, v+max(vc)*0.01, f"{v:,}\n({v/len(df_model)*100:.1f}%)", ha="center", fontsize=8)
            ax.set_title(f"Target Class Distribution: {tc_clean}", fontsize=10)
            fig.tight_layout(); fig_show(fig)

    # ── ML Readiness Checklist ──
    sec("✅","ML Readiness Checklist")
    checks = [
        (df_model.isnull().sum().sum() == 0, "No missing values"),
        (len(detect_types(df_model)[1]) == 0, "All columns numeric (encoded)"),
        (df_model.duplicated().sum() == 0, "No duplicate rows"),
        (df_model.shape[1] > 1, "Multiple features available"),
        (tc_clean is not None and tc_clean in df_model.columns, "Target column present"),
    ]
    all_pass = all(c[0] for c in checks)
    for passed, label in checks:
        icon = "✅" if passed else "❌"
        color = "#4ade80" if passed else "#f87171"
        st.markdown(f'<div style="padding:6px 0;font-size:0.88rem">'
                    f'<span style="color:{color}">{icon}</span> {label}</div>',
                    unsafe_allow_html=True)

    if all_pass:
        ibox("🎉 <b>Dataset is fully ready for Machine Learning!</b> You can now train your model.", "good")
    else:
        ibox("⚠️ Some checks failed. Review the issues above before training.", "warn")

    # Downloads
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        buf1 = io.StringIO(); df_clean.to_csv(buf1, index=False)
        stem = re.sub(r"\.[^.]+$","",uploaded.name)
        st.download_button("⬇️ Download Cleaned CSV",  buf1.getvalue(),
                           file_name=f"cleaned_{stem}.csv", mime="text/csv")
    with c2:
        buf2 = io.StringIO(); df_model.to_csv(buf2, index=False)
        st.download_button("🚀 Download Model-Ready CSV", buf2.getvalue(),
                           file_name=f"model_ready_{stem}.csv", mime="text/csv")