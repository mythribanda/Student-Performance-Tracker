"""
Student Performance Tracker & Analysis System
Streamlit Dashboard — app/app.py
Compatible with Streamlit ≥ 1.40 (uses width= instead of use_container_width=)
"""

import os, sys, warnings, io
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit"); sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'StudentPerformanceFactors.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')

# ── Colours ───────────────────────────────────────────────────────────────────
PRIMARY = '#6C63FF'; ACCENT = '#48BB78'; TEXT = '#E2E8F0'
PALETTE = [PRIMARY, ACCENT, '#F6AD55', '#FC8181', '#63B3ED', '#B794F4']

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Tracker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:#0F0F1A;color:#E2E8F0;}
.main{background:#0F0F1A;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1A1A2E 0%,#16213E 100%);border-right:1px solid rgba(108,99,255,.2);}
[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#1A1A2E,#16213E);border:1px solid rgba(108,99,255,.3);border-radius:12px;padding:16px;}
[data-testid="stMetricValue"]{color:#6C63FF !important;font-weight:700;}
[data-testid="stMetricLabel"]{color:#94A3B8 !important;}
h1,h2,h3{color:#E2E8F0 !important;}
.hero{font-size:2.6rem;font-weight:700;background:linear-gradient(135deg,#6C63FF,#48BB78);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;}
.sec{font-size:1.3rem;font-weight:600;color:#E2E8F0;border-left:3px solid #6C63FF;padding-left:12px;margin:22px 0 14px;}
.card{background:linear-gradient(135deg,rgba(108,99,255,.1),rgba(72,187,120,.05));border:1px solid rgba(108,99,255,.25);border-radius:12px;padding:14px 18px;margin:7px 0;}
.card p{color:#CBD5E0;margin:0;font-size:.88rem;}
.card h4{color:#6C63FF;margin:0 0 5px;font-size:.97rem;}
.score-big{font-size:3.8rem;font-weight:700;text-align:center;background:linear-gradient(135deg,#6C63FF,#48BB78);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stButton>button{background:linear-gradient(135deg,#6C63FF,#5A54E8);color:#fff;border:none;border-radius:8px;padding:10px 24px;font-weight:600;font-family:'Space Grotesk',sans-serif;}
.stButton>button:hover{opacity:.9;transform:translateY(-1px);}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mpl_dark():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1A1A2E', 'axes.facecolor': '#16213E',
        'axes.edgecolor':   '#334155', 'axes.labelcolor': '#CBD5E0',
        'xtick.color':      '#94A3B8', 'ytick.color':    '#94A3B8',
        'grid.color':       '#1E3A5F', 'grid.alpha':     0.4,
        'text.color':       '#E2E8F0',
    })


def show(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, width='stretch')   # ← replaces use_container_width=True
    plt.close(fig)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data
def numeric_df(df):
    from sklearn.preprocessing import LabelEncoder
    d = df.copy()
    for col in d.select_dtypes(include='object').columns:
        d[col] = LabelEncoder().fit_transform(d[col].astype(str))
    return d


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Performance")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Data Analysis",
        "🤖 Model Performance",
        "🔮 Score Predictor",
    ], label_visibility='collapsed')
    st.markdown("---")
    st.markdown("<p style='color:#94A3B8;font-size:.85rem;'>ML-powered system to analyse student behaviour and predict exam scores.</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    if st.button("⚡ Train / Retrain Models"):
        with st.spinner("Training all models…"):
            try:
                from train import main as train_main
                train_main()
                st.cache_resource.clear()
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Analysis":
    st.markdown('<div class="hero">📊 Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Exploratory analysis of student performance factors</p>",
                unsafe_allow_html=True)

    df     = load_data()
    df_num = numeric_df(df)

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📚 Students",  f"{len(df):,}")
    c2.metric("📋 Features",  str(df.shape[1]-1))
    c3.metric("🎯 Avg Score", f"{df['Exam_Score'].mean():.1f}")
    c4.metric("⬆️ Max Score", f"{df['Exam_Score'].max():.0f}")
    c5.metric("⬇️ Min Score", f"{df['Exam_Score'].min():.0f}")

    # Dataset overview
    st.markdown('<div class="sec">Dataset Overview</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📋 Sample Data", "📈 Statistics"])
    with t1:
        st.dataframe(df.head(20), width='stretch', height=280)   # ← fixed
    with t2:
        st.dataframe(df.describe().round(2), width='stretch')     # ← fixed

    # Upload
    with st.expander("📤 Upload Your Own CSV"):
        up = st.file_uploader("CSV with same columns", type='csv')
        if up:
            st.dataframe(pd.read_csv(up).head(10), width='stretch')

    mpl_dark()

    # Row 1 — distribution + motivation box
    st.markdown('<div class="sec">Score Distributions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df['Exam_Score'], bins=30, color=PRIMARY, alpha=.85, edgecolor='#0F0F1A')
        ax.axvline(df['Exam_Score'].mean(), color=ACCENT, lw=2, ls='--',
                   label=f"Mean: {df['Exam_Score'].mean():.1f}")
        ax.set_title('Exam Score Distribution', fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.set_xlabel('Exam Score'); ax.set_ylabel('Count')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        show(fig)

    with col2:
        order = ['Low','Medium','High']
        pal_m = {'Low':'#FC8181','Medium':'#F6AD55','High':ACCENT}
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=df, x='Motivation_Level', y='Exam_Score',
                    palette=pal_m, order=order, ax=ax)
        ax.set_title('Score by Motivation Level', fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.grid(True, alpha=.3)
        show(fig)

    # Row 2 — study hours + attendance
    st.markdown('<div class="sec">Key Factor Relationships</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df['Hours_Studied'], df['Exam_Score'], alpha=.3, c=PRIMARY, s=15, edgecolors='none')
        m,b = np.polyfit(df['Hours_Studied'], df['Exam_Score'], 1)
        xs  = np.linspace(df['Hours_Studied'].min(), df['Hours_Studied'].max(), 100)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2, label=f'Trend (slope={m:.2f})')
        ax.set_title('Study Hours vs Exam Score', fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.set_xlabel('Hours Studied'); ax.set_ylabel('Exam Score')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        show(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df['Attendance'], df['Exam_Score'], alpha=.3, c='#F6AD55', s=15, edgecolors='none')
        m,b = np.polyfit(df['Attendance'], df['Exam_Score'], 1)
        xs  = np.linspace(df['Attendance'].min(), df['Attendance'].max(), 100)
        ax.plot(xs, m*xs+b, color='#FC8181', lw=2, label=f'Trend (slope={m:.2f})')
        ax.set_title('Attendance vs Exam Score', fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.set_xlabel('Attendance (%)'); ax.set_ylabel('Exam Score')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        show(fig)

    # Row 3 — sleep + parental involvement
    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df['Sleep_Hours'], df['Exam_Score'], alpha=.3, c='#B794F4', s=15, edgecolors='none')
        m,b = np.polyfit(df['Sleep_Hours'], df['Exam_Score'], 1)
        xs  = np.linspace(df['Sleep_Hours'].min(), df['Sleep_Hours'].max(), 100)
        ax.plot(xs, m*xs+b, color='#63B3ED', lw=2, label=f'Trend (slope={m:.2f})')
        ax.set_title('Sleep Hours vs Exam Score', fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.set_xlabel('Sleep Hours'); ax.set_ylabel('Exam Score')
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        show(fig)

    with col6:
        fig, ax = plt.subplots(figsize=(6,4))
        means = df.groupby('Parental_Involvement')['Exam_Score'].mean().reindex(order)
        bar_cols = ['#FC8181','#F6AD55',ACCENT]
        bars = ax.bar(means.index, means.values, color=bar_cols, edgecolor='none', width=0.5)
        for bar,val in zip(bars, means.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.3, f'{val:.1f}',
                    ha='center', color=TEXT, fontsize=10, fontweight='600')
        ax.set_title('Avg Score by Parental Involvement', fontsize=12, fontweight='bold', color=TEXT, pad=10)
        ax.set_ylabel('Average Score'); ax.grid(True, alpha=.3, axis='y')
        show(fig)

    # Correlation heatmap
    st.markdown('<div class="sec">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14,7))
    corr = df_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap=sns.diverging_palette(260,150,as_cmap=True),
                ax=ax, linewidths=.5, linecolor='#0F0F1A',
                annot_kws={'size':8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', color=TEXT, pad=14)
    show(fig)

    # Previous scores scatter
    st.markdown('<div class="sec">Previous Scores vs Exam Score</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12,4))
    sc = ax.scatter(df['Previous_Scores'], df['Exam_Score'],
                    c=df_num['Motivation_Level'], cmap='viridis',
                    alpha=.45, s=18, edgecolors='none')
    plt.colorbar(sc, ax=ax).set_label('Motivation (encoded)', color=TEXT)
    m,b = np.polyfit(df['Previous_Scores'], df['Exam_Score'], 1)
    xs  = np.linspace(df['Previous_Scores'].min(), df['Previous_Scores'].max(), 100)
    ax.plot(xs, m*xs+b, color='#FC8181', lw=2, ls='--')
    ax.set_title('Previous Scores vs Exam Score (coloured by Motivation)',
                 fontsize=12, fontweight='bold', color=TEXT)
    ax.set_xlabel('Previous Scores'); ax.set_ylabel('Exam Score'); ax.grid(True, alpha=.3)
    show(fig)

    # Insights
    st.markdown('<div class="sec">Key Insights</div>', unsafe_allow_html=True)
    insights = [
        ("📚 Study Hours Drive Scores",
         f"Students studying ≥30 hrs/week score {df[df['Hours_Studied']>=30]['Exam_Score'].mean():.1f} on average, "
         f"vs {df[df['Hours_Studied']<10]['Exam_Score'].mean():.1f} for those studying <10 hrs."),
        ("🎯 Attendance is Critical",
         f"High attendance (≥90 %) → avg {df[df['Attendance']>=90]['Exam_Score'].mean():.1f}; "
         f"low (<75 %) → avg {df[df['Attendance']<75]['Exam_Score'].mean():.1f}."),
        ("💡 Motivation Matters",
         f"High motivation: {df[df['Motivation_Level']=='High']['Exam_Score'].mean():.1f} "
         f"vs Low motivation: {df[df['Motivation_Level']=='Low']['Exam_Score'].mean():.1f}."),
        ("👨‍👩‍👧 Parental Involvement",
         f"High parental involvement → avg score {df[df['Parental_Involvement']=='High']['Exam_Score'].mean():.1f}; "
         f"Low → {df[df['Parental_Involvement']=='Low']['Exam_Score'].mean():.1f}."),
        ("🎓 Tutoring Helps",
         f"Students with tutoring sessions score higher on average than those without."),
        ("😴 Sleep Quality",
         f"Students sleeping 7–9 hours average "
         f"{df[(df['Sleep_Hours']>=7)&(df['Sleep_Hours']<=9)]['Exam_Score'].mean():.1f} — optimal rest matters."),
    ]
    for title, body in insights:
        st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="hero">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Comparing ML models trained on student performance data</p>",
                unsafe_allow_html=True)

    payload = load_model()
    if payload is None:
        st.warning("⚠️ No trained model found. Click **⚡ Train / Retrain Models** in the sidebar.")
        st.stop()

    results = payload['all_results']
    best    = payload['best_model_name']
    fi      = payload.get('feature_importance', {})
    bm      = results[best]

    st.markdown(f"""<div class="card" style="border-color:rgba(72,187,120,.5);">
        <h4>🏆 Best Model: {best}</h4>
        <p>R² = <strong>{bm['R2']}</strong> &nbsp;|&nbsp;
           MAE = <strong>{bm['MAE']}</strong> &nbsp;|&nbsp;
           RMSE = <strong>{bm['RMSE']}</strong></p>
    </div>""", unsafe_allow_html=True)

    # Metrics table
    st.markdown('<div class="sec">All Model Metrics</div>', unsafe_allow_html=True)
    df_r = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Model'})
    df_r['Best'] = df_r['Model'].apply(lambda m: '🏆' if m == best else '')
    st.dataframe(df_r.set_index('Model'), width='stretch')    # ← fixed

    # Bar charts
    st.markdown('<div class="sec">Model Comparison</div>', unsafe_allow_html=True)
    mpl_dark()
    names  = list(results.keys())
    r2s    = [results[m]['R2']   for m in names]
    maes   = [results[m]['MAE']  for m in names]
    rmses  = [results[m]['RMSE'] for m in names]
    cols   = [ACCENT if m == best else PRIMARY for m in names]

    fig, axes = plt.subplots(1,3, figsize=(14,4))
    for ax, vals, title, higher in zip(axes,
            [r2s, maes, rmses],
            ['R² Score (↑ better)', 'MAE (↓ better)', 'RMSE (↓ better)'],
            [True, False, False]):
        bars = ax.barh(names, vals, color=cols, edgecolor='none', height=.55)
        for bar,val in zip(bars, vals):
            ax.text(val + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', color=TEXT, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT)
        ax.grid(True, alpha=.3, axis='x')
    plt.tight_layout(pad=2)
    show(fig)

    # Feature importance
    if fi:
        st.markdown('<div class="sec">Feature Importance</div>', unsafe_allow_html=True)
        fi_s = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        fig, ax = plt.subplots(figsize=(10,6))
        fn = list(fi_s.keys()); fv = list(fi_s.values())
        bc = [ACCENT if i==0 else PRIMARY for i in range(len(fn))]
        ax.barh(fn[::-1], fv[::-1], color=bc[::-1], edgecolor='none', height=.6)
        for i,(n_,v_) in enumerate(zip(fn[::-1], fv[::-1])):
            ax.text(v_+max(fv)*0.005, i, f'{v_:.4f}', va='center', color=TEXT, fontsize=9)
        ax.set_title(f'Feature Importance — {best}', fontsize=13, fontweight='bold', color=TEXT, pad=12)
        ax.set_xlabel('Importance Score'); ax.grid(True, alpha=.3, axis='x')
        show(fig)
        st.markdown(f'<div class="card"><h4>🔍 Top Feature: {fn[0]}</h4>'
                    f'<p>This has the highest influence on the predicted exam score.</p></div>',
                    unsafe_allow_html=True)

    # Radar
    st.markdown('<div class="sec">Radar Comparison</div>', unsafe_allow_html=True)
    max_mae = max(maes); max_rmse = max(rmses)
    cats = ['R²', 'MAE\n(inv)', 'RMSE\n(inv)']
    N = len(cats)
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(7,5), subplot_kw=dict(polar=True))
    ax.set_facecolor('#16213E')
    for i,(mn, r2, mae, rmse) in enumerate(zip(names, r2s, maes, rmses)):
        vals = [r2, 1-mae/max_mae, 1-rmse/max_rmse] + [r2]
        ax.plot(angles, vals, color=PALETTE[i%len(PALETTE)], lw=2, label=mn)
        ax.fill(angles, vals, color=PALETTE[i%len(PALETTE)], alpha=.07)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, color=TEXT, fontsize=10)
    ax.set_yticks([.2,.4,.6,.8,1]); ax.set_yticklabels(['.2','.4','.6','.8','1'], color='#475569', fontsize=8)
    ax.grid(color='#1E3A5F', alpha=.5); ax.spines['polar'].set_color('#334155')
    ax.set_title('Normalised Model Performance', fontsize=12, fontweight='bold', color=TEXT, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.1), fontsize=9,
              framealpha=.2, labelcolor=TEXT)
    show(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SCORE PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Score Predictor":
    st.markdown('<div class="hero">🔮 Score Predictor</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Enter student details to predict their expected exam score</p>",
                unsafe_allow_html=True)

    payload = load_model()
    if payload is None:
        st.warning("⚠️ No trained model found. Click **⚡ Train / Retrain Models** in the sidebar.")
        st.stop()

    from predict import predict_single, predict_batch

    # ── Single prediction ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">Single Student Prediction</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        hours      = st.slider("📚 Hours Studied (per week)", 1, 44, 20)
        attendance = st.slider("📅 Attendance (%)", 60.0, 100.0, 80.0, 0.5)
        sleep      = st.slider("😴 Sleep Hours (per night)", 4, 10, 7)
        prev_score = st.slider("📝 Previous Score", 50, 100, 70)
        tutoring   = st.slider("🎓 Tutoring Sessions (per month)", 0, 8, 2)
        phys_act   = st.slider("🏃 Physical Activity (hrs/week)", 0, 6, 3)

    with c2:
        motivation   = st.selectbox("💡 Motivation Level",        ['Low','Medium','High'], index=1)
        parental_inv = st.selectbox("👨‍👩‍👧 Parental Involvement",  ['Low','Medium','High'], index=1)
        access_res   = st.selectbox("🖥️ Access to Resources",    ['Low','Medium','High'], index=1)
        family_inc   = st.selectbox("💰 Family Income",            ['Low','Medium','High'], index=1)
        teacher_q    = st.selectbox("👩‍🏫 Teacher Quality",        ['Low','Medium','High'], index=1)
        peer_inf     = st.selectbox("👥 Peer Influence",           ['Negative','Neutral','Positive'], index=1)
        internet     = st.selectbox("🌐 Internet Access",          ['Yes','No'])
        extra_curr   = st.selectbox("🎭 Extracurricular",          ['Yes','No'])
        learning_dis = st.selectbox("📋 Learning Disabilities",    ['Yes','No'], index=1)
        school_type  = st.selectbox("🏫 School Type",              ['Public','Private'])
        par_edu      = st.selectbox("🎓 Parental Education",       ['High School','College','Postgraduate'], index=1)
        distance     = st.selectbox("🏠 Distance from Home",       ['Near','Moderate','Far'])
        gender       = st.selectbox("👤 Gender",                   ['Male','Female'])

    if st.button("🔮 Predict Exam Score", use_container_width=False):
        score = predict_single(
            Hours_Studied=hours,
            Attendance=attendance,
            Sleep_Hours=sleep,
            Previous_Scores=prev_score,
            Tutoring_Sessions=tutoring,
            Physical_Activity=phys_act,
            Motivation_Level=motivation,
            Parental_Involvement=parental_inv,
            Access_to_Resources=access_res,
            Family_Income=family_inc,
            Teacher_Quality=teacher_q,
            Peer_Influence=peer_inf,
            Internet_Access=internet,
            Extracurricular_Activities=extra_curr,
            Learning_Disabilities=learning_dis,
            School_Type=school_type,
            Parental_Education_Level=par_edu,
            Distance_from_Home=distance,
            Gender=gender,
        )

        if   score >= 85: grade, gc = 'A — Excellent',  '#48BB78'
        elif score >= 70: grade, gc = 'B — Good',        '#6C63FF'
        elif score >= 55: grade, gc = 'C — Average',     '#F6AD55'
        else:             grade, gc = 'D — Needs Work',  '#FC8181'

        st.markdown("---")
        _, mid, _ = st.columns([1,2,1])
        with mid:
            st.markdown(f'<div class="score-big">{score}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:center;"><span style="background:{gc}20;color:{gc};'
                f'border:1px solid {gc};padding:5px 18px;border-radius:20px;font-weight:600;">'
                f'{grade}</span></div>', unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;color:#94A3B8;margin-top:8px;'>"
                        f"Predicted by: <strong>{payload['best_model_name']}</strong></p>",
                        unsafe_allow_html=True)

        # Gauge
        mpl_dark()
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.set_aspect('equal'); ax.axis('off')
        theta = np.linspace(np.pi, 0, 300)
        ax.plot(np.cos(theta), np.sin(theta), lw=18, color='#1E3A5F', solid_capstyle='butt')
        score_angle = np.pi*(1 - score/100)
        tf = np.linspace(np.pi, score_angle, 300)
        c0, c1_ = (0.42,0.38,1.0), (0.28,0.73,0.47)
        for i in range(len(tf)-1):
            fr = i/len(tf)
            col = tuple(c0[c]*(1-fr)+c1_[c]*fr for c in range(3))
            ax.plot([np.cos(tf[i]),np.cos(tf[i+1])],
                    [np.sin(tf[i]),np.sin(tf[i+1])], lw=18, color=col, solid_capstyle='butt')
        ax.text(0,-0.15,f'{score}', ha='center', va='center', fontsize=34, fontweight='700', color=TEXT)
        ax.text(0,-0.45,'Predicted Score', ha='center', va='center', fontsize=10, color='#94A3B8')
        ax.text(-1.0,-0.1,'0',   ha='center', color='#475569', fontsize=9)
        ax.text( 1.0,-0.1,'100', ha='center', color='#475569', fontsize=9)
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.6,1.1)
        _, mid2, _ = st.columns([1,2,1])
        with mid2:
            show(fig)

        # Tips
        st.markdown('<div class="sec">Personalised Recommendations</div>', unsafe_allow_html=True)
        tips = []
        if hours < 20:
            tips.append(("📚 Increase Study Time",
                         f"You study {hours} hrs/week. Aim for 20–30 hrs for stronger retention."))
        if attendance < 80:
            tips.append(("📅 Improve Attendance",
                         f"Attendance is {attendance:.0f}%. Target ≥ 90% to avoid missing key lessons."))
        if sleep < 7 or sleep > 9:
            tips.append(("😴 Optimise Sleep",
                         f"You sleep {sleep} hrs/night. Research shows 7–9 hrs is optimal."))
        if motivation == 'Low':
            tips.append(("💡 Boost Motivation",
                         "Set small daily goals and track progress. Momentum builds naturally."))
        if parental_inv == 'Low':
            tips.append(("👨‍👩‍👧 Parental Support",
                         "Students with higher parental involvement consistently score higher."))
        if peer_inf == 'Negative':
            tips.append(("👥 Peer Environment",
                         "Surrounding yourself with positive peers can significantly lift scores."))
        if tutoring == 0:
            tips.append(("🎓 Consider Tutoring",
                         "Even 1–2 sessions/month can address weak areas effectively."))
        if not tips:
            tips.append(("🌟 Great Habits!", "Your study profile looks strong — keep it consistent!"))
        for title, body in tips:
            st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                        unsafe_allow_html=True)

    # ── What-if simulator ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">What-If Score Simulator</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;font-size:.9rem;'>See how changing study hours affects the predicted score (all other factors held at medium/default).</p>",
                unsafe_allow_html=True)

    base_h = st.slider("Base Hours Studied (per week)", 1, 44, 20, key='wi')
    hr_range   = list(range(1, 45))
    sim_scores = []
    for h in hr_range:
        s = predict_single(
            Hours_Studied=h, Attendance=80, Sleep_Hours=7, Previous_Scores=70,
            Tutoring_Sessions=2, Physical_Activity=3,
            Motivation_Level='Medium', Parental_Involvement='Medium',
            Access_to_Resources='Medium', Family_Income='Medium',
            Teacher_Quality='Medium', Peer_Influence='Neutral',
            Internet_Access='Yes', Extracurricular_Activities='No',
            Learning_Disabilities='No', School_Type='Public',
            Parental_Education_Level='College', Distance_from_Home='Moderate',
            Gender='Male',
        )
        sim_scores.append(s)

    mpl_dark()
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot(hr_range, sim_scores, color=PRIMARY, lw=2.5)
    ax.fill_between(hr_range, sim_scores, alpha=.15, color=PRIMARY)
    base_s = sim_scores[base_h - 1]
    ax.axvline(base_h, color=ACCENT, ls='--', lw=1.5)
    ax.scatter([base_h], [base_s], color=ACCENT, s=80, zorder=5)
    ax.annotate(f'{base_s:.1f}', (base_h, base_s),
                textcoords='offset points', xytext=(8,6), color=ACCENT, fontweight='600')
    ax.set_title('Predicted Score vs Weekly Study Hours (other factors fixed at medium)',
                 fontsize=11, fontweight='bold', color=TEXT, pad=10)
    ax.set_xlabel('Hours Studied (per week)'); ax.set_ylabel('Predicted Score')
    ax.grid(True, alpha=.3)
    show(fig)

    # ── Batch predictor ───────────────────────────────────────────────────────
    st.markdown('<div class="sec">Batch Prediction — Upload CSV</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;font-size:.9rem;'>Upload a CSV with student feature columns to predict scores in bulk.</p>",
                unsafe_allow_html=True)
    batch_file = st.file_uploader("Upload CSV (feature columns only, no Exam_Score)", type='csv', key='batch')
    if batch_file:
        df_b = pd.read_csv(batch_file)
        st.dataframe(df_b.head(), width='stretch')    # ← fixed
        if st.button("📊 Predict Batch"):
            try:
                preds = predict_batch(df_b)
                df_b['Predicted_Exam_Score'] = preds
                st.success(f"Predicted scores for {len(df_b)} students!")
                st.dataframe(df_b, width='stretch')   # ← fixed
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=df_b.to_csv(index=False).encode('utf-8'),
                    file_name="student_predictions.csv",
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")