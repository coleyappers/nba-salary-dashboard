"""
NBA Salary Fair-Market Value Detector
Streamlit dashboard for the IEOR 142A project.

Run locally:    streamlit run app.py
Deploy:         push to GitHub, connect to share.streamlit.io
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from pathlib import Path

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="NBA Fair-Market Value Detector",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.02em;
    color: #0a1838;
}

h1 { font-size: 3.5rem !important; margin-bottom: 0 !important; }

.hero {
    background: linear-gradient(135deg, #0a1838 0%, #1d428a 60%, #c8102e 100%);
    padding: 2.5rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 2rem;
}
.hero h1 { color: white; font-size: 4rem !important; margin-bottom: 0 !important; line-height: 1; }
.hero .tag {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    max-width: 800px;
}
.hero .meta {
    color: rgba(255,255,255,0.65);
    font-size: 0.85rem;
    margin-top: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.metric-box {
    background: white;
    border: 1px solid #e6e8ec;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.metric-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    color: #6a7385;
    font-weight: 600;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #0a1838;
    line-height: 1.1;
    margin-top: 0.3rem;
}
.metric-delta-pos { color: #c8102e; font-weight: 600; }
.metric-delta-neg { color: #1d428a; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
    padding: 0.75rem 1.25rem;
}

.footer {
    color: #8b94a3;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e6e8ec;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    with open('model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts['model']
feature_cols = artifacts['feature_cols']
df_model = artifacts['df_model'].copy()
X = artifacts['X']
shap_values = artifacts['shap_values']
expected_value = artifacts['explainer_expected_value']
results_df = artifacts['results_df']

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>NBA FAIR-MARKET VALUE DETECTOR</h1>
    <div class="tag">A salary prediction model that flags the league's most over- and underpaid contracts heading into the 2025-26 season. Trained on 2024-25 production data with XGBoost.</div>
    <div class="meta">IEOR 142A · Spring 2026 · Cole Yap · Lily Salazar · Ian Reikes · Kai Chen · Andrew Vitt</div>
</div>
""", unsafe_allow_html=True)

# ---------- TOP-LINE METRICS ----------
col1, col2, col3, col4 = st.columns(4)
best_row = results_df.iloc[results_df['Test R²'].idxmax()]
xgb_row = results_df[results_df['Model'] == 'XGBoost'].iloc[0]

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Players Modeled</div>
        <div class="metric-value">{len(df_model)}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">XGBoost CV R²</div>
        <div class="metric-value">{xgb_row['CV R² (mean ± sd)'].split(' ')[0]}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Median |Residual|</div>
        <div class="metric-value">${df_model['residual_dollars'].abs().median()/1e6:.1f}M</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    most_over = df_model.nlargest(1, 'residual_dollars').iloc[0]
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Most Overpaid</div>
        <div class="metric-value" style="font-size: 1.4rem;">{most_over['Player'].split()[0][:1]}. {most_over['Player'].split()[-1]}</div>
        <div class="metric-delta-pos">+${most_over['residual_dollars']/1e6:.1f}M</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Leaderboard", "🔍 Player Predictor", "🧠 Model Diagnostics", "📖 About"])

# =================================================================
# TAB 1: LEADERBOARD
# =================================================================
with tab1:
    st.subheader("League-wide Over/Underpaid Leaderboard")
    st.caption("Residual = Actual 2025-26 salary minus model-predicted fair-market value. Positive = paid above production; negative = paid below.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        view = st.radio("View", ["Most Overpaid", "Most Underpaid", "Full Table"], horizontal=False)
    with c2:
        n_show = st.slider("Number of rows", 5, 50, 15)
    with c3:
        team_filter = st.multiselect(
            "Filter by team (optional)",
            sorted(df_model['Team'].unique()),
            default=[]
        )

    df_display = df_model.copy()
    if team_filter:
        df_display = df_display[df_display['Team'].isin(team_filter)]

    if view == "Most Overpaid":
        df_display = df_display.nlargest(n_show, 'residual_dollars')
    elif view == "Most Underpaid":
        df_display = df_display.nsmallest(n_show, 'residual_dollars')
    else:
        df_display = df_display.sort_values('residual_dollars', ascending=False).head(n_show)

    show = pd.DataFrame({
        'Player': df_display['Player'].values,
        'Team': df_display['Team'].values,
        'Pos': df_display['Pos'].values,
        'Age': df_display['Age'].astype(int).values,
        'PTS/36': df_display['PTS_per36'].round(1).values,
        'AST/36': df_display['AST_per36'].round(1).values,
        'TRB/36': df_display['TRB_per36'].round(1).values,
        'All-Star': df_display['is_allstar'].map({1: '★', 0: ''}).values,
        'Actual': df_display['2025-26'].apply(lambda x: f'${x/1e6:.1f}M').values,
        'Predicted': df_display['predicted_salary'].apply(lambda x: f'${x/1e6:.1f}M').values,
        'Residual': df_display['residual_dollars'].apply(
            lambda x: f'+${x/1e6:.1f}M' if x >= 0 else f'-${abs(x)/1e6:.1f}M'
        ).values,
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Bar chart
    st.write("")
    fig, ax = plt.subplots(figsize=(11, max(4, n_show * 0.35)))
    plot_df = df_display.sort_values('residual_dollars')
    colors = ['#1d428a' if v < 0 else '#c8102e' for v in plot_df['residual_dollars']]
    ax.barh(plot_df['Player'], plot_df['residual_dollars'] / 1e6, color=colors, edgecolor='white')
    ax.set_xlabel('Residual ($M, actual − predicted)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('#fafbfc')
    plt.tight_layout()
    st.pyplot(fig)

# =================================================================
# TAB 2: PLAYER PREDICTOR
# =================================================================
with tab2:
    st.subheader("Predict Fair-Market Value")
    st.caption("Either pick an existing player to see their breakdown, or build a hypothetical player from scratch.")

    mode = st.radio("Mode", ["Existing Player", "Hypothetical Player"], horizontal=True)

    if mode == "Existing Player":
        player = st.selectbox("Select player", sorted(df_model['Player'].unique()))
        idx = df_model.index[df_model['Player'] == player][0]
        pos_in_X = list(X.index).index(idx)
        row = df_model.loc[idx]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Actual 2025-26</div>
                <div class="metric-value">${row['2025-26']/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Predicted (Fair-Market)</div>
                <div class="metric-value">${row['predicted_salary']/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            r = row['residual_dollars']
            label = "Overpaid by" if r >= 0 else "Underpaid by"
            cls = "metric-delta-pos" if r >= 0 else "metric-delta-neg"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">${abs(r)/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("**SHAP feature contributions** — how each feature pushes the prediction up (red) or down (blue) from the league average:")

        # SHAP waterfall
        fig, ax = plt.subplots(figsize=(11, 6))
        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values[pos_in_X],
            X.iloc[pos_in_X],
            max_display=12,
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.markdown("**Build a hypothetical player profile:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 19, 42, 26)
            g = st.slider("Games played", 20, 82, 70)
            gs = st.slider("Games started", 0, 82, 50)
            mp_pg = st.slider("Minutes per game", 10.0, 40.0, 30.0)
        with c2:
            pts = st.slider("Points per 36", 5.0, 35.0, 18.0)
            ast = st.slider("Assists per 36", 0.0, 14.0, 4.0)
            trb = st.slider("Rebounds per 36", 0.0, 18.0, 6.0)
            stl = st.slider("Steals per 36", 0.0, 4.0, 1.0)
            blk = st.slider("Blocks per 36", 0.0, 4.0, 0.5)
        with c3:
            efg = st.slider("eFG%", 0.30, 0.70, 0.52, 0.01)
            ftpct = st.slider("FT%", 0.40, 1.00, 0.78, 0.01)
            tppct = st.slider("3P%", 0.0, 0.50, 0.36, 0.01)
            shots_per_min = st.slider("Shots per minute", 0.10, 1.20, 0.55, 0.01)
            tov_pg = st.slider("Turnovers per game", 0.0, 6.0, 1.5, 0.1)

        c4, c5 = st.columns(2)
        with c4:
            pos = st.selectbox("Position", ['PG', 'SG', 'SF', 'PF', 'C'])
        with c5:
            allstar = st.checkbox("All-Star")
            mvp = st.checkbox("MVP candidate")
            dpoy = st.checkbox("DPOY candidate")

        # Build feature vector
        row = {c: 0 for c in feature_cols}
        row['Age'] = age; row['Age_sq'] = age ** 2
        row['G'] = g; row['GS'] = gs; row['MP_pg'] = mp_pg
        row['PTS_per36'] = pts; row['AST_per36'] = ast; row['TRB_per36'] = trb
        row['STL_per36'] = stl; row['BLK_per36'] = blk
        row['eFG%'] = efg; row['FT%'] = ftpct; row['3P%'] = tppct
        row['shots_per_min'] = shots_per_min; row['stocks_per36'] = stl + blk
        row['TOV_pg'] = tov_pg
        row['is_allstar'] = int(allstar); row['is_mvp_candidate'] = int(mvp); row['is_dpoy_candidate'] = int(dpoy)
        for p in ['PG', 'SG', 'SF', 'PF']:
            if f'Pos_{p}' in row:
                row[f'Pos_{p}'] = 1 if pos == p else 0

        x_input = pd.DataFrame([row])[feature_cols].astype(float)
        pred_log = model.predict(x_input)[0]
        pred_dollars = np.exp(pred_log)

        st.markdown("---")
        st.markdown(f"""
        <div class="metric-box" style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div class="metric-label">Predicted Fair-Market Value</div>
            <div class="metric-value" style="font-size: 4rem; color: #c8102e;">${pred_dollars/1e6:.1f}M</div>
            <div style="color: #6a7385; font-size: 0.9rem; margin-top: 0.5rem;">For the 2025-26 season, based on the inputs above</div>
        </div>
        """, unsafe_allow_html=True)

# =================================================================
# TAB 3: MODEL DIAGNOSTICS
# =================================================================
with tab3:
    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("**XGBoost** was selected as the primary model based on cross-validated R². It captures non-linear effects (the aging curve, max-contract ceilings) that linear models cannot, and SHAP enables per-prediction explanations needed for the analyst use case.")

    st.write("")
    st.subheader("Global Feature Importance (Mean |SHAP|)")

    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp.index, imp.values, color='#1d428a', edgecolor='white')
    ax.set_xlabel('Mean |SHAP value| (impact on log-salary prediction)')
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('#fafbfc')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Top drivers** are minutes per game (workload/role), age (career stage), and per-36 points (production rate).
    All-Star and MVP-candidate flags carry meaningful weight as proxies for max-contract eligibility.
    """)

# =================================================================
# TAB 4: ABOUT
# =================================================================
with tab4:
    st.markdown("""
    ### About this project

    This dashboard accompanies an IEOR 142A (Spring 2026) project building a fair-market value model for NBA player salaries. The core idea: NBA salaries are not a clean function of on-court production because of CBA-mandated structures (rookie scales, max-contract tiers, supermax eligibility). A regression model trained on production therefore *under-fits salary by design* — and the residuals from that model are themselves a useful signal, identifying which contracts the league has structurally mispriced.

    ### Data
    - **Player stats:** Basketball-Reference 2024-25 regular-season totals
    - **Salaries:** Basketball-Reference 2025-26 contract data
    - **Sample:** 344 players above the $2.3M effective league-minimum cutoff with ≥20 games played

    ### Methodology
    - Five models compared (Linear, Ridge, Lasso, Random Forest, XGBoost)
    - 5-fold cross-validation for honest performance estimates
    - Hyperparameters tuned via GridSearchCV
    - Feature engineering: per-36 stats, age², shot volume, position dummies, award binaries
    - SHAP for global and per-prediction interpretability

    ### Limitations
    - Single season of stats (multi-year rolling averages would improve fit)
    - No defensive impact metrics beyond steals/blocks
    - No contract-context features by design (adding them would defeat the inefficiency-detection purpose)
    - Tight sample (n=344) inflates CV variance

    ### Code & report
    Full notebook, report, and presentation script available in the project repo.
    """)

st.markdown("""
<div class="footer">
    Built for IEOR 142A · Spring 2026 · UC Berkeley · Data: Basketball-Reference
</div>
""", unsafe_allow_html=True)
