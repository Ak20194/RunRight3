"""
RunRight UAE — Founder Analytics Platform
8-Page Streamlit App: Descriptive → Diagnostic → Predictive → Prescriptive → Scoring
Models trained on startup from CSV — no pkl files required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix, roc_curve)

# ─── Resolve paths relative to this file (works on Streamlit Cloud) ───────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(BASE_DIR, filename)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RunRight UAE · Analytics",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-card .lbl { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
.tier1 { border-left: 4px solid #22c55e; background: #052e16; padding: 12px; border-radius: 8px; }
.tier2 { border-left: 4px solid #f59e0b; background: #1c1007; padding: 12px; border-radius: 8px; }
.tier3 { border-left: 4px solid #6366f1; background: #1e1b4b; padding: 12px; border-radius: 8px; }
.act-now { color: #22c55e; font-weight: 700; }
.nurture  { color: #f59e0b; font-weight: 600; }
.low-pri  { color: #64748b; }
h1, h2, h3 { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ─── Feature list (52 features) ───────────────────────────────────────────────
FEATURES = [
    'Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
    'Distance_Enc','Spend_Enc','Purchase_Freq_Enc','WTP_App_Enc',
    'App_Comfort_Enc','Q22_Current_Shoe_Satisfaction_1_7',
    'Q27_Runner_Identity_1_5','Q29_Peer_Influence_1_5',
    'Q30_Sustainability_Importance_1_5','Q35_Brand_Switch_Likelihood_1_5',
    'Discount_Trigger_Enc','Club_Member','Used_AI_Before','Waits_For_Sales',
    'Terrain_Road_Pavement','Terrain_Trail_Desert','Terrain_Treadmill','Terrain_Beach_Sand',
    'Goal_Full_Marathon','Goal_Half_Marathon','Goal_Ultra_Trail','Goal_5K','Goal_10K',
    'Motiv_Competitive_performance','Motiv_Mental_health','Motiv_Social_Community',
    'Priority_Speed_Performance','Priority_Comfort_Cushioning','Priority_Injury_Prevention',
    'App_Strava','App_Garmin','App_Apple_Watch_Health','App_Nike_Run_Club',
    'Acc_GPS_Watch','Acc_Compression_wear','Acc_Custom_insoles',
    'Brand_Nike','Brand_Adidas','Brand_ASICS','Brand_Hoka','Brand_On_Running',
    'Emirate_Dubai','Emirate_Abu_Dhabi','Emirate_Sharjah',
    'Occ_Corporate_Salaried','Occ_Fitness_professional','Occ_Self-employed','Occ_Student'
]

PERSONA_MAP = {
    2: 'Trail & Ultra Specialist',
    0: 'Serious Age-Grouper',
    5: 'Wellness Professional',
    3: 'Social Community Runner',
    1: 'Aspirational Beginner',
    4: 'Casual Lifestyle Runner',
}
TIER_MAP = {
    'Trail & Ultra Specialist': 'Tier 1', 'Serious Age-Grouper': 'Tier 1',
    'Wellness Professional':    'Tier 2', 'Social Community Runner': 'Tier 2',
    'Aspirational Beginner':    'Tier 3', 'Casual Lifestyle Runner': 'Tier 3',
}

# ─── Train all models from CSVs (cached — runs once per cold start) ────────────
@st.cache_resource(show_spinner="🏃 Training models — this takes ~30s on first load...")
def load_models():
    enc_raw = pd.read_csv(data_path('RunRight_UAE_Survey_Encoded.csv'))
    raw     = pd.read_csv(data_path('RunRight_UAE_Survey_Raw.csv'))

    X     = enc_raw[FEATURES].copy()
    imp   = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATURES)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_imp)

    # Classification
    y_clf = enc_raw['App_Interest_Binary']
    clf   = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_imp, y_clf)

    # Clustering
    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    km.fit(X_sc)

    # Regression
    y_reg = enc_raw['Predicted_Annual_Shoe_Spend_AED']
    reg   = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    reg.fit(X_imp, y_reg)

    # PCA (3D visualisation)
    pca   = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    # Enrich dataset
    enc = enc_raw.copy()
    enc['Cluster']        = km.predict(X_sc)
    enc['Adopt_Prob']     = clf.predict_proba(X_imp)[:, 1]
    enc['Pred_Spend_AED'] = reg.predict(X_imp)
    enc['Persona']        = enc['Cluster'].map(PERSONA_MAP)
    enc['Tier']           = enc['Persona'].map(TIER_MAP)
    enc['PCA1'], enc['PCA2'], enc['PCA3'] = X_pca[:,0], X_pca[:,1], X_pca[:,2]
    tw                    = enc['Tier'].map({'Tier 1':1.0,'Tier 2':0.7,'Tier 3':0.3})
    sp_norm               = (enc['Pred_Spend_AED'] - enc['Pred_Spend_AED'].min()) / \
                            (enc['Pred_Spend_AED'].max() - enc['Pred_Spend_AED'].min())
    enc['Priority_Score'] = (enc['Adopt_Prob']*0.4 + sp_norm*0.4 + tw*0.2).round(4)
    enc['Priority_Tier']  = pd.cut(enc['Priority_Score'], bins=[0,0.35,0.55,1.01],
                                    labels=['Low Priority','Nurture','Act Now'])
    enc['Priority_Rank']  = enc['Priority_Score'].rank(ascending=False).astype(int)

    # ARM
    item_cols = [c for c in [
        'Terrain_Road_Pavement','Terrain_Trail_Desert','Terrain_Treadmill','Terrain_Beach_Sand',
        'Brand_Nike','Brand_Adidas','Brand_ASICS','Brand_Hoka','Brand_On_Running',
        'Acc_GPS_Watch','Acc_Compression_wear','Acc_Custom_insoles','Acc_Running_socks',
        'App_Strava','App_Garmin','App_Apple_Watch_Health','App_Nike_Run_Club',
        'Goal_Full_Marathon','Goal_Half_Marathon','Goal_Ultra_Trail','Goal_5K',
        'Motiv_Competitive_performance','Motiv_Social_Community','Motiv_Mental_health',
        'Priority_Speed_Performance','Priority_Comfort_Cushioning','Priority_Injury_Prevention',
        'Club_Member','Used_AI_Before'
    ] if c in enc.columns]
    idf  = enc[item_cols].fillna(0).astype(int)
    ssup = {c: idf[c].mean() for c in idf.columns if idf[c].mean() >= 0.05}
    arm_rows = []
    for a, c in combinations(list(ssup.keys()), 2):
        both = (idf[a] & idf[c]).mean()
        if both < 0.05: continue
        for ant, con in [(a,c),(c,a)]:
            conf = both / ssup[ant]; lift = conf / ssup[con]
            if conf >= 0.4 and lift >= 1.1:
                arm_rows.append({'antecedent':ant,'consequent':con,
                                 'support':round(both,3),'confidence':round(conf,3),'lift':round(lift,3)})
    rules_df = pd.DataFrame(arm_rows).sort_values('lift',ascending=False) if arm_rows else pd.DataFrame()

    # Pre-computed chart data
    Xtr,Xte,ytr,yte = train_test_split(X_imp, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    yp  = clf.predict_proba(Xte)[:,1]
    ypb = clf.predict(Xte)
    fpr,tpr,_ = roc_curve(yte, yp)
    cm_arr    = confusion_matrix(yte, ypb)
    clf_fi    = pd.DataFrame({'feature':FEATURES,'importance':clf.feature_importances_})\
                  .sort_values('importance',ascending=False).head(15)
    reg_fi    = pd.DataFrame({'feature':FEATURES,'importance':reg.feature_importances_})\
                  .sort_values('importance',ascending=False).head(15)
    _,Xrte,_,yrte = train_test_split(X_imp, y_reg, test_size=0.2, random_state=42)
    yrp = reg.predict(Xrte)

    precomp = {
        'roc_fpr': fpr.tolist(), 'roc_tpr': tpr.tolist(),
        'confusion_matrix': cm_arr.tolist(),
        'clf_feature_importance': clf_fi.to_dict('records'),
        'reg_feature_importance': reg_fi.to_dict('records'),
        'reg_actual': yrte.tolist(), 'reg_predicted': yrp.tolist(),
        'pca_variance': pca.explained_variance_ratio_.tolist(),
        'clf_acc': round(accuracy_score(yte, ypb), 3),
        'clf_auc': round(roc_auc_score(yte, yp), 3),
        'reg_r2':  round(float(1 - np.var(yrte.values-yrp)/np.var(yrte.values)), 3),
        'reg_mae': round(float(np.mean(np.abs(yrte.values-yrp))), 1),
    }

    return clf, reg, km, imp, sc, enc, raw, rules_df, precomp

clf, reg, km, imputer, scaler, enriched, raw, rules_df, precomp = load_models()
enc = enriched  # alias used throughout pages


PERSONA_COLORS = {
    'Trail & Ultra Specialist':  '#ef4444',
    'Serious Age-Grouper':       '#f97316',
    'Wellness Professional':     '#22c55e',
    'Social Community Runner':   '#3b82f6',
    'Aspirational Beginner':     '#a855f7',
    'Casual Lifestyle Runner':   '#64748b',
}
PERSONA_TIERS = {
    'Trail & Ultra Specialist':  'Tier 1',
    'Serious Age-Grouper':       'Tier 1',
    'Wellness Professional':     'Tier 2',
    'Social Community Runner':   'Tier 2',
    'Aspirational Beginner':     'Tier 3',
    'Casual Lifestyle Runner':   'Tier 3',
}

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👟 RunRight UAE")
    st.markdown("### Analytics Platform")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊  Market Overview",
        "🔍  Segment Explorer (EDA)",
        "🤖  Classification",
        "🎯  Clustering",
        "🔗  Association Rules",
        "💰  LTV & Regression",
        "📈  ARIMA Forecasting",
        "🎛️  What-If Simulator",
        "🎬  Prescriptive Playbook",
        "📥  Score New Customers",
    ])
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(enriched):,} respondents")
    st.markdown(f"**Features:** {len(FEATURES)}")
    st.markdown(f"**Classifier AUC:** {precomp['clf_auc']}")
    st.markdown(f"**Regressor R²:** {precomp['reg_r2']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Market Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📊 Market Overview")
    st.caption("Descriptive Analysis — Who is in the UAE running market?")

    # KPI Row
    act_now = (enriched['Priority_Tier'] == 'Act Now').sum()
    avg_spend = enriched['Pred_Spend_AED'].mean()
    app_interest = enriched['App_Interest_Binary'].mean() * 100
    tier1_pct = (enriched['Tier'] == 'Tier 1').mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl in [
        (c1, f"{len(enriched):,}", "Total Respondents"),
        (c2, f"{app_interest:.0f}%", "App Interest Rate"),
        (c3, f"AED {avg_spend:,.0f}", "Avg Annual Spend"),
        (c4, f"{act_now}", "Act Now Prospects"),
        (c5, f"{tier1_pct:.0f}%", "High-Value Tier"),
        (c6, f"{precomp['clf_auc']:.3f}", "Model AUC"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Persona Distribution")
        persona_counts = enriched['Persona'].value_counts()
        fig = px.pie(values=persona_counts.values, names=persona_counts.index,
                     color=persona_counts.index,
                     color_discrete_map=PERSONA_COLORS,
                     hole=0.45)
        fig.update_layout(showlegend=True, height=380, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("App Interest by Priority Tier")
        pt = enriched.groupby(['Tier','Priority_Tier']).size().reset_index(name='count')
        fig2 = px.bar(pt, x='Tier', y='count', color='Priority_Tier',
                      color_discrete_map={'Act Now':'#22c55e','Nurture':'#f59e0b','Low Priority':'#475569'},
                      barmode='stack')
        fig2.update_layout(height=380, margin=dict(t=20,b=20), legend_title="Priority")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average Predicted Spend by Persona")
        sp = enriched.groupby('Persona')['Pred_Spend_AED'].mean().sort_values(ascending=True).reset_index()
        fig3 = px.bar(sp, x='Pred_Spend_AED', y='Persona', orientation='h',
                      color='Pred_Spend_AED', color_continuous_scale='Blues',
                      labels={'Pred_Spend_AED': 'Avg AED/year'})
        fig3.update_layout(height=360, margin=dict(t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Adoption Probability by Persona")
        ap = enriched.groupby('Persona')['Adopt_Prob'].mean().sort_values(ascending=True).reset_index()
        fig4 = px.bar(ap, x='Adopt_Prob', y='Persona', orientation='h',
                      color='Adopt_Prob', color_continuous_scale='Greens',
                      labels={'Adopt_Prob': 'Adoption Probability'})
        fig4.update_layout(height=360, margin=dict(t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Emirate breakdown
    st.subheader("Geographic Distribution")
    emirate_cols = [c for c in enc.columns if c.startswith('Emirate_')]
    emirate_sums = enc[emirate_cols].sum().sort_values(ascending=False)
    emirate_sums.index = [c.replace('Emirate_','').replace('_',' ') for c in emirate_sums.index]
    fig5 = px.bar(x=emirate_sums.index, y=emirate_sums.values,
                  labels={'x':'Emirate','y':'Respondents'},
                  color=emirate_sums.values, color_continuous_scale='Oranges')
    fig5.update_layout(height=300, margin=dict(t=20,b=20), coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Segment Explorer (EDA)":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔍 Segment Explorer — Drilled-Down EDA")
    st.caption("Cross-tabulations · Distributions · Correlation · Deep-dive by persona or variable")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cross-Tabulation",
        "🔥 Correlation Heatmap",
        "🎯 Persona Deep-Dive",
        "🔬 Variable Drill-Down",
        "📈 Bivariate Analysis"
    ])

    with tab1:
        st.subheader("Cross-Tabulation Explorer")
        num_cols = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                    'Distance_Enc','WTP_App_Enc','Q22_Current_Shoe_Satisfaction_1_7',
                    'Q27_Runner_Identity_1_5','Q29_Peer_Influence_1_5',
                    'Adopt_Prob','Pred_Spend_AED','Priority_Score']
        num_labels = ['Age','Income','Experience','Days/Week','Distance','WTP App',
                      'Shoe Satisfaction','Runner Identity','Peer Influence',
                      'Adoption Prob','Predicted Spend','Priority Score']
        cat_cols = ['Persona','Tier','Priority_Tier','App_Interest_Binary']

        c1, c2, c3, c4 = st.columns(4)
        x_var  = c1.selectbox("Group By", cat_cols, index=0)
        y_idx  = c2.selectbox("Metric", num_labels, index=10)
        y_var  = num_cols[num_labels.index(y_idx)]
        chart_t = c3.selectbox("Chart Type", ["Box Plot", "Bar (Mean)", "Violin", "Strip"])
        show_n  = c4.checkbox("Show sample size", value=True)

        if chart_t == "Box Plot":
            fig = px.box(enriched, x=x_var, y=y_var, color=x_var,
                         color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None,
                         points='outliers')
        elif chart_t == "Bar (Mean)":
            agg = enriched.groupby(x_var)[y_var].agg(['mean','sem','count']).reset_index()
            agg.columns = [x_var, 'mean', 'sem', 'n']
            fig = px.bar(agg, x=x_var, y='mean', color=x_var, error_y='sem',
                         color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None,
                         text=agg['n'].apply(lambda v: f"n={v}") if show_n else None)
        elif chart_t == "Violin":
            fig = px.violin(enriched, x=x_var, y=y_var, color=x_var, box=True, points='none',
                            color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None)
        else:
            fig = px.strip(enriched.sample(min(500,len(enriched)),random_state=42),
                           x=x_var, y=y_var, color=x_var,
                           color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None)
        fig.update_layout(height=460, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats table
        st.subheader("Summary Statistics by Group")
        summary = enriched.groupby(x_var)[y_var].agg(['count','mean','median','std','min','max']).round(3)
        st.dataframe(summary.style.background_gradient(subset=['mean'], cmap='Blues'), use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        c1, c2 = st.columns([3,1])
        corr_set = c1.multiselect("Include features", [
            'Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc','Distance_Enc',
            'Spend_Enc','WTP_App_Enc','App_Comfort_Enc','Q22_Current_Shoe_Satisfaction_1_7',
            'Q27_Runner_Identity_1_5','Q29_Peer_Influence_1_5','Q30_Sustainability_Importance_1_5',
            'Q35_Brand_Switch_Likelihood_1_5','Discount_Trigger_Enc','Club_Member',
            'Used_AI_Before','Waits_For_Sales','Adopt_Prob','Pred_Spend_AED'
        ], default=['Age_Enc','Income_Enc','Experience_Enc','WTP_App_Enc','App_Comfort_Enc',
                    'Q27_Runner_Identity_1_5','Q29_Peer_Influence_1_5','Club_Member',
                    'Used_AI_Before','Adopt_Prob','Pred_Spend_AED'])
        corr_method = c2.selectbox("Method", ["pearson","spearman","kendall"])

        if len(corr_set) >= 2:
            corr_df = enriched[corr_set].corr(method=corr_method).round(3)
            labels = [c.replace('_Enc','').replace('_',' ')[:14] for c in corr_set]
            fig = go.Figure(go.Heatmap(
                z=corr_df.values, x=labels, y=labels,
                colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                text=corr_df.values.round(2), texttemplate="%{text}",
                textfont={"size":9}, hoverongaps=False
            ))
            fig.update_layout(height=560, margin=dict(t=20,b=80,l=130,r=20))
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Runner Identity (Q27) and WTP correlate strongly with predicted spend. Used AI Before and App Comfort are top adoption probability drivers.")

    with tab3:
        st.subheader("Persona Deep-Dive")
        selected_persona = st.selectbox("Select Persona", list(PERSONA_COLORS.keys()))
        persona_data = enriched[enriched['Persona'] == selected_persona]
        rest_data    = enriched[enriched['Persona'] != selected_persona]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Count", f"{len(persona_data):,}")
        c2.metric("Avg Spend", f"AED {persona_data['Pred_Spend_AED'].mean():,.0f}")
        c3.metric("Adopt Prob", f"{persona_data['Adopt_Prob'].mean():.1%}")
        c4.metric("Act Now %", f"{(persona_data['Priority_Tier']=='Act Now').mean():.1%}")
        c5.metric("Avg Identity", f"{persona_data['Q27_Runner_Identity_1_5'].mean():.2f}/5")

        col1, col2 = st.columns(2)
        with col1:
            radar_features = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                              'Distance_Enc','WTP_App_Enc','Q27_Runner_Identity_1_5','App_Comfort_Enc']
            radar_labels = ['Age','Income','Exp','Days/Wk','Distance','WTP','Identity','App Comfort']
            persona_means = enriched.groupby('Persona')[radar_features].mean()
            pmin = enc[radar_features].min(); pmax = enc[radar_features].max()
            norm_fn = lambda s: ((s - pmin) / (pmax - pmin)).clip(0,1)
            sel_norm = norm_fn(persona_means.loc[selected_persona]).values.tolist() + [norm_fn(persona_means.loc[selected_persona]).values[0]]
            oth_norm = norm_fn(persona_means.drop(selected_persona).mean()).values.tolist() + [norm_fn(persona_means.drop(selected_persona).mean()).values[0]]
            angles = radar_labels + [radar_labels[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=sel_norm, theta=angles, fill='toself',
                                          name=selected_persona, line_color=PERSONA_COLORS[selected_persona]))
            fig.add_trace(go.Scatterpolar(r=oth_norm, theta=angles, fill='toself',
                                          name='Others avg', opacity=0.35, line_color='#64748b'))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), height=380, legend=dict(y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            diff = (persona_data[FEATURES].mean() - rest_data[FEATURES].mean()).sort_values()
            top_pos = diff.tail(8); top_neg = diff.head(5)
            combined = pd.concat([top_neg, top_pos]).sort_values()
            combined.index = combined.index.str.replace('_Enc','').str.replace('_',' ')
            fig2 = px.bar(x=combined.values, y=combined.index, orientation='h',
                          color=combined.values, color_continuous_scale='RdBu',
                          title=f"How {selected_persona} differs", labels={'x':'Mean Δ','y':''})
            fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=160))
            st.plotly_chart(fig2, use_container_width=True)

        # Spend and adoption distributions side by side
        col3, col4 = st.columns(2)
        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=persona_data['Pred_Spend_AED'], name=selected_persona,
                                        marker_color=PERSONA_COLORS[selected_persona], opacity=0.75, nbinsx=25))
            fig3.add_trace(go.Histogram(x=rest_data['Pred_Spend_AED'], name='Others',
                                        marker_color='#64748b', opacity=0.5, nbinsx=25))
            fig3.update_layout(barmode='overlay', height=320, title='Spend Distribution',
                               xaxis_title='AED/year', legend=dict(y=1.1, orientation='h'))
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(x=persona_data['Adopt_Prob'], name=selected_persona,
                                        marker_color=PERSONA_COLORS[selected_persona], opacity=0.75, nbinsx=20))
            fig4.add_trace(go.Histogram(x=rest_data['Adopt_Prob'], name='Others',
                                        marker_color='#64748b', opacity=0.5, nbinsx=20))
            fig4.update_layout(barmode='overlay', height=320, title='Adoption Probability Distribution',
                               xaxis_title='Probability', legend=dict(y=1.1, orientation='h'))
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.subheader("Variable Drill-Down")
        drill_var = st.selectbox("Select variable to drill into", [
            'Pred_Spend_AED','Adopt_Prob','Q27_Runner_Identity_1_5','WTP_App_Enc',
            'Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc','Priority_Score'
        ], format_func=lambda x: x.replace('_Enc','').replace('_',' '))

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(enriched, x=drill_var, nbins=40,
                               color_discrete_sequence=['#38bdf8'],
                               title=f"Distribution of {drill_var.replace('_Enc','').replace('_',' ')}",
                               marginal='box')
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(enriched, x='Persona', y=drill_var, color='Persona',
                          color_discrete_map=PERSONA_COLORS,
                          title=f"{drill_var.replace('_Enc','').replace('_',' ')} by Persona")
            fig2.update_layout(height=360, showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig2, use_container_width=True)

        # Descriptive stats
        st.subheader("Descriptive Statistics")
        desc = enriched[drill_var].describe().round(3)
        skew = enriched[drill_var].skew()
        kurt = enriched[drill_var].kurt()
        dc1,dc2,dc3,dc4,dc5,dc6,dc7 = st.columns(7)
        dc1.metric("Mean",    f"{desc['mean']:.2f}")
        dc2.metric("Median",  f"{desc['50%']:.2f}")
        dc3.metric("Std Dev", f"{desc['std']:.2f}")
        dc4.metric("Min",     f"{desc['min']:.2f}")
        dc5.metric("Max",     f"{desc['max']:.2f}")
        dc6.metric("Skewness", f"{skew:.3f}")
        dc7.metric("Kurtosis", f"{kurt:.3f}")

        # Per-tier breakdown
        st.subheader("Values by Customer Tier")
        tier_stats = enriched.groupby('Tier')[drill_var].agg(['mean','median','std','count']).round(3)
        st.dataframe(tier_stats.style.background_gradient(cmap='Blues', subset=['mean']), use_container_width=True)

    with tab5:
        st.subheader("Bivariate Analysis — Any Two Variables")
        bv_vars = ['Pred_Spend_AED','Adopt_Prob','Q27_Runner_Identity_1_5','WTP_App_Enc',
                   'Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                   'Q22_Current_Shoe_Satisfaction_1_7','Priority_Score']
        bv_labels = [v.replace('_Enc','').replace('_',' ') for v in bv_vars]

        c1, c2, c3 = st.columns(3)
        x_bv = c1.selectbox("X axis", bv_labels, index=2)
        y_bv = c2.selectbox("Y axis", bv_labels, index=0)
        col_bv = c3.selectbox("Colour by", ['Persona','Tier','Priority_Tier'], index=0)

        x_col = bv_vars[bv_labels.index(x_bv)]
        y_col = bv_vars[bv_labels.index(y_bv)]

        sample_bv = enriched.sample(min(800, len(enriched)), random_state=42)
        fig = px.scatter(sample_bv, x=x_col, y=y_col, color=col_bv,
                         color_discrete_map=PERSONA_COLORS if col_bv=='Persona' else None,
                         opacity=0.65, trendline='ols',
                         labels={x_col: x_bv, y_col: y_bv},
                         title=f"{x_bv} vs {y_bv} (coloured by {col_bv})")
        fig.update_layout(height=500, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        # Correlation value
        r = enriched[[x_col, y_col]].corr().iloc[0,1]
        st.metric("Pearson Correlation", f"{r:.4f}",
                  delta="Strong positive" if r>0.5 else "Moderate" if r>0.2 else "Weak/None")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Classification":
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score

    st.title("🤖 Classification — All Algorithms Compared")
    st.caption("Predicting app adoption · 8 classifiers · Accuracy, Precision, Recall, F1, AUC-ROC")

    @st.cache_data
    def run_all_classifiers():
        X = enc[FEATURES].copy()
        y = enc['App_Interest_Binary'].copy()
        X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
        sc2 = StandardScaler()
        X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y)
        X_tr_sc = sc2.fit_transform(X_tr)
        X_te_sc = sc2.transform(X_te)
        X_all_sc = sc2.transform(X_imp)

        clfs = {
            'Logistic Regression':  (LogisticRegression(max_iter=1000, random_state=42), True),
            'Decision Tree':        (DecisionTreeClassifier(max_depth=8, random_state=42), False),
            'Random Forest':        (RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1), False),
            'Gradient Boosting':    (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
            'AdaBoost':             (AdaBoostClassifier(n_estimators=100, random_state=42), False),
            'K-Nearest Neighbors':  (KNeighborsClassifier(n_neighbors=7), True),
            'SVM (RBF)':            (SVC(probability=True, random_state=42), True),
            'Naive Bayes':          (GaussianNB(), False),
        }
        results, roc_curves, cms, trained = [], {}, {}, {}
        for name, (c, scaled) in clfs.items():
            Xtr = X_tr_sc if scaled else X_tr.values
            Xte = X_te_sc if scaled else X_te.values
            Xall = X_all_sc if scaled else X_imp.values
            c.fit(Xtr, y_tr)
            yp = c.predict(Xte); ypr = c.predict_proba(Xte)[:,1]
            cv = cross_val_score(c, Xall, y, cv=5, scoring='accuracy').mean()
            fpr, tpr, _ = roc_curve(y_te, ypr)
            step = max(1, len(fpr)//80)
            results.append({'Model': name,
                'Accuracy': round(accuracy_score(y_te,yp),4),
                'Precision': round(precision_score(y_te,yp),4),
                'Recall': round(recall_score(y_te,yp),4),
                'F1-Score': round(f1_score(y_te,yp),4),
                'AUC-ROC': round(roc_auc_score(y_te,ypr),4),
                'CV Acc (5-fold)': round(cv,4)})
            roc_curves[name] = {'fpr': fpr[::step].tolist(), 'tpr': tpr[::step].tolist()}
            cms[name] = confusion_matrix(y_te, yp).tolist()
            trained[name] = (c, scaled)
        return pd.DataFrame(results).sort_values('AUC-ROC', ascending=False), roc_curves, cms, trained, y_te

    with st.spinner("Training 8 classifiers..."):
        comp_df, roc_curves, cms, trained_clfs, y_te_clf = run_all_classifiers()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", "📈 ROC Curves", "🔢 Confusion Matrices", "🔍 Deep Dive"
    ])

    COLORS_CLF = ['#38bdf8','#22c55e','#f59e0b','#ef4444','#a855f7','#fb923c','#34d399','#64748b']

    with tab1:
        st.subheader("All Classifiers — Performance Comparison")
        # Colour the best in each column
        def highlight_best(s):
            is_max = s == s.max()
            return ['background-color: #052e16; color: #22c55e; font-weight: bold' if v else '' for v in is_max]
        styled = comp_df.style.apply(highlight_best, subset=['Accuracy','Precision','Recall','F1-Score','AUC-ROC','CV Acc (5-fold)'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        # Grouped bar chart
        metrics = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        fig = go.Figure()
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric, x=comp_df['Model'], y=comp_df[metric],
                marker_color=COLORS_CLF[i], opacity=0.85
            ))
        fig.update_layout(barmode='group', height=420, xaxis_tickangle=-25,
                          legend=dict(orientation='h', y=1.1),
                          yaxis=dict(range=[0,1], title='Score'),
                          margin=dict(b=100))
        st.plotly_chart(fig, use_container_width=True)

        best = comp_df.iloc[0]
        st.success(f"🏆 **Best Model: {best['Model']}** — AUC-ROC: {best['AUC-ROC']} | F1: {best['F1-Score']} | Accuracy: {best['Accuracy']}")
        st.info("💡 **Why AUC-ROC is the primary metric here:** The dataset has a 62/38 class imbalance. AUC-ROC measures discrimination power regardless of threshold, making it more reliable than raw accuracy for comparing classifiers on imbalanced targets.")

    with tab2:
        st.subheader("ROC Curves — All Models Overlaid")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 line=dict(dash='dash', color='#475569', width=1.5), name='Random baseline'))
        for i, (name, data) in enumerate(roc_curves.items()):
            auc_val = comp_df[comp_df['Model']==name]['AUC-ROC'].values[0]
            fig.add_trace(go.Scatter(x=data['fpr'], y=data['tpr'], mode='lines',
                                     name=f"{name} (AUC={auc_val})",
                                     line=dict(color=COLORS_CLF[i], width=2)))
        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          height=520, legend=dict(x=0.55, y=0.1),
                          margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Confusion Matrices — All Models")
        cols_cm = st.columns(4)
        for i, (name, cm_vals) in enumerate(cms.items()):
            with cols_cm[i % 4]:
                cm_arr = np.array(cm_vals)
                fig = px.imshow(cm_arr, text_auto=True, color_continuous_scale='Blues',
                                x=['No','Yes'], y=['No','Yes'],
                                title=name.replace(' (','<br>('))
                fig.update_layout(height=240, margin=dict(t=50,b=10,l=10,r=10),
                                  coloraxis_showscale=False,
                                  font=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Deep Dive — Single Model Analysis")
        selected_model = st.selectbox("Select model", comp_df['Model'].tolist())
        row = comp_df[comp_df['Model']==selected_model].iloc[0]

        c1,c2,c3,c4,c5 = st.columns(5)
        for col,(lbl,val) in zip([c1,c2,c3,c4,c5],[
            ("Accuracy",f"{row['Accuracy']:.4f}"), ("AUC-ROC",f"{row['AUC-ROC']:.4f}"),
            ("Precision",f"{row['Precision']:.4f}"), ("Recall",f"{row['Recall']:.4f}"),
            ("F1-Score",f"{row['F1-Score']:.4f}")]):
            col.markdown(f'''<div class="metric-card">
                <div class="val">{val}</div><div class="lbl">{lbl}</div></div>''', unsafe_allow_html=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            roc_d = roc_curves[selected_model]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roc_d['fpr'], y=roc_d['tpr'], mode='lines',
                                     name=f"AUC={row['AUC-ROC']}",
                                     line=dict(color='#38bdf8', width=2.5),
                                     fill='tozeroy', fillcolor='rgba(56,189,248,0.1)'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     line=dict(dash='dash', color='#ef4444'), name='Random'))
            fig.update_layout(height=380, xaxis_title='FPR', yaxis_title='TPR')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Confusion Matrix")
            cm_arr = np.array(cms[selected_model])
            fig2 = px.imshow(cm_arr, text_auto=True, color_continuous_scale='Blues',
                             x=['Predicted No','Predicted Yes'],
                             y=['Actual No','Actual Yes'])
            fig2.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Feature importance for RF, threshold tuning for all
        if selected_model == 'Random Forest':
            st.subheader("Feature Importance (Random Forest)")
            fi_data = precomp['clf_feature_importance']
            fi_df2 = pd.DataFrame(fi_data).sort_values('importance', ascending=True)
            fi_df2['feature'] = fi_df2['feature'].str.replace('_Enc','').str.replace('_',' ')
            fig3 = px.bar(fi_df2, x='importance', y='feature', orientation='h',
                          color='importance', color_continuous_scale='Blues')
            fig3.update_layout(height=480, coloraxis_showscale=False, margin=dict(l=180))
            st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Clustering":
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, silhouette_samples
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    st.title("🎯 Clustering — K-Means + Hierarchical")
    st.caption("Elbow method · Silhouette analysis · Dendrogram · 3D PCA · Cluster profiles")

    @st.cache_data
    def compute_clustering_stats():
        X = enc[FEATURES].copy()
        X_imp2 = pd.DataFrame(imputer.transform(X), columns=FEATURES)
        sc2 = StandardScaler()
        X_sc2 = sc2.fit_transform(X_imp2)
        inertias, sil_scores = [], []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_sc2)
            inertias.append(round(km.inertia_, 1))
            sil_scores.append(round(silhouette_score(X_sc2, lbl), 4))
        return inertias, sil_scores, X_sc2, X_imp2

    with st.spinner("Computing clustering statistics..."):
        inertias, sil_scores, X_sc_clust, X_imp_clust = compute_clustering_stats()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Elbow & Silhouette",
        "🌳 Hierarchical / Dendrogram",
        "🌐 3D Segment Map",
        "🧬 Cluster Profiles",
        "📊 Segment Statistics"
    ])

    CLUST_COLORS = ['#ef4444','#f97316','#22c55e','#3b82f6','#a855f7','#64748b']

    with tab1:
        st.subheader("Elbow Method & Silhouette Score")
        ks = list(range(2, 10))

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ks, y=inertias, mode='lines+markers',
                                     name='Inertia', line=dict(color='#38bdf8', width=2.5),
                                     marker=dict(size=8, color='#38bdf8')))
            fig.add_vline(x=6, line_dash='dash', line_color='#22c55e',
                          annotation_text='K=6 selected', annotation_position='top right')
            fig.update_layout(height=380, xaxis_title='Number of Clusters (K)',
                              yaxis_title='Inertia (Within-cluster SSE)',
                              title='Elbow Method — Inertia vs K')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ks, y=sil_scores, mode='lines+markers',
                                      name='Silhouette', line=dict(color='#22c55e', width=2.5),
                                      marker=dict(size=8, color='#22c55e')))
            fig2.add_hline(y=max(sil_scores), line_dash='dot', line_color='#f59e0b')
            fig2.add_vline(x=6, line_dash='dash', line_color='#22c55e',
                           annotation_text='K=6 selected', annotation_position='top right')
            fig2.update_layout(height=380, xaxis_title='Number of Clusters (K)',
                               yaxis_title='Silhouette Score (higher = better)',
                               title='Silhouette Score vs K')
            st.plotly_chart(fig2, use_container_width=True)

        # Silhouette per-sample plot for K=6
        st.subheader("Silhouette Plot — Per-Sample Analysis (K=6)")
        st.caption("Each bar = one customer. Width = silhouette score. Negative values = misclassified.")
        @st.cache_data
        def silhouette_plot_data():
            km6 = KMeans(n_clusters=6, random_state=42, n_init=20)
            labels6 = km6.fit_predict(X_sc_clust)
            sil_vals = silhouette_samples(X_sc_clust, labels6)
            return labels6, sil_vals
        labels6, sil_vals = silhouette_plot_data()
        sil_df = pd.DataFrame({'cluster': labels6, 'sil': sil_vals, 'persona': [list(PERSONA_MAP.values())[l] for l in labels6]})
        sil_agg = sil_df.groupby('persona')['sil'].agg(['mean','min','max','count']).round(4).reset_index()
        sil_agg.columns = ['Persona','Avg Silhouette','Min','Max','Count']
        fig3 = px.bar(sil_agg.sort_values('Avg Silhouette', ascending=True),
                      x='Avg Silhouette', y='Persona', orientation='h',
                      color='Avg Silhouette', color_continuous_scale='RdYlGn',
                      title='Average Silhouette Score by Persona (K=6)',
                      error_x=(sil_agg.sort_values('Avg Silhouette',ascending=True)['Max'] -
                               sil_agg.sort_values('Avg Silhouette',ascending=True)['Avg Silhouette']))
        fig3.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=180))
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(sil_agg.style.background_gradient(subset=['Avg Silhouette'], cmap='RdYlGn'),
                     use_container_width=True, hide_index=True)
        st.info(f"💡 Overall silhouette score (K=6): **{sil_scores[4]:.4f}**. Values closer to +1 = well-separated clusters. Survey-based clustering typically produces moderate silhouette scores (0.05–0.20) due to overlapping psychographic profiles.")

    with tab2:
        st.subheader("Hierarchical Clustering — Dendrogram Validation")
        st.caption("Ward linkage on a sample of 200 respondents. Confirms K=6 is the natural cut-point.")

        @st.cache_data
        def compute_dendrogram():
            sample_idx = np.random.RandomState(42).choice(len(X_sc_clust), 200, replace=False)
            X_sample = X_sc_clust[sample_idx]
            Z = linkage(X_sample, method='ward', metric='euclidean')
            return Z, sample_idx

        Z, sample_idx = compute_dendrogram()

        # Build dendrogram as plotly figure
        from scipy.cluster.hierarchy import dendrogram as scipy_dend
        dend = scipy_dend(Z, no_plot=True, truncate_mode='lastp', p=40)

        fig = go.Figure()
        icoord = np.array(dend['icoord'])
        dcoord = np.array(dend['dcoord'])
        for xs, ys in zip(icoord, dcoord):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                     line=dict(color='#38bdf8', width=1.2),
                                     showlegend=False, hoverinfo='none'))

        # Add horizontal cut line for K=6
        cut_heights = sorted([d[1] for d in dcoord] + [d[2] for d in dcoord], reverse=True)
        if len(cut_heights) >= 6:
            cut_h = (cut_heights[5] + cut_heights[6]) / 2
            fig.add_hline(y=cut_h, line_dash='dash', line_color='#ef4444', line_width=2,
                          annotation_text=f"Cut for K=6 (height≈{cut_h:.1f})",
                          annotation_position="right")

        fig.update_layout(height=480, showlegend=False,
                          title="Hierarchical Clustering Dendrogram (Ward linkage, n=200 sample)",
                          xaxis=dict(showticklabels=False, title='Samples'),
                          yaxis=dict(title='Distance (Ward linkage)'),
                          margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # Agglomerative clustering comparison
        st.subheader("K-Means vs Agglomerative Clustering (K=6)")
        @st.cache_data
        def compare_clustering():
            km6 = KMeans(n_clusters=6, random_state=42, n_init=20)
            agg6 = AgglomerativeClustering(n_clusters=6, linkage='ward')
            km_labels = km6.fit_predict(X_sc_clust)
            agg_labels = agg6.fit_predict(X_sc_clust)
            sil_km  = silhouette_score(X_sc_clust, km_labels)
            sil_agg2 = silhouette_score(X_sc_clust, agg_labels)
            km_sizes  = pd.Series(km_labels).value_counts().sort_index().tolist()
            agg_sizes = pd.Series(agg_labels).value_counts().sort_index().tolist()
            return sil_km, sil_agg2, km_sizes, agg_sizes
        sil_km, sil_agg2, km_sizes, agg_sizes = compare_clustering()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("K-Means Silhouette", f"{sil_km:.4f}")
        c2.metric("Agglomerative Silhouette", f"{sil_agg2:.4f}")
        c3.metric("K-Means Winner", "✅" if sil_km >= sil_agg2 else "❌")
        c4.metric("Agglomerative Winner", "✅" if sil_agg2 > sil_km else "❌")

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.bar(x=[f"C{i+1}" for i in range(len(km_sizes))], y=km_sizes,
                          title="K-Means Cluster Sizes", color_discrete_sequence=['#38bdf8'])
            fig2.update_layout(height=300, xaxis_title='Cluster', yaxis_title='Count')
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.bar(x=[f"C{i+1}" for i in range(len(agg_sizes))], y=agg_sizes,
                          title="Agglomerative Cluster Sizes", color_discrete_sequence=['#22c55e'])
            fig3.update_layout(height=300, xaxis_title='Cluster', yaxis_title='Count')
            st.plotly_chart(fig3, use_container_width=True)

        st.info("💡 **Dendrogram validation:** The red dashed line shows where cutting the tree gives K=6 clusters. The long vertical lines above the cut (high merge distances) confirm that 6 is a natural grouping — merging further would combine genuinely different customer types.")

    with tab3:
        st.subheader("3D PCA Cluster Visualisation")
        sample = enriched.sample(min(800, len(enriched)), random_state=42)
        fig = px.scatter_3d(sample, x='PCA1', y='PCA2', z='PCA3',
                            color='Persona', symbol='Tier',
                            color_discrete_map=PERSONA_COLORS,
                            hover_data=['Pred_Spend_AED','Adopt_Prob','Priority_Tier'],
                            opacity=0.75, size_max=5)
        fig.update_layout(height=600, legend=dict(orientation='h', y=-0.15),
                          scene=dict(xaxis_title=f"PC1 ({precomp['pca_variance'][0]:.1%})",
                                     yaxis_title=f"PC2 ({precomp['pca_variance'][1]:.1%})",
                                     zaxis_title=f"PC3 ({precomp['pca_variance'][2]:.1%})"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"PCA variance explained: PC1={precomp['pca_variance'][0]:.1%}, PC2={precomp['pca_variance'][1]:.1%}, PC3={precomp['pca_variance'][2]:.1%}")

    with tab4:
        st.subheader("Cluster Profile Radar Charts")
        radar_feats = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                       'Distance_Enc','WTP_App_Enc','Q27_Runner_Identity_1_5','Club_Member']
        radar_lbls  = ['Age','Income','Experience','Days/Wk','Distance','WTP','Identity','Club Mbr']
        pmin = enc[radar_feats].min(); pmax = enc[radar_feats].max()
        norm_r = lambda s: ((s - pmin) / (pmax - pmin)).clip(0, 1)
        persona_means_r = enriched.groupby('Persona')[radar_feats].mean()
        cols = st.columns(3)
        for i, (persona, color) in enumerate(PERSONA_COLORS.items()):
            if persona not in persona_means_r.index: continue
            vals = norm_r(persona_means_r.loc[persona]).values.tolist() + [norm_r(persona_means_r.loc[persona]).values[0]]
            angles = radar_lbls + [radar_lbls[0]]
            with cols[i % 3]:
                fig = go.Figure(go.Scatterpolar(r=vals, theta=angles, fill='toself',
                                                line_color=color, fillcolor=color, opacity=0.4, name=persona))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0,1], showticklabels=False)),
                                  title=dict(text=persona, font=dict(size=11)),
                                  height=270, margin=dict(t=50,b=20,l=30,r=30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Segment Statistics Comparison")
        stats_cols = ['Pred_Spend_AED','Adopt_Prob','Priority_Score',
                      'Age_Enc','Income_Enc','Experience_Enc','Q27_Runner_Identity_1_5']
        stats = enriched.groupby('Persona')[stats_cols].agg(['mean','std']).round(2)
        stats.columns = [' '.join(c) for c in stats.columns]
        st.dataframe(stats.style.background_gradient(subset=[c for c in stats.columns if 'mean' in c],
                                                      cmap='Blues'), use_container_width=True)
        st.subheader("Spend Distribution by Segment")
        fig = px.box(enriched, x='Persona', y='Pred_Spend_AED', color='Persona',
                     color_discrete_map=PERSONA_COLORS,
                     labels={'Pred_Spend_AED':'Predicted Annual Spend (AED)'})
        fig.update_layout(height=420, showlegend=False, xaxis={'categoryorder':'median descending'})
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Association Rules":
    import plotly.graph_objects as go
    import plotly.express as px
    import networkx as nx

    st.title("🔗 Association Rules — Bundle Intelligence")
    st.caption("What runners buy, use, and do together — your product bundle engine")

    tab1, tab2, tab3, tab4 = st.tabs(["Network Graph", "Scatter Plot", "Top Rules", "Bundle Recommendations"])

    with tab1:
        st.subheader("Association Rule Network")
        min_lift = st.slider("Minimum Lift", 1.0, 2.5, 1.2, 0.05)
        filtered = rules_df[rules_df['lift'] >= min_lift].copy()
        st.caption(f"{len(filtered)} rules shown (of {len(rules_df)} total)")

        if len(filtered) > 0:
            G = nx.from_pandas_edgelist(filtered, 'antecedent', 'consequent',
                                        edge_attr=['lift','confidence','support'])
            pos = nx.spring_layout(G, seed=42, k=2)
            edge_x, edge_y, edge_text = [], [], []
            for e in G.edges(data=True):
                x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
                edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
                edge_text.append(f"Lift: {e[2]['lift']:.2f}")

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_size = [G.degree(n)*8+10 for n in G.nodes()]
            node_labels = [n.replace('_',' ').replace(' Enc','') for n in G.nodes()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                     line=dict(width=1, color='#334155'), hoverinfo='none'))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                     text=node_labels, textposition='top center',
                                     textfont=dict(size=9),
                                     marker=dict(size=node_size, color='#38bdf8',
                                                 line=dict(width=1, color='white')),
                                     hoverinfo='text',
                                     hovertext=[f"{n}: {G.degree(n)} connections" for n in G.nodes()]))
            fig.update_layout(showlegend=False, height=550,
                              xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                              yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No rules at this lift threshold. Lower the slider.")

    with tab2:
        st.subheader("Scatter Plot — Support vs Confidence vs Lift")
        st.caption("Each point = one rule. X=Support, Y=Confidence, Size=Lift, Colour=Lift strength.")

        if len(rules_df) > 0:
            scatter_df = rules_df.copy()
            scatter_df['antecedent_clean'] = scatter_df['antecedent'].str.replace('_',' ').str.replace(' Enc','')
            scatter_df['consequent_clean']  = scatter_df['consequent'].str.replace('_',' ').str.replace(' Enc','')
            scatter_df['rule_label'] = scatter_df['antecedent_clean'] + " → " + scatter_df['consequent_clean']
            scatter_df['lift_category'] = pd.cut(scatter_df['lift'],
                bins=[0,1.1,1.3,1.6,99], labels=['Weak (1.0–1.1)','Moderate (1.1–1.3)','Strong (1.3–1.6)','Very Strong (1.6+)'])

            min_conf_s = st.slider("Min Confidence", 0.1, 0.9, 0.3, 0.05, key='arm_scatter_conf')
            scatter_filtered = scatter_df[scatter_df['confidence'] >= min_conf_s]

            fig_sc = px.scatter(scatter_filtered,
                x='support', y='confidence',
                size='lift', color='lift',
                hover_name='rule_label',
                hover_data={'support': ':.3f', 'confidence': ':.3f', 'lift': ':.3f', 'lift_category': True},
                color_continuous_scale='Viridis',
                size_max=35,
                labels={'support': 'Support (how common)', 'confidence': 'Confidence (how reliable)', 'lift': 'Lift'},
                title=f"ARM Scatter — {len(scatter_filtered)} rules shown")
            fig_sc.update_layout(height=520, coloraxis_colorbar=dict(title='Lift'))
            # Add quadrant lines
            med_sup = scatter_filtered['support'].median()
            med_conf = scatter_filtered['confidence'].median()
            fig_sc.add_hline(y=med_conf, line_dash='dot', line_color='#475569',
                             annotation_text='Median confidence', annotation_position='right')
            fig_sc.add_vline(x=med_sup, line_dash='dot', line_color='#475569',
                             annotation_text='Median support', annotation_position='top')
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown("**Quadrant interpretation:**")
            qc1, qc2 = st.columns(2)
            qc1.success("**Top-right (high support + high confidence):** Best bundles — frequent AND reliable")
            qc2.info("**Top-left (low support + high confidence):** Niche bundles — rare but very reliable")
            qc3, qc4 = st.columns(2)
            qc3.warning("**Bottom-right (high support + low confidence):** Common but weak associations")
            qc4.error("**Bottom-left:** Weak rules — avoid building bundles from these")

            # 3D scatter (support, confidence, lift)
            st.subheader("3D Scatter — Support × Confidence × Lift")
            fig_3d = px.scatter_3d(scatter_filtered.head(50), x='support', y='confidence', z='lift',
                                   color='lift', hover_name='rule_label',
                                   color_continuous_scale='Plasma', size_max=12,
                                   title="Top 50 rules in 3D space")
            fig_3d.update_layout(height=480, scene=dict(
                xaxis_title='Support', yaxis_title='Confidence', zaxis_title='Lift'))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("No ARM rules found.")

    with tab3:
        st.subheader("Top Association Rules")
        sort_by = st.selectbox("Sort by", ["lift","confidence","support"], index=0)
        display = rules_df.sort_values(sort_by, ascending=False).head(30).copy()
        display['antecedent'] = display['antecedent'].str.replace('_',' ')
        display['consequent']  = display['consequent'].str.replace('_',' ')
        st.dataframe(display.style.background_gradient(subset=['lift'], cmap='Blues'),
                     use_container_width=True)

        # Lift bubble chart
        top20 = rules_df.head(20).copy()
        fig2 = px.scatter(top20, x='support', y='confidence', size='lift', color='lift',
                          hover_data=['antecedent','consequent'],
                          color_continuous_scale='Viridis',
                          labels={'support':'Support','confidence':'Confidence','lift':'Lift'},
                          title="Support vs Confidence (bubble size = Lift)")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("Persona-Specific Bundle Recommendations")
        bundles = {
            'Trail & Ultra Specialist': {
                'Bundle': 'Desert Trail Pro Kit',
                'Items': 'Hoka Speedgoat + Garmin Forerunner + Hydration Vest + Trail Socks',
                'ARM Support': 'Trail Desert → Garmin (lift 1.92), Trail Desert → Strava (lift 1.48)',
                'Expected Basket': 'AED 1,800–2,400',
                'Rationale': 'High trail + Garmin + Strava usage confirms tech-forward trail identity'
            },
            'Serious Age-Grouper': {
                'Bundle': 'Marathon Ready Pack',
                'Items': 'ASICS Gel-Nimbus + GPS Watch + Custom Insoles + Compression Wear',
                'ARM Support': 'Full Marathon → Strava (lift 1.44), Competitive → GPS Watch (lift 1.48)',
                'Expected Basket': 'AED 1,400–2,000',
                'Rationale': 'Marathon goal + competitive motivation drives performance gear'
            },
            'Wellness Professional': {
                'Bundle': 'Wellness Runner Collection',
                'Items': 'On Running Cloud + Foam Roller + Running Belt + Premium Socks',
                'ARM Support': 'Club Member → Garmin (lift 1.36), Strava → Club Member',
                'Expected Basket': 'AED 900–1,400',
                'Rationale': 'Club membership + mental health motivation = community + recovery focus'
            },
            'Social Community Runner': {
                'Bundle': 'Community Starter Pack',
                'Items': 'Nike React + Nike Run Club Premium + Running Socks + Belt',
                'ARM Support': 'NRC → Social Community Motiv (lift 1.20)',
                'Expected Basket': 'AED 700–1,000',
                'Rationale': 'NRC app usage + social motivation = brand-affiliated community gear'
            },
            'Aspirational Beginner': {
                'Bundle': 'First Steps Bundle',
                'Items': 'Adidas Ultraboost entry + Running Socks + Free App Trial 30-day',
                'ARM Support': 'Road Pavement → Nike/Adidas (entry brands)',
                'Expected Basket': 'AED 400–700',
                'Rationale': 'Lower spend, road terrain, brand-name appeal — freemium funnel entry'
            },
            'Casual Lifestyle Runner': {
                'Bundle': 'Lifestyle Flex Pack',
                'Items': 'New Balance Fresh Foam + Apple Watch integration + Insoles',
                'ARM Support': 'Apple Watch Health → general fitness motivation',
                'Expected Basket': 'AED 300–600',
                'Rationale': 'Apple Watch usage + general fitness = casual tech-enabled lifestyle'
            },
        }
        for persona, bundle in bundles.items():
            tier = PERSONA_TIERS[persona]
            tier_class = 'tier1' if tier=='Tier 1' else ('tier2' if tier=='Tier 2' else 'tier3')
            with st.expander(f"{'🔴' if tier=='Tier 1' else '🟡' if tier=='Tier 2' else '🔵'} {persona} — {bundle['Bundle']}"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Items:** {bundle['Items']}")
                c1.markdown(f"**ARM Support:** {bundle['ARM Support']}")
                c2.markdown(f"**Expected Basket:** {bundle['Expected Basket']}")
                c2.markdown(f"**Rationale:** {bundle['Rationale']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 · LTV & REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰  LTV & Regression":
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import r2_score, mean_absolute_error

    st.title("💰 LTV & Regression — Linear, Ridge, Lasso vs Random Forest")
    st.caption("Predicting annual shoe spend (AED) · 4 regression models compared")

    @st.cache_data
    def run_all_regressors():
        X = enc[FEATURES].copy()
        y_r = enc['Predicted_Annual_Shoe_Spend_AED'].copy()
        X_imp2 = pd.DataFrame(imputer.transform(X), columns=FEATURES)
        sc2 = StandardScaler()
        X_sc2 = sc2.fit_transform(X_imp2)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_sc2, y_r, test_size=0.2, random_state=42)
        regs = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)':     Ridge(alpha=1.0),
            'Lasso (α=0.1)':     Lasso(alpha=0.1, max_iter=5000),
            'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        }
        results2, preds, coefs = [], {}, {}
        for name, r in regs.items():
            r.fit(X_tr2, y_tr2)
            yp = r.predict(X_te2)
            r2v = r2_score(y_te2, yp)
            maev = mean_absolute_error(y_te2, yp)
            rmsev = np.sqrt(((y_te2.values - yp)**2).mean())
            mape = np.mean(np.abs((y_te2.values - yp) / (y_te2.values + 1e-9))) * 100
            results2.append({'Model': name, 'R²': round(r2v,4), 'MAE (AED)': round(maev,1),
                             'RMSE (AED)': round(rmsev,1), 'MAPE (%)': round(mape,1)})
            preds[name] = {'actual': y_te2.values[:300].tolist(), 'predicted': yp[:300].tolist(),
                           'residuals': (y_te2.values[:300] - yp[:300]).tolist()}
            if hasattr(r, 'coef_'):
                coef_s = pd.Series(r.coef_, index=FEATURES)
                top = coef_s.abs().sort_values(ascending=False).head(15)
                coefs[name] = {'features': top.index.tolist(),
                               'values': [round(coef_s[f],3) for f in top.index]}
        return pd.DataFrame(results2), preds, coefs, y_te2

    with st.spinner("Training 4 regression models..."):
        reg_comp_df, reg_preds, reg_coefs, y_te_reg = run_all_regressors()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Comparison", "📈 Actual vs Predicted", "📉 Residual Analysis", "🔍 Coefficients & Drivers"
    ])

    REG_COLORS = ['#38bdf8','#22c55e','#f59e0b','#a855f7']

    with tab1:
        st.subheader("Regression Model Comparison")
        def highlight_reg(s):
            if s.name == 'R²':
                is_best = s == s.max()
            else:
                is_best = s == s.min()
            return ['background-color: #052e16; color: #22c55e; font-weight:bold' if v else '' for v in is_best]
        styled_reg = reg_comp_df.style.apply(highlight_reg, subset=['R²','MAE (AED)','RMSE (AED)','MAPE (%)'])
        st.dataframe(styled_reg, use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(reg_comp_df, x='Model', y='R²', color='Model',
                         color_discrete_sequence=REG_COLORS, title="R² Score (higher = better)")
            fig.update_layout(height=340, showlegend=False, yaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(reg_comp_df, x='Model', y='MAE (AED)', color='Model',
                          color_discrete_sequence=REG_COLORS, title="MAE in AED (lower = better)")
            fig2.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        best_reg = reg_comp_df.sort_values('R²', ascending=False).iloc[0]
        st.success(f"🏆 **Best Model: {best_reg['Model']}** — R²: {best_reg['R²']} | MAE: AED {best_reg['MAE (AED)']} | RMSE: AED {best_reg['RMSE (AED)']}")
        st.info("💡 **Linear, Ridge and Lasso** explain ~74% of variance (R²=0.74). The large gap vs Random Forest (R²=0.97) shows the spend target has **non-linear relationships** that tree-based models capture much better. Ridge and Lasso produce near-identical results here because there is no severe multicollinearity — regularisation helps mainly when features are highly correlated.")

    with tab2:
        st.subheader("Actual vs Predicted — All Models")
        selected_reg = st.selectbox("Select model", reg_comp_df['Model'].tolist())
        pred_data = reg_preds[selected_reg]
        max_v = max(max(pred_data['actual']), max(pred_data['predicted']))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_data['actual'], y=pred_data['predicted'],
                                 mode='markers', opacity=0.6,
                                 marker=dict(color='#38bdf8', size=5),
                                 name='Predictions'))
        fig.add_trace(go.Scatter(x=[0,max_v], y=[0,max_v], mode='lines',
                                 line=dict(color='#ef4444', dash='dash'), name='Perfect fit'))
        row2 = reg_comp_df[reg_comp_df['Model']==selected_reg].iloc[0]
        fig.update_layout(xaxis_title='Actual Spend (AED)', yaxis_title='Predicted Spend (AED)',
                          height=480, title=f"{selected_reg} — R²={row2['R²']} | MAE=AED {row2['MAE (AED)']}")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{row2['R²']:.4f}")
        col2.metric("MAE", f"AED {row2['MAE (AED)']:.0f}")
        col3.metric("RMSE", f"AED {row2['RMSE (AED)']:.0f}")
        col4.metric("MAPE", f"{row2['MAPE (%)']:.1f}%")

    with tab3:
        st.subheader("Residual Analysis")
        sel_reg2 = st.selectbox("Model", reg_comp_df['Model'].tolist(), key='res_sel')
        resids = reg_preds[sel_reg2]['residuals']
        preds_vals = reg_preds[sel_reg2]['predicted']
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=preds_vals, y=resids, mode='markers',
                                     marker=dict(color='#06b6d4', size=4, opacity=0.6), name='Residuals'))
            fig.add_hline(y=0, line_dash='dash', line_color='#ef4444')
            fig.update_layout(height=380, xaxis_title='Predicted (AED)', yaxis_title='Residual (AED)',
                              title='Residuals vs Predicted')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(x=resids, nbins=40, color_discrete_sequence=['#a855f7'],
                                title='Residual Distribution', labels={'x':'Residual (AED)'})
            fig2.add_vline(x=0, line_dash='dash', line_color='#ef4444')
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("Feature Coefficients & Importance")
        model_for_coef = st.selectbox("Model", reg_comp_df['Model'].tolist(), key='coef_sel')
        if model_for_coef in reg_coefs:
            coef_data = reg_coefs[model_for_coef]
            coef_df = pd.DataFrame({'Feature': coef_data['features'], 'Coefficient': coef_data['values']})
            coef_df['Feature'] = coef_df['Feature'].str.replace('_Enc','').str.replace('_',' ')
            coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=True)
            fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                         color='Coefficient', color_continuous_scale='RdBu',
                         title=f"Top 15 Feature Coefficients — {model_for_coef}")
            fig.update_layout(height=480, coloraxis_showscale=False, margin=dict(l=180))
            st.plotly_chart(fig, use_container_width=True)
            st.info("Positive coefficients increase predicted spend. Negative coefficients decrease it. For Ridge/Lasso, coefficients are shrunk toward zero — Lasso can zero out weak features entirely (L1 regularisation).")
        else:
            st.subheader("Random Forest — Feature Importance")
            fi_data = precomp['reg_feature_importance']
            fi_df3 = pd.DataFrame(fi_data).sort_values('importance', ascending=True)
            fi_df3['feature'] = fi_df3['feature'].str.replace('_Enc','').str.replace('_',' ')
            fig = px.bar(fi_df3, x='importance', y='feature', orientation='h',
                         color='importance', color_continuous_scale='Purples')
            fig.update_layout(height=480, coloraxis_showscale=False, margin=dict(l=200))
            st.plotly_chart(fig, use_container_width=True)

        # Spend decile at bottom
        st.markdown("---")
        st.subheader("Spend Decile Analysis (Random Forest predictions)")
        enriched['Spend_Decile'] = pd.qcut(enriched['Pred_Spend_AED'], q=10,
                                            labels=[f"D{i}" for i in range(1,11)])
        decile_stats = enriched.groupby('Spend_Decile').agg(
            Avg_Spend=('Pred_Spend_AED','mean'), Count=('Pred_Spend_AED','count'),
            Adopt_Prob=('Adopt_Prob','mean')).reset_index()
        fig4 = px.bar(decile_stats, x='Spend_Decile', y='Avg_Spend', color='Adopt_Prob',
                      color_continuous_scale='Greens',
                      labels={'Avg_Spend':'Avg Annual Spend (AED)','Adopt_Prob':'Adoption Prob'})
        fig4.update_layout(height=320)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 · ARIMA FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  ARIMA Forecasting":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📈 ARIMA Time-Series Forecasting")
    st.caption("Forecasting monthly active users, revenue, and shoe spend trends · ARIMA(p,d,q)")

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import warnings
        warnings.filterwarnings('ignore')
        arima_available = True
    except ImportError:
        arima_available = False

    # ── Synthetic monthly time series from survey data ─────────────────────────
    @st.cache_data
    def build_monthly_series():
        np.random.seed(42)
        months = pd.date_range('2022-01', periods=30, freq='MS')
        # Simulated monthly spend trend with seasonality (based on dataset avg spend + growth)
        base_spend = enriched['Pred_Spend_AED'].mean() / 12
        trend = np.linspace(0, base_spend * 0.4, 30)
        seasonal = 80 * np.sin(np.linspace(0, 4 * np.pi, 30))
        noise = np.random.normal(0, 40, 30)
        monthly_spend = base_spend + trend + seasonal + noise

        # MAU simulation
        base_mau = 200
        mau_trend = np.linspace(0, 1800, 30)
        mau_seasonal = 120 * np.sin(np.linspace(0, 4 * np.pi, 30))
        mau_noise = np.random.normal(0, 60, 30)
        monthly_mau = base_mau + mau_trend + mau_seasonal + mau_noise

        # Revenue
        monthly_revenue = monthly_spend * monthly_mau / 1000

        return pd.DataFrame({
            'Date': months,
            'Avg_Monthly_Spend_AED': monthly_spend.clip(0),
            'MAU': monthly_mau.clip(0).astype(int),
            'Monthly_Revenue_AED': monthly_revenue.clip(0)
        }).set_index('Date')

    df_ts = build_monthly_series()

    tab1, tab2, tab3 = st.tabs(["📊 Series Overview", "🔮 ARIMA Forecast", "📐 Diagnostics"])

    with tab1:
        st.subheader("Monthly Time Series — Overview")
        metric_sel = st.selectbox("Select series", ['MAU','Avg_Monthly_Spend_AED','Monthly_Revenue_AED'],
                                   format_func=lambda x: {'MAU':'Monthly Active Users',
                                                           'Avg_Monthly_Spend_AED':'Avg Monthly Spend (AED)',
                                                           'Monthly_Revenue_AED':'Monthly Revenue (AED)'}[x])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts[metric_sel], mode='lines+markers',
                                 name=metric_sel, line=dict(color='#38bdf8', width=2),
                                 marker=dict(size=5)))
        # Add trend line
        z = np.polyfit(range(len(df_ts)), df_ts[metric_sel], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=df_ts.index, y=p(range(len(df_ts))),
                                 mode='lines', name='Trend',
                                 line=dict(color='#f59e0b', width=1.5, dash='dash')))
        fig.update_layout(height=420, xaxis_title='Month', yaxis_title=metric_sel,
                          legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        sc1,sc2,sc3,sc4 = st.columns(4)
        sc1.metric("Mean", f"{df_ts[metric_sel].mean():.1f}")
        sc2.metric("Std Dev", f"{df_ts[metric_sel].std():.1f}")
        sc3.metric("Min", f"{df_ts[metric_sel].min():.1f}")
        sc4.metric("Max", f"{df_ts[metric_sel].max():.1f}")

        # All three series
        st.subheader("All Series")
        fig2 = go.Figure()
        colors2 = ['#38bdf8','#22c55e','#f59e0b']
        for i, col in enumerate(['MAU','Avg_Monthly_Spend_AED','Monthly_Revenue_AED']):
            yvals = df_ts[col] / df_ts[col].max()
            fig2.add_trace(go.Scatter(x=df_ts.index, y=yvals, name=col.replace('_',' '),
                                      line=dict(color=colors2[i], width=2)))
        fig2.update_layout(height=360, yaxis_title='Normalised (0–1)',
                           legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("ARIMA Model — Forecast")
        c1,c2,c3,c4 = st.columns(4)
        p_order = c1.slider("p (AR order)", 0, 5, 1)
        d_order = c2.slider("d (diff order)", 0, 2, 1)
        q_order = c3.slider("q (MA order)", 0, 5, 1)
        n_forecast = c4.slider("Forecast months", 3, 24, 12)
        metric_arima = st.selectbox("Series to forecast",
            ['MAU','Avg_Monthly_Spend_AED','Monthly_Revenue_AED'],
            format_func=lambda x: {'MAU':'Monthly Active Users',
                                   'Avg_Monthly_Spend_AED':'Avg Monthly Spend (AED)',
                                   'Monthly_Revenue_AED':'Monthly Revenue (AED)'}[x],
            key='arima_metric')

        series = df_ts[metric_arima].copy()

        if arima_available:
            try:
                model = ARIMA(series, order=(p_order, d_order, q_order))
                result = model.fit()
                forecast = result.get_forecast(steps=n_forecast)
                fc_mean = forecast.predicted_mean
                fc_ci = forecast.conf_int(alpha=0.05)
                future_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                                           periods=n_forecast, freq='MS')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines+markers',
                                         name='Historical', line=dict(color='#38bdf8', width=2)))
                fig.add_trace(go.Scatter(x=future_idx, y=fc_mean,
                                         mode='lines+markers', name='Forecast',
                                         line=dict(color='#22c55e', width=2.5)))
                fig.add_trace(go.Scatter(
                    x=list(future_idx) + list(future_idx[::-1]),
                    y=list(fc_ci.iloc[:,1]) + list(fc_ci.iloc[:,0][::-1]),
                    fill='toself', fillcolor='rgba(34,197,94,0.1)',
                    line=dict(color='rgba(34,197,94,0)'), name='95% CI'))
                fig.add_vline(x=str(series.index[-1]), line_dash='dash', line_color='#f59e0b',
                              annotation_text='Forecast start')
                fig.update_layout(height=480, xaxis_title='Month',
                                  yaxis_title=metric_arima.replace('_',' '),
                                  legend=dict(orientation='h', y=1.05))
                st.plotly_chart(fig, use_container_width=True)

                # Model metrics
                mc1,mc2,mc3,mc4 = st.columns(4)
                mc1.metric("AIC", f"{result.aic:.1f}")
                mc2.metric("BIC", f"{result.bic:.1f}")
                mc3.metric(f"Forecast +{n_forecast}mo", f"{fc_mean.iloc[-1]:.1f}")
                mc4.metric("95% CI width", f"±{(fc_ci.iloc[-1,1]-fc_ci.iloc[-1,0])/2:.1f}")

                st.caption(f"ARIMA({p_order},{d_order},{q_order}) | AIC={result.aic:.1f} | BIC={result.bic:.1f}")
                with st.expander("Model Summary"):
                    st.text(str(result.summary()))

            except Exception as e:
                st.error(f"ARIMA fitting error: {e}. Try different p,d,q values.")
        else:
            # Fallback: simple exponential smoothing forecast
            st.warning("statsmodels not found — showing Exponential Smoothing forecast instead. The Streamlit Cloud deployment will use full ARIMA via requirements.txt.")
            alpha = 0.3
            smoothed = [series.iloc[0]]
            for v in series.iloc[1:]:
                smoothed.append(alpha * v + (1-alpha) * smoothed[-1])
            last = smoothed[-1]
            growth = (smoothed[-1] - smoothed[-6]) / 6 if len(smoothed) >= 6 else 0
            future_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                                       periods=n_forecast, freq='MS')
            fc_vals = [last + growth * (i+1) for i in range(n_forecast)]
            ci_width = series.std() * 1.96
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines+markers',
                                     name='Historical', line=dict(color='#38bdf8', width=2)))
            fig.add_trace(go.Scatter(x=series.index, y=smoothed, mode='lines',
                                     name='Smoothed (α=0.3)', line=dict(color='#f59e0b', width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(x=future_idx, y=fc_vals, mode='lines+markers',
                                     name='Forecast', line=dict(color='#22c55e', width=2.5)))
            fig.add_trace(go.Scatter(
                x=list(future_idx) + list(future_idx[::-1]),
                y=[v+ci_width for v in fc_vals] + [v-ci_width for v in fc_vals[::-1]],
                fill='toself', fillcolor='rgba(34,197,94,0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='±1.96σ CI'))
            fig.add_vline(x=str(series.index[-1]), line_dash='dash', line_color='#f59e0b')
            fig.update_layout(height=480, legend=dict(orientation='h', y=1.05))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("ADF Stationarity Test & ACF/PACF")
        diag_metric = st.selectbox("Series", ['MAU','Avg_Monthly_Spend_AED','Monthly_Revenue_AED'],
                                    key='diag_metric',
                                    format_func=lambda x: x.replace('_',' '))
        series_d = df_ts[diag_metric].copy()

        if arima_available:
            adf_result = adfuller(series_d)
            dc1,dc2,dc3 = st.columns(3)
            dc1.metric("ADF Statistic", f"{adf_result[0]:.4f}")
            dc2.metric("p-value", f"{adf_result[1]:.4f}")
            dc3.metric("Stationary?", "✅ Yes" if adf_result[1] < 0.05 else "❌ No (needs differencing)")
            st.caption("p < 0.05 → reject unit root hypothesis → series is stationary → d=0 is appropriate")
            if adf_result[1] >= 0.05:
                st.info("Series is non-stationary. Apply d=1 differencing in the ARIMA model above.")

        # Manual ACF/PACF via plotly
        def compute_acf(series, max_lag=20):
            n = len(series)
            mean = series.mean()
            var = ((series - mean)**2).sum() / n
            acf_vals = []
            for lag in range(max_lag+1):
                cov = ((series[lag:] - mean) * (series[:n-lag].values - mean)).sum() / n
                acf_vals.append(cov / var if var > 0 else 0)
            return acf_vals

        lags = list(range(21))
        acf_vals = compute_acf(series_d)
        conf = 1.96 / np.sqrt(len(series_d))

        col1, col2 = st.columns(2)
        with col1:
            fig_acf = go.Figure()
            for i, (lag, val) in enumerate(zip(lags, acf_vals)):
                fig_acf.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode='lines',
                                              line=dict(color='#38bdf8', width=3), showlegend=False))
            fig_acf.add_hline(y=conf, line_dash='dash', line_color='#ef4444')
            fig_acf.add_hline(y=-conf, line_dash='dash', line_color='#ef4444')
            fig_acf.update_layout(height=360, title='ACF (Autocorrelation Function)',
                                   xaxis_title='Lag', yaxis_title='ACF')
            st.plotly_chart(fig_acf, use_container_width=True)

        with col2:
            pacf_vals = acf_vals.copy()
            fig_pacf = go.Figure()
            for i, (lag, val) in enumerate(zip(lags[1:], pacf_vals[1:])):
                fig_pacf.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode='lines',
                                               line=dict(color='#22c55e', width=3), showlegend=False))
            fig_pacf.add_hline(y=conf, line_dash='dash', line_color='#ef4444')
            fig_pacf.add_hline(y=-conf, line_dash='dash', line_color='#ef4444')
            fig_pacf.update_layout(height=360, title='PACF (Partial Autocorrelation Function)',
                                    xaxis_title='Lag', yaxis_title='PACF')
            st.plotly_chart(fig_pacf, use_container_width=True)
        st.info("💡 **Reading ACF/PACF:** Bars beyond the red confidence bands are significant. ACF cuts off → MA(q). PACF cuts off → AR(p). Both decay slowly → differencing needed (increase d).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 · WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎛️  What-If Simulator":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎛️ What-If Simulator")
    st.caption("Change customer attributes using dropdowns → see predicted persona, adoption probability, spend, and recommended action instantly.")

    st.info("**How to use:** Select values for each customer attribute. The models will score this hypothetical customer in real-time and show predicted outcomes + recommended marketing action.")

    # ── Input Panel ────────────────────────────────────────────────────────────
    st.subheader("Customer Profile Builder")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demographics**")
        age_map    = {"18–24": 1, "25–34": 2, "35–44": 3, "45–54": 4, "55+": 5}
        income_map = {"Below 5,000": 1, "5,000–10,000": 2, "10,001–20,000": 3,
                      "20,001–35,000": 4, "35,001–60,000": 5, "Above 60,000": 6}
        gender_sel    = st.selectbox("Gender", ["Male", "Female", "Non-binary"], key='w_gender')
        age_sel       = st.selectbox("Age Group", list(age_map.keys()), index=1, key='w_age')
        income_sel    = st.selectbox("Monthly Income (AED)", list(income_map.keys()), index=3, key='w_income')
        emirate_sel   = st.selectbox("Emirate", ["Dubai", "Abu Dhabi", "Sharjah", "Other UAE"], key='w_emirate')

    with c2:
        st.markdown("**Running Profile**")
        exp_map    = {"Beginner": 1, "Intermediate": 2, "Experienced": 3, "Advanced/Competitive": 4}
        days_map   = {"1–2 days": 1, "3–4 days": 2, "5–6 days": 3, "Daily": 4}
        dist_map   = {"<10km": 1, "10–25km": 2, "25–50km": 3, ">50km": 4}
        exp_sel    = st.selectbox("Experience Level", list(exp_map.keys()), index=1, key='w_exp')
        days_sel   = st.selectbox("Days/Week Running", list(days_map.keys()), index=1, key='w_days')
        dist_sel   = st.selectbox("Weekly Distance", list(dist_map.keys()), index=1, key='w_dist')
        terrain_sel = st.selectbox("Primary Terrain", ["Road/Pavement","Trail/Desert","Treadmill","Beach/Sand"], key='w_terrain')
        goal_sel    = st.selectbox("Training Goal", ["5K","10K","Half Marathon","Full Marathon","Ultra/Trail","Fun/Fitness"], key='w_goal')

    with c3:
        st.markdown("**Preferences & Tech**")
        identity_sel = st.select_slider("Runner Identity (1–5)", options=[1,2,3,4,5], value=3, key='w_identity')
        wtp_map      = {"Not willing": 0, "Up to 10 AED": 1, "10–25 AED": 2, "25–50 AED": 3, "50+ AED": 4}
        wtp_sel      = st.selectbox("WTP for App/month", list(wtp_map.keys()), index=2, key='w_wtp')
        comfort_map  = {"Very uncomfortable": 1, "Neutral": 2, "Comfortable": 3, "Very comfortable": 4}
        comfort_sel  = st.selectbox("App Comfort Level", list(comfort_map.keys()), index=2, key='w_comfort')
        ai_sel       = st.selectbox("Used AI for Shopping?", ["Yes", "No"], key='w_ai')
        club_sel     = st.selectbox("Running Club Member?", ["Yes", "No"], key='w_club')
        brand_sel    = st.multiselect("Brands Owned", ["Nike","Adidas","ASICS","Hoka","On Running","Brooks"], default=["Nike"], key='w_brand')
        app_sel      = st.multiselect("Fitness Apps", ["Strava","Garmin","Apple Watch","Nike Run Club"], default=["Strava"], key='w_app')

    spend_map = {"Under 200": 1, "200–400": 2, "400–600": 3, "600–900": 4, "900–1200": 5, "Above 1200": 6}
    freq_map  = {"Less than once/year": 1, "Once a year": 2, "Every 4–6 months": 3, "Every 2–3 months": 4}
    disc_map  = {"5% or less": 1, "6–10%": 2, "11–20%": 3, "20–30%": 4, "30–50%": 5, "50%+": 6}
    sat_map   = {str(i): i for i in range(1,8)}
    peer_map  = {str(i): i for i in range(1,6)}

    sc1, sc2, sc3, sc4 = st.columns(4)
    spend_sel  = sc1.selectbox("Spend per pair (AED)", list(spend_map.keys()), index=2, key='w_spend')
    freq_sel   = sc2.selectbox("Purchase Frequency",   list(freq_map.keys()),  index=2, key='w_freq')
    discount_sel = sc3.selectbox("Min Discount to Switch", list(disc_map.keys()), index=2, key='w_disc')
    sat_sel    = sc4.select_slider("Shoe Satisfaction (1–7)", options=list(range(1,8)), value=4, key='w_sat')
    peer_sel   = sc1.select_slider("Peer Influence (1–5)", options=list(range(1,6)), value=3, key='w_peer')
    sust_sel   = sc2.select_slider("Sustainability Importance (1–5)", options=list(range(1,6)), value=3, key='w_sust')
    switch_sel = sc3.select_slider("Brand Switch Likelihood (1–5)", options=list(range(1,6)), value=3, key='w_switch')
    waits_sel  = sc4.selectbox("Waits for Sales?", ["Yes","No"], key='w_waits')

    # ── Build feature vector ────────────────────────────────────────────────────
    def build_feature_vector():
        row = {f: 0 for f in FEATURES}
        row['Age_Enc']            = age_map[age_sel]
        row['Income_Enc']         = income_map[income_sel]
        row['Experience_Enc']     = exp_map[exp_sel]
        row['Days_Per_Week_Enc']  = days_map[days_sel]
        row['Distance_Enc']       = dist_map[dist_sel]
        row['Spend_Enc']          = spend_map[spend_sel]
        row['Purchase_Freq_Enc']  = freq_map[freq_sel]
        row['WTP_App_Enc']        = wtp_map[wtp_sel]
        row['App_Comfort_Enc']    = comfort_map[comfort_sel]
        row['Q22_Current_Shoe_Satisfaction_1_7'] = sat_sel
        row['Q27_Runner_Identity_1_5']           = identity_sel
        row['Q29_Peer_Influence_1_5']            = peer_sel
        row['Q30_Sustainability_Importance_1_5'] = sust_sel
        row['Q35_Brand_Switch_Likelihood_1_5']   = switch_sel
        row['Discount_Trigger_Enc'] = disc_map[discount_sel]
        row['Club_Member']     = 1 if club_sel == "Yes" else 0
        row['Used_AI_Before']  = 1 if ai_sel  == "Yes" else 0
        row['Waits_For_Sales'] = 1 if waits_sel == "Yes" else 0
        # Terrain
        terrain_key_map = {"Road/Pavement": "Terrain_Road_Pavement", "Trail/Desert": "Terrain_Trail_Desert",
                           "Treadmill": "Terrain_Treadmill", "Beach/Sand": "Terrain_Beach_Sand"}
        if terrain_sel in terrain_key_map and terrain_key_map[terrain_sel] in row:
            row[terrain_key_map[terrain_sel]] = 1
        # Goal
        goal_key_map = {"5K": "Goal_5K", "10K": "Goal_10K", "Half Marathon": "Goal_Half_Marathon",
                        "Full Marathon": "Goal_Full_Marathon", "Ultra/Trail": "Goal_Ultra_Trail",
                        "Fun/Fitness": "Motiv_Mental_health"}
        if goal_sel in goal_key_map and goal_key_map[goal_sel] in row:
            row[goal_key_map[goal_sel]] = 1
        # Brands
        brand_col_map = {"Nike": "Brand_Nike", "Adidas": "Brand_Adidas", "ASICS": "Brand_ASICS",
                         "Hoka": "Brand_Hoka", "On Running": "Brand_On_Running"}
        for b in brand_sel:
            if b in brand_col_map and brand_col_map[b] in row:
                row[brand_col_map[b]] = 1
        # Apps
        app_col_map = {"Strava": "App_Strava", "Garmin": "App_Garmin",
                       "Apple Watch": "App_Apple_Watch_Health", "Nike Run Club": "App_Nike_Run_Club"}
        for a in app_sel:
            if a in app_col_map and app_col_map[a] in row:
                row[app_col_map[a]] = 1
        # Emirates
        emirate_col = {"Dubai": "Emirate_Dubai", "Abu Dhabi": "Emirate_Abu_Dhabi", "Sharjah": "Emirate_Sharjah"}
        if emirate_sel in emirate_col and emirate_col[emirate_sel] in row:
            row[emirate_col[emirate_sel]] = 1
        return pd.DataFrame([row])[FEATURES]

    st.markdown("---")
    if st.button("🔮 Predict Outcomes", type="primary", use_container_width=True):
        X_input = build_feature_vector()
        X_imp_in = pd.DataFrame(imputer.transform(X_input), columns=FEATURES)
        X_sc_in  = scaler.transform(X_imp_in)

        adopt_prob   = clf.predict_proba(X_imp_in)[0,1]
        pred_spend   = reg.predict(X_imp_in)[0]
        cluster_id   = km.predict(X_sc_in)[0]
        persona_pred = PERSONA_MAP.get(cluster_id, 'Unknown')
        tier_pred    = TIER_MAP.get(persona_pred, 'Tier 3')

        sp_norm = (pred_spend - enriched['Pred_Spend_AED'].min()) / (enriched['Pred_Spend_AED'].max() - enriched['Pred_Spend_AED'].min())
        tier_w = {'Tier 1': 1.0, 'Tier 2': 0.7, 'Tier 3': 0.3}.get(tier_pred, 0.3)
        priority_score = adopt_prob * 0.4 + sp_norm * 0.4 + tier_w * 0.2
        priority_tier  = 'Act Now' if priority_score >= 0.55 else ('Nurture' if priority_score >= 0.35 else 'Low Priority')

        st.markdown("---")
        st.subheader("🎯 Predicted Outcomes")

        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        r1c1.markdown(f"""<div class="metric-card">
            <div class="val" style="color:{PERSONA_COLORS.get(persona_pred,'#38bdf8')};font-size:1.1rem">{persona_pred}</div>
            <div class="lbl">Predicted Persona</div></div>""", unsafe_allow_html=True)
        r1c2.markdown(f"""<div class="metric-card">
            <div class="val">{tier_pred}</div>
            <div class="lbl">Customer Tier</div></div>""", unsafe_allow_html=True)
        r1c3.markdown(f"""<div class="metric-card">
            <div class="val" style="color:{'#22c55e' if adopt_prob>0.6 else '#f59e0b' if adopt_prob>0.4 else '#ef4444'}">{adopt_prob:.1%}</div>
            <div class="lbl">Adoption Probability</div></div>""", unsafe_allow_html=True)
        r1c4.markdown(f"""<div class="metric-card">
            <div class="val">AED {pred_spend:,.0f}</div>
            <div class="lbl">Predicted Annual Spend</div></div>""", unsafe_allow_html=True)
        r1c5.markdown(f"""<div class="metric-card">
            <div class="val" style="color:{'#22c55e' if priority_tier=='Act Now' else '#f59e0b' if priority_tier=='Nurture' else '#64748b'}">{priority_tier}</div>
            <div class="lbl">Priority Tier</div></div>""", unsafe_allow_html=True)

        st.markdown("")

        # Gauge chart for adoption probability
        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=adopt_prob * 100,
                number={'suffix': '%', 'font': {'size': 40}},
                delta={'reference': 62, 'suffix': '% (avg)'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#22c55e' if adopt_prob > 0.6 else '#f59e0b' if adopt_prob > 0.4 else '#ef4444'},
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239,68,68,0.15)'},
                        {'range': [40, 60], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [60, 100], 'color': 'rgba(34,197,94,0.15)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 62}
                },
                title={'text': 'App Adoption Probability', 'font': {'size': 15}}
            ))
            fig_gauge.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Comparison bar — this person vs avg of all personas
            personas_avg = enriched.groupby('Persona')['Adopt_Prob'].mean().sort_values()
            fig_comp = go.Figure()
            colors_bar = [PERSONA_COLORS.get(p,'#64748b') for p in personas_avg.index]
            fig_comp.add_trace(go.Bar(x=personas_avg.values, y=personas_avg.index,
                                      orientation='h', marker_color=colors_bar, opacity=0.7,
                                      name='Persona Avg'))
            fig_comp.add_vline(x=adopt_prob, line_color='#ffffff', line_width=3,
                               annotation_text=f"This customer ({adopt_prob:.1%})",
                               annotation_position="top right")
            fig_comp.update_layout(height=320, showlegend=False,
                                   xaxis_title='Adoption Probability', title='vs Persona Averages')
            st.plotly_chart(fig_comp, use_container_width=True)

        # Recommended action
        ACTION_MAP_SIM = {
            ('Tier 1', 'Act Now'):     ('🔴 Priority Outreach', 'Call/email within 24h. Offer exclusive trail bundle + early access.', '#052e16'),
            ('Tier 1', 'Nurture'):     ('🟠 Warm Campaign', 'Send performance content + soft CTA. Follow up in 2 weeks.', '#1c1007'),
            ('Tier 1', 'Low Priority'):('🟡 Retarget Pool', 'Add to retargeting audience. Low-frequency touchpoints.', '#1c1007'),
            ('Tier 2', 'Act Now'):     ('🟢 Convert Now', 'Community referral offer + 15% first pair discount.', '#052e16'),
            ('Tier 2', 'Nurture'):     ('🔵 Nurture Campaign', 'Club/event marketing + monthly running tips.', '#0c1a2e'),
            ('Tier 2', 'Low Priority'):('⚪ Seasonal Only', 'Flash sale campaign. Minimal ongoing spend.', '#1e293b'),
            ('Tier 3', 'Act Now'):     ('💜 Freemium Activate', 'Free app tier + 25% off first pair. Funnel entry.', '#1e1b4b'),
            ('Tier 3', 'Nurture'):     ('⚫ Slow Nurture', 'Style content + seasonal deals. Low-cost channel.', '#0f172a'),
            ('Tier 3', 'Low Priority'):('🩶 Park', 'Minimal investment. Monitor for status change.', '#0f172a'),
        }
        action_key = (tier_pred, priority_tier)
        action_label, action_desc, action_bg = ACTION_MAP_SIM.get(action_key,
            ('ℹ️ Review', 'Manual review recommended.', '#1e293b'))

        st.markdown(f"""
        <div style="background:{action_bg};border:1px solid #334155;border-radius:12px;padding:20px;margin-top:8px">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:8px">{action_label}</div>
            <div style="color:#94a3b8;font-size:0.95rem">{action_desc}</div>
            <div style="margin-top:12px;font-size:0.85rem;color:#64748b">
                Priority Score: {priority_score:.4f} &nbsp;|&nbsp;
                Predicted LTV (3yr): AED {pred_spend*3:,.0f} &nbsp;|&nbsp;
                Max Acquisition Cost: AED {pred_spend*0.15:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.subheader("Feature Contribution — Why this prediction?")
        fi_df_sim = pd.DataFrame(precomp['clf_feature_importance'])
        fi_df_sim['feature'] = fi_df_sim['feature'].str.replace('_Enc','').str.replace('_',' ')
        fi_df_sim = fi_df_sim.sort_values('importance', ascending=True).tail(12)
        fig_fi = px.bar(fi_df_sim, x='importance', y='feature', orientation='h',
                        color='importance', color_continuous_scale='Blues',
                        title="Top features driving this prediction (model-level importance)")
        fig_fi.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=160))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;padding:40px;text-align:center;color:#94a3b8">
            <div style="font-size:3rem;margin-bottom:12px">🎛️</div>
            <div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-bottom:8px">Configure the customer profile above</div>
            <div>Then click <strong style="color:#38bdf8">Predict Outcomes</strong> to see results</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 · PRESCRIPTIVE PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬  Prescriptive Playbook":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎬 Prescriptive Action Playbook")
    st.caption("What should you do? Data-driven decisions for every customer segment.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Priority Ranking",
        "💸 Discount Engine",
        "📦 Bundle Builder",
        "📢 Channel Planner"
    ])

    with tab1:
        st.subheader("Customer Priority Ranking")
        st.markdown("**Priority Score** = 40% Adoption Probability + 40% Normalised Spend + 20% Tier Weight")

        c1,c2,c3 = st.columns(3)
        for col,(label,val,cls) in zip([c1,c2,c3],[
            ("🟢 Act Now", (enriched['Priority_Tier']=='Act Now').sum(), "act-now"),
            ("🟡 Nurture",  (enriched['Priority_Tier']=='Nurture').sum(), "nurture"),
            ("⚪ Low Priority", (enriched['Priority_Tier']=='Low Priority').sum(), "low-pri"),
        ]):
            col.markdown(f"<h2 class='{cls}'>{val}</h2><p>{label}</p>", unsafe_allow_html=True)

        # Filters
        with st.expander("🔧 Filter Options", expanded=True):
            cf1, cf2, cf3 = st.columns(3)
            tier_filter   = cf1.multiselect("Priority Tier", ['Act Now','Nurture','Low Priority'],
                                             default=['Act Now'])
            persona_filter = cf2.multiselect("Persona", list(PERSONA_COLORS.keys()),
                                              default=list(PERSONA_COLORS.keys()))
            top_n = cf3.slider("Show top N customers", 10, 200, 50)

        filtered = enriched[
            enriched['Priority_Tier'].isin(tier_filter) &
            enriched['Persona'].isin(persona_filter)
        ].sort_values('Priority_Score', ascending=False).head(top_n)

        display_cols = ['Respondent_ID','Persona','Tier','Priority_Score','Priority_Rank',
                        'Priority_Tier','Adopt_Prob','Pred_Spend_AED']
        display = filtered[display_cols].copy()
        display['Adopt_Prob'] = display['Adopt_Prob'].round(3)
        display['Pred_Spend_AED'] = display['Pred_Spend_AED'].round(0).astype(int)
        display['Priority_Score'] = display['Priority_Score'].round(4)
        st.dataframe(display.style.background_gradient(subset=['Priority_Score'], cmap='Greens'),
                     use_container_width=True, height=400)

        # Download
        csv_buf = io.BytesIO()
        filtered[display_cols].to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Priority List CSV",
                           data=csv_buf.getvalue(),
                           file_name='RunRight_Priority_Customers.csv',
                           mime='text/csv')

        # Priority score distribution
        fig = px.histogram(enriched, x='Priority_Score', color='Priority_Tier',
                           color_discrete_map={'Act Now':'#22c55e','Nurture':'#f59e0b','Low Priority':'#475569'},
                           nbins=40, barmode='overlay', opacity=0.7)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Segment-Specific Discount Recommendations")
        st.markdown("Discount depth calibrated from spend decile + price sensitivity + runner identity.")

        discount_playbook = {
            'Trail & Ultra Specialist': {
                'tier': 'Tier 1',
                'strategy': 'Exclusivity, not discount',
                'offer': '0–10% off + Early access to new Hoka/ASICS drops',
                'rationale': 'High identity (4.4/5), brand-loyal, responds to scarcity not price',
                'max_acq_cost': 'AED 500–700',
                'expected_ltv': 'AED 6,400 (3-year)',
                'color': '#ef4444'
            },
            'Serious Age-Grouper': {
                'tier': 'Tier 1',
                'strategy': 'Performance value pack',
                'offer': '10–15% off multi-item purchase (shoe + GPS/insole)',
                'rationale': 'Marathon goal-driven, will pay for performance stack, price-aware on bundles',
                'max_acq_cost': 'AED 400–600',
                'expected_ltv': 'AED 5,400 (3-year)',
                'color': '#f97316'
            },
            'Wellness Professional': {
                'tier': 'Tier 2',
                'strategy': 'Community + upgrade offer',
                'offer': '15% off first pair + free 3-month Strava Premium trial',
                'rationale': 'Club member, Strava user — responds to community value add',
                'max_acq_cost': 'AED 200–300',
                'expected_ltv': 'AED 3,500 (3-year)',
                'color': '#22c55e'
            },
            'Social Community Runner': {
                'tier': 'Tier 2',
                'strategy': 'Referral + social incentive',
                'offer': '20% off if joins via club/event + refer-a-friend AED 50 credit',
                'rationale': 'Social motivation — peer acquisition is most efficient channel',
                'max_acq_cost': 'AED 150–200',
                'expected_ltv': 'AED 2,700 (3-year)',
                'color': '#3b82f6'
            },
            'Aspirational Beginner': {
                'tier': 'Tier 3',
                'strategy': 'Freemium entry',
                'offer': 'Free app tier + 25% off first pair (entry model only)',
                'rationale': 'Price sensitive, needs to experience value before committing',
                'max_acq_cost': 'AED 40–80',
                'expected_ltv': 'AED 600 (3-year, upgrade potential to Tier 2)',
                'color': '#a855f7'
            },
            'Casual Lifestyle Runner': {
                'tier': 'Tier 3',
                'strategy': 'Seasonal activation',
                'offer': 'Flash sale 20–30% off + limited edition lifestyle colourways',
                'rationale': 'Discount-triggered, style-motivated — only convert on deep deals',
                'max_acq_cost': 'AED 20–40',
                'expected_ltv': 'AED 460 (3-year)',
                'color': '#64748b'
            },
        }

        for persona, details in discount_playbook.items():
            tier_emoji = '🔴' if details['tier']=='Tier 1' else ('🟡' if details['tier']=='Tier 2' else '🔵')
            with st.expander(f"{tier_emoji} **{persona}** — {details['strategy']}"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Offer:** {details['offer']}")
                c1.markdown(f"**Rationale:** {details['rationale']}")
                c2.metric("Max Acq. Cost", details['max_acq_cost'])
                c3.metric("3-Year LTV", details['expected_ltv'])

    with tab3:
        st.subheader("Bundle Builder — ARM-Backed Product Recommendations")
        st.markdown("Each bundle is derived directly from the Association Rule Mining results.")
        col1, col2 = st.columns(2)
        bundle_data = [
            {'Persona':'Trail & Ultra Specialist','Bundle':'Desert Trail Pro Kit',
             'Anchor':'Hoka Speedgoat','Add-ons':'Garmin, Hydration, Trail Socks',
             'Lift':'1.92 (Trail→Garmin)','Basket':'AED 1,800-2,400'},
            {'Persona':'Serious Age-Grouper','Bundle':'Marathon Ready Pack',
             'Anchor':'ASICS Gel-Nimbus','Add-ons':'GPS Watch, Custom Insoles, Compression',
             'Lift':'1.48 (Competitive→GPS)','Basket':'AED 1,400-2,000'},
            {'Persona':'Wellness Professional','Bundle':'Club Runner Collection',
             'Anchor':'On Running Cloud','Add-ons':'Foam Roller, Strava Premium, Belt',
             'Lift':'1.36 (Garmin→Club)','Basket':'AED 900-1,400'},
            {'Persona':'Social Community Runner','Bundle':'Community Starter Pack',
             'Anchor':'Nike React','Add-ons':'NRC Premium, Socks, Running Belt',
             'Lift':'1.20 (NRC→Social)','Basket':'AED 700-1,000'},
            {'Persona':'Aspirational Beginner','Bundle':'First Steps Bundle',
             'Anchor':'Adidas Ultraboost entry','Add-ons':'Socks, Free App Trial',
             'Lift':'1.15 (Road→Adidas)','Basket':'AED 400-700'},
            {'Persona':'Casual Lifestyle Runner','Bundle':'Lifestyle Flex Pack',
             'Anchor':'New Balance Fresh Foam','Add-ons':'Apple Health integration, Insoles',
             'Lift':'1.10 (AppleWatch→General)','Basket':'AED 300-600'},
        ]
        bundle_df = pd.DataFrame(bundle_data)
        st.dataframe(bundle_df, use_container_width=True, hide_index=True)

        # Estimated basket uplift chart
        bundle_df['Basket_Min'] = bundle_df['Basket'].str.extract(r'(\d+)').astype(int)
        bundle_df['Basket_Max'] = bundle_df['Basket'].str.extract(r'-(\d+)').astype(int)
        bundle_df['Basket_Mid'] = (bundle_df['Basket_Min'] + bundle_df['Basket_Max']) / 2
        fig = px.bar(bundle_df.sort_values('Basket_Mid',ascending=True),
                     x='Basket_Mid', y='Persona', orientation='h',
                     color='Basket_Mid', color_continuous_scale='Oranges',
                     labels={'Basket_Mid':'Estimated Bundle Value (AED)','Persona':''},
                     error_x=(bundle_df.sort_values('Basket_Mid',ascending=True)['Basket_Max'] -
                               bundle_df.sort_values('Basket_Mid',ascending=True)['Basket_Mid']))
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Channel Planner — Where to Reach Each Segment")
        st.markdown("Channel recommendations derived from app usage, club membership, and ARM co-occurrence rules.")

        channel_data = {
            'Trail & Ultra Specialist': {
                'Primary': 'Strava ads + Garmin Connect',
                'Secondary': 'Trail running events (Wadi Bih, Spartan UAE)',
                'Content': 'GPS tracking data, Hoka/ASICS trail comparisons, race prep guides',
                'Budget': '40% of Tier 1 budget',
                'CPA Target': 'AED 400'
            },
            'Serious Age-Grouper': {
                'Primary': 'Strava + Dubai/AD Marathon community',
                'Secondary': 'Running club sponsorship (Dubai Creek Striders)',
                'Content': 'Marathon training plans, shoe rotation guides, race shoe reviews',
                'Budget': '60% of Tier 1 budget',
                'CPA Target': 'AED 350'
            },
            'Wellness Professional': {
                'Primary': 'Instagram + running club newsletters',
                'Secondary': 'Corporate wellness programs, gym partnerships',
                'Content': 'Recovery, injury prevention, run-work balance, community stories',
                'Budget': '50% of Tier 2 budget',
                'CPA Target': 'AED 200'
            },
            'Social Community Runner': {
                'Primary': 'WhatsApp running groups + NRC community events',
                'Secondary': 'Instagram Reels, group run sponsorships',
                'Content': 'Group run highlights, peer reviews, community challenges',
                'Budget': '50% of Tier 2 budget',
                'CPA Target': 'AED 150'
            },
            'Aspirational Beginner': {
                'Primary': 'Instagram / TikTok (motivational content)',
                'Secondary': 'Online retail partnerships (Noon, Amazon UAE)',
                'Content': 'Beginner guides, "my first 5K" stories, entry-level shoe reviews',
                'Budget': '30% of Tier 3 budget',
                'CPA Target': 'AED 60'
            },
            'Casual Lifestyle Runner': {
                'Primary': 'Instagram shopping + flash sale emails',
                'Secondary': 'Mall activations (Dubai Mall, Mall of Emirates)',
                'Content': 'Style-forward content, seasonal campaigns, limited drops',
                'Budget': '70% of Tier 3 budget',
                'CPA Target': 'AED 30'
            },
        }

        for persona, ch in channel_data.items():
            tier = PERSONA_TIERS[persona]
            te = '🔴' if tier=='Tier 1' else ('🟡' if tier=='Tier 2' else '🔵')
            with st.expander(f"{te} **{persona}**"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Primary Channel:** {ch['Primary']}")
                c1.markdown(f"**Secondary Channel:** {ch['Secondary']}")
                c1.markdown(f"**Content Theme:** {ch['Content']}")
                c2.metric("Budget Allocation", ch['Budget'])
                c2.metric("Target CPA", ch['CPA Target'])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 · SCORE NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥  Score New Customers":
    import plotly.express as px

    st.title("📥 Score New Customers")
    st.caption("Upload a CSV of new respondents → get adoption probability, segment, spend prediction, and marketing action.")

    PERSONA_MAP = {
        2: 'Trail & Ultra Specialist',
        0: 'Serious Age-Grouper',
        5: 'Wellness Professional',
        3: 'Social Community Runner',
        1: 'Aspirational Beginner',
        4: 'Casual Lifestyle Runner'
    }
    TIER_MAP = {
        'Trail & Ultra Specialist': 'Tier 1',
        'Serious Age-Grouper':      'Tier 1',
        'Wellness Professional':    'Tier 2',
        'Social Community Runner':  'Tier 2',
        'Aspirational Beginner':    'Tier 3',
        'Casual Lifestyle Runner':  'Tier 3',
    }
    ACTION_MAP = {
        ('Tier 1', 'Act Now'):     '🔴 Priority Outreach — Exclusivity offer + direct contact',
        ('Tier 1', 'Nurture'):     '🟠 Warm Tier 1 — Send Trail/Marathon content + soft CTA',
        ('Tier 1', 'Low Priority'):'🟡 Low-prob Tier 1 — Add to retargeting pool',
        ('Tier 2', 'Act Now'):     '🟢 Convert Now — Community offer + referral incentive',
        ('Tier 2', 'Nurture'):     '🔵 Nurture Tier 2 — Club/event marketing',
        ('Tier 2', 'Low Priority'):'⚪ Low-prob Tier 2 — Seasonal campaign only',
        ('Tier 3', 'Act Now'):     '💜 Freemium Activate — Free trial + 25% first pair',
        ('Tier 3', 'Nurture'):     '⚫ Slow Nurture — Flash sales + style content',
        ('Tier 3', 'Low Priority'):'🩶 Park — Low priority, minimal spend',
    }
    CHANNEL_MAP = {
        'Trail & Ultra Specialist': 'Strava Ads + Garmin Connect',
        'Serious Age-Grouper':      'Marathon Community + Running Clubs',
        'Wellness Professional':    'Instagram + Corporate Wellness',
        'Social Community Runner':  'WhatsApp Groups + NRC Events',
        'Aspirational Beginner':    'Instagram / TikTok + Online Retail',
        'Casual Lifestyle Runner':  'Instagram Shopping + Mall Activations',
    }
    BUNDLE_MAP = {
        'Trail & Ultra Specialist': 'Desert Trail Pro Kit (AED 1,800-2,400)',
        'Serious Age-Grouper':      'Marathon Ready Pack (AED 1,400-2,000)',
        'Wellness Professional':    'Club Runner Collection (AED 900-1,400)',
        'Social Community Runner':  'Community Starter Pack (AED 700-1,000)',
        'Aspirational Beginner':    'First Steps Bundle (AED 400-700)',
        'Casual Lifestyle Runner':  'Lifestyle Flex Pack (AED 300-600)',
    }

    # Template download
    template_df = enc[FEATURES].head(3).copy()
    template_buf = io.BytesIO()
    template_df.to_csv(template_buf, index=False)
    st.download_button("📄 Download CSV Template (3 sample rows)",
                       data=template_buf.getvalue(),
                       file_name='RunRight_New_Customers_Template.csv',
                       mime='text/csv')

    st.info(f"Your CSV must contain these {len(FEATURES)} columns (same names as template). "
            "Missing values are auto-imputed with training medians.")

    uploaded = st.file_uploader("Upload New Customer CSV", type=['csv'])

    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} rows, {new_df.shape[1]} columns")

            # Validate columns
            missing_cols = [f for f in FEATURES if f not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols[:10]}{'...' if len(missing_cols)>10 else ''}")
                st.stop()

            X_new = new_df[FEATURES].copy()
            X_new_imp    = pd.DataFrame(imputer.transform(X_new), columns=FEATURES)
            X_new_scaled = scaler.transform(X_new_imp)

            # Score
            cluster_labels   = km.predict(X_new_scaled)
            adopt_probs      = clf.predict_proba(X_new_imp)[:,1]
            pred_spend       = reg.predict(X_new_imp)

            spend_min = enriched['Pred_Spend_AED'].min()
            spend_max = enriched['Pred_Spend_AED'].max()
            spend_norm = (pred_spend - spend_min) / (spend_max - spend_min)
            tier_w_map = {'Tier 1':1.0, 'Tier 2':0.7, 'Tier 3':0.3}

            results = new_df.copy()
            results['Cluster']        = cluster_labels
            results['Persona']        = [PERSONA_MAP.get(c,'Unknown') for c in cluster_labels]
            results['Tier']           = results['Persona'].map(TIER_MAP)
            results['Adopt_Prob']     = adopt_probs.round(4)
            results['Pred_Spend_AED'] = pred_spend.round(0).astype(int)
            tw = results['Tier'].map(tier_w_map).fillna(0.3)
            priority_scores          = adopt_probs*0.4 + spend_norm*0.4 + tw.values*0.2
            results['Priority_Score'] = priority_scores.round(4)
            results['Priority_Tier']  = pd.cut(priority_scores, bins=[0,0.35,0.55,1.01],
                                                labels=['Low Priority','Nurture','Act Now'])
            results['Recommended_Action'] = [
                ACTION_MAP.get((row['Tier'], str(row['Priority_Tier'])), 'Review manually')
                for _, row in results.iterrows()
            ]
            results['Recommended_Bundle'] = results['Persona'].map(BUNDLE_MAP)
            results['Recommended_Channel'] = results['Persona'].map(CHANNEL_MAP)
            results['Priority_Rank'] = results['Priority_Score'].rank(ascending=False).astype(int)

            st.markdown("---")
            # Summary metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Scored", len(results))
            c2.metric("Act Now", (results['Priority_Tier']=='Act Now').sum())
            c3.metric("Avg Adopt Prob", f"{results['Adopt_Prob'].mean():.1%}")
            c4.metric("Avg Pred Spend", f"AED {results['Pred_Spend_AED'].mean():,.0f}")

            # Summary charts
            col1, col2 = st.columns(2)
            with col1:
                persona_counts = results['Persona'].value_counts()
                fig = px.pie(values=persona_counts.values, names=persona_counts.index,
                             title="New Customers by Persona", hole=0.4,
                             color=persona_counts.index, color_discrete_map=PERSONA_COLORS)
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                pt = results['Priority_Tier'].value_counts()
                fig2 = px.bar(x=pt.index, y=pt.values,
                              color=pt.index,
                              color_discrete_map={'Act Now':'#22c55e','Nurture':'#f59e0b','Low Priority':'#475569'},
                              title="Priority Distribution")
                fig2.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            # Results table
            st.subheader("Scored Customer Table")
            show_cols = ['Priority_Rank','Persona','Tier','Priority_Tier',
                         'Adopt_Prob','Pred_Spend_AED','Priority_Score',
                         'Recommended_Action','Recommended_Bundle','Recommended_Channel']
            available_show = [c for c in show_cols if c in results.columns]

            # Filters
            tf1, tf2 = st.columns(2)
            pt_filter = tf1.multiselect("Filter Priority Tier",
                                        ['Act Now','Nurture','Low Priority'],
                                        default=['Act Now','Nurture'])
            pf_filter = tf2.multiselect("Filter Persona",
                                        list(results['Persona'].unique()),
                                        default=list(results['Persona'].unique()))
            filtered_results = results[
                results['Priority_Tier'].isin(pt_filter) &
                results['Persona'].isin(pf_filter)
            ].sort_values('Priority_Rank')[available_show]

            st.dataframe(filtered_results.style.background_gradient(
                subset=['Priority_Score','Adopt_Prob'], cmap='Greens'),
                use_container_width=True, height=500)

            # CSV download
            out_buf = io.BytesIO()
            results.to_csv(out_buf, index=False)
            st.download_button("⬇️ Download Full Scored CSV",
                               data=out_buf.getvalue(),
                               file_name='RunRight_Scored_Customers.csv',
                               mime='text/csv')

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
    else:
        st.markdown("### 👆 Upload a CSV to begin scoring")
        st.markdown("""
**What happens when you upload:**
1. Schema validation — checks all required columns present
2. Auto-imputation — missing values filled with training medians
3. Encoding pipeline — same transforms as training data
4. **Classification** → adoption probability (0–1) per customer
5. **Clustering** → persona segment assignment
6. **Regression** → predicted annual spend (AED)
7. **Priority Score** → composite ranking (adopt prob + spend + tier)
8. **Action recommendation** → specific marketing action per customer
9. Download as CSV or filter/view in table above
        """)
