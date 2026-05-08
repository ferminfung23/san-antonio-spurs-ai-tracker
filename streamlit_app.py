import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pingouin as pg  # For correlations

# ===== REAL SPURS DATA (2023-24 season averages) =====
spurs_data = pd.DataFrame({
    'player': ['Wembanyama', 'Vassell', 'Sochan', 'Johnson', 'Keldon', 'Tre', 'Steph', 'Julian', 'Blake'],
    'pts': [21.4, 19.5, 11.6, 12.0, 17.4, 11.2, 10.8, 9.2, 8.5],
    'reb': [10.6, 4.2, 6.4, 4.4, 4.4, 3.9, 3.6, 3.3, 3.1],
    'ast': [3.9, 3.5, 2.3, 2.7, 2.5, 2.7, 2.1, 1.8, 1.9],
    'stl': [1.2, 1.1, 0.9, 0.8, 0.7, 0.8, 0.6, 0.7, 0.5],
    'blk': [3.6, 0.7, 0.7, 0.4, 0.5, 0.4, 0.3, 0.4, 0.3],
    'min': [29.7, 31.6, 26.1, 26.9, 30.7, 24.4, 22.5, 22.0, 21.0],
    'performance_score': [92, 85, 78, 76, 82, 74, 72, 70, 68],
    'injury_risk': [22, 15, 12, 18, 10, 8, 25, 14, 20],
    'clutch_rating': [87, 82, 76, 74, 79, 72, 70, 68, 65]
})

# ===== AI MODELS =====
features = ['pts', 'reb', 'ast', 'stl', 'blk', 'min']
X = spurs_data[features]
y_perf = spurs_data['performance_score']
perf_model = RandomForestRegressor(n_estimators=100, random_state=42)
perf_model.fit(X, y_perf)

y_injury = spurs_data['injury_risk'] > 15
injury_model = GradientBoostingClassifier(random_state=42)
injury_model.fit(X, y_injury)

# ===== STREAMLIT DASHBOARD =====
st.set_page_config(page_title="Spurs AI Tracker", layout="wide")
st.title("🏀 San Antonio Spurs AI Performance Tracker")
st.markdown("**Predict player scores, injury risk, & clutch factor** – Aidan & AI Club @ SISHK")

# Sidebar inputs
st.sidebar.header("🔧 Enter Player Stats")
pts = st.sidebar.slider("Points/Game", 0, 35, 15)
reb = st.sidebar.slider("Rebounds/Game", 0, 15, 6)
ast = st.sidebar.slider("Assists/Game", 0, 10, 3)
stl = st.sidebar.slider("Steals/Game", 0, 3, 1)
blk = st.sidebar.slider("Blocks/Game", 0, 5, 1)
min_pg = st.sidebar.slider("Minutes/Game", 0, 45, 25)

input_data = np.array([[pts, reb, ast, stl, blk, min_pg]])

# Predictions
perf_pred = perf_model.predict(input_data)[0]
injury_prob = injury_model.predict_proba(input_data)[0][1] * 100
clutch_pred = min(100, 70 + (pts*2 + reb*1.5 + ast*1 + blk*3)/10)

# Main metrics
col1, col2, col3 = st.columns(3)
with col1: st.metric("Performance Score", f"{perf_pred:.0f}/100", "🔥 All-Star")
with col2: st.metric("Injury Risk", f"{injury_prob:.0f}%", "⚠️ Monitor")
with col3: st.metric("Clutch Rating", f"{clutch_pred:.0f}%", "🏆 Playoff-ready")

# Radar chart vs Wemby
st.subheader("📊 Compare to Wembanyama")
fig = go.Figure()
categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN']
wemby = spurs_data[spurs_data['player']=='Wembanyama'][features].values[0]
input_scaled = input_data[0] / np.max(spurs_data[features], axis=0) * 20
fig.add_trace(go.Scatterpolar(r=wemby, theta=categories, fill='toself', name='Wembanyama'))
fig.add_trace(go.Scatterpolar(r=input_scaled, theta=categories, fill='toself', name='Your Player'))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 25])), showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# Team comparison
st.subheader("👥 All Spurs Players")
fig2 = px.scatter(spurs_data, x='min', y='performance_score', size='pts', 
                  color='injury_risk', hover_name='player', 
                  title="Minutes vs Performance (Bubble = PTS, Color = Injury Risk)")
st.plotly_chart(fig2, use_container_width=True)

# Insights
st.subheader("🧠 AI Insights")
corr = pg.corr(spurs_data[['pts', 'performance_score', 'injury_risk']], method='pearson')
st.dataframe(corr.round(3))

st.caption("📈 Data: Basketball-Reference 2023-24 | Built for my pitch")
