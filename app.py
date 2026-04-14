import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# PAGE CONFIG
st.set_page_config(page_title="Cricket Runs Predictor", layout="centered")

# UI STYLING
st.markdown("""
<style>

/* FULL PAGE BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #9333ea);
    background-attachment: fixed;
}

/* Center card */
section.main > div {
    display: flex;
    justify-content: center;
}

/* Glass effect card */
.block-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    padding: 35px;
    border-radius: 18px;
    max-width: 1000px;  
    width: 100%;        
    margin-top: 40px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
}

/* Title */
h1 {
    text-align: center;
    color: white;
}

/* Subtitle */
p {
    color: #e2e8f0;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 12px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}

/* Hover */
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px rgba(34,197,94,0.7);
}

/* Back button */
.stButton>button:nth-child(2) {
    background: linear-gradient(135deg, #ef4444, #dc2626);
}

/* Inputs */
label {
    font-weight: 600;
    color: white;
}

/* Divider */
hr {
    margin-top: 20px;
    margin-bottom: 20px;
    border-color: rgba(255,255,255,0.2);
}

.metric-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
    margin: 10px;   /* 🔥 adds space around each box */
}

/* spacing between columns */
.metric-container {
    display: flex;
    gap: 20px;   /* 🔥 creates separation */
}

</style>
""", unsafe_allow_html=True)

# LOAD DATA
df = pd.read_csv("merged_dataset.csv")

data = pickle.load(open("cricket_lgbm_model.pkl", "rb"))
model = data['model']
scaler = data['scaler']
le_player = data['player_encoder']
le_team = data['team_encoder']

# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = "input"

# ------------------ PAGE 1 ------------------
if st.session_state.page == "input":

    st.title("🏏 Cricket Runs Predictor")
    st.markdown(
        "<p style='text-align:center;color:gray;'>Predict runs using ML model</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # Match Selection
    # 🔥 GET UNIQUE TEAMS FROM MATCH COLUMN
    teams = sorted(set(
        team.strip()
        for match_name in df['name_y'].dropna()
        for team in match_name.replace("vs", "v").split("v")
    ))

# SELECT TEAMS SEPARATELY
    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("🏏 Team 1", teams)

    with col2:
        team2 = st.selectbox("🏏 Team 2", teams)

# CREATE MATCH STRING FOR MODEL
    match = f"{team1} v {team2}"

    # Filter players
    filtered_players = df[
    df['name_y'].str.contains(team1, case=False) &
    df['name_y'].str.contains(team2, case=False)
]['name_x'].dropna().unique()

    player = st.selectbox(
        "👤 Select Player",
        sorted(filtered_players)
    )

    # Inputs
    col1, col2 = st.columns(2)

    with col1:
        balls = st.number_input("🏏 Balls Faced", min_value=0, value=30)

    with col2:
        over = st.number_input("⏱ Over", min_value=0.0, value=10.0)

    st.divider()

    # PREDICT BUTTON
    if st.button("🚀 Predict Runs", use_container_width=True):

        with st.spinner("Predicting... ⏳"):
            try:
                # 🔥 SPLIT TEAMS
                try:
                    match_clean = match.lower().replace("vs", "v")
                    team1, team2 = [t.strip() for t in match_clean.split("v")]
                except:
                    team1, team2 = "Unknown", "Unknown"

                # Encode
                player_id = le_player.transform([player])[0]
                match_id = le_team.transform([match])[0]

                # Input Data
                input_data = pd.DataFrame([{
                    'balls': balls,
                    'over_x': over,
                    'player_encoded': player_id,
                    'team_encoded': match_id
                }])

                input_data[['balls','over_x']] = scaler.transform(
                    input_data[['balls','over_x']]
                )

                # Prediction
                pred = model.predict(input_data)
                pred = np.expm1(pred)

                # Store values
                st.session_state.result = int(pred[0])
                st.session_state.player = player
                st.session_state.match = match
                st.session_state.team1 = team1
                st.session_state.team2 = team2

                st.session_state.page = "result"
                st.rerun()

            except Exception as e:
                st.error("❌ Error in prediction")
                st.write(e)

# ------------------ PAGE 2 ------------------
elif st.session_state.page == "result":

    st.title("📊 Prediction Result")

    # RESULT CARD
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#0f172a,#1e293b);
        padding:35px;
        border-radius:18px;
        text-align:center;
        box-shadow:0px 0px 20px rgba(0,0,0,0.3)
    ">
        <h2 style="color:white;">🏏 {st.session_state.player}</h2>
        <h4 style="color:#38bdf8;">
            {st.session_state.team1.upper()} 🆚 {st.session_state.team2.upper()}
        </h4>
        <h1 style="color:#22c55e;font-size:50px;">
            {st.session_state.result} Runs
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ------------------ EXTRA ANALYSIS ------------------
    balls = st.session_state.get("balls", 30)
    runs = st.session_state.result

    # Strike Rate
    if balls > 0:
        strike_rate = (runs / balls) * 100
    else:
        strike_rate = 0

    # Performance + color
    if runs < 20:
        performance = "Poor"
        perf_color = "#ef4444"   # red
    elif runs < 50:
        performance = "Average"
        perf_color = "#f59e0b"   # yellow
    else:
        performance = "Good"
        perf_color = "#22c55e"   # green

    # ------------------ WHITE CARDS ------------------

    st.divider()

    col1, col2 = st.columns(2, gap="large")  # 🔥 important

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:16px;color:#555;">⚡ Strike Rate</div>
            <div style="font-size:30px;font-weight:bold;color:#3b82f6;">
            {strike_rate:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:16px;color:#555;">📊 Performance</div>
            <div style="font-size:30px;font-weight:bold;color:{perf_color};">
                {performance}
            </div>
        </div>
        """, unsafe_allow_html=True)
    # ------------------ GRAPH ------------------
    import plotly.graph_objects as go

    st.divider()

    if balls > 0:

        balls_list = np.arange(1, balls + 1)

        # 🔥 Curve based on strike rate
        if strike_rate < 90:
            curve = np.log(balls_list + 1)
            line_color = "#ef4444"   # red
        elif strike_rate < 120:
            curve = balls_list ** 1.2
            line_color = "#f59e0b"   # yellow
        else:
            curve = balls_list ** 1.5
            line_color = "#22c55e"   # green

    # Normalize to predicted runs
        curve = curve / max(curve) * runs

    # 🎨 Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=balls_list,
            y=curve,
            mode='lines+markers',
            line=dict(color=line_color, width=4),
            marker=dict(size=6),
            fill='tozeroy',   # 🔥 area fill
        ))

    # 🎨 Layout styling
        fig.update_layout(
            title="📈 Runs Progression",
            xaxis_title="Balls",
            yaxis_title="Runs",
            template="plotly_dark",   # 🔥 modern dark theme
            title_x=0.3
        )

        st.plotly_chart(fig, use_container_width=True)

    # BACK BUTTON
    if st.button("⬅ Back", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()