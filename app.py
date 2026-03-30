import streamlit as st
import pickle
import numpy as np
import pandas as pd

# PAGE CONFIG
st.set_page_config(page_title="Cricket Runs Predictor", layout="centered")

# UI STYLING
st.markdown("""
<style>

/* FULL PAGE BACKGROUND (CONTRAST GRADIENT) */
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
    max-width: 720px;
    margin-top: 40px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
}

/* Title styling */
h1 {
    text-align: center;
    color: white;
}

/* Subtitle */
p {
    color: #e2e8f0;
}

/* BUTTON - CONTRAST COLOR */
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

/* Hover effect */
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

# PAGE 1 - INPUT
if st.session_state.page == "input":

    st.title("🏏 Cricket Runs Predictor")
    st.markdown(
        "<p style='text-align:center;color:gray;'>Predict runs using ML model</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # Match
    match = st.selectbox(
        "🏟 Select Match",
        sorted(df['name_y'].dropna().unique())
    )

    # Filter players
    filtered_players = df[df['name_y'] == match]['name_x'].dropna().unique()

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

    # BUTTON WITH SPINNER
    if st.button("🚀 Predict Runs", use_container_width=True):

        with st.spinner("Predicting... ⏳"):
            try:
                player_id = le_player.transform([player])[0]
                match_id = le_team.transform([match])[0]

                input_data = pd.DataFrame([{
                    'balls': balls,
                    'over_x': over,
                    'player_encoded': player_id,
                    'team_encoded': match_id
                }])

                input_data[['balls','over_x']] = scaler.transform(
                    input_data[['balls','over_x']]
                )

                pred = model.predict(input_data)
                pred = np.expm1(pred)

                st.session_state.result = int(pred[0])
                st.session_state.player = player
                st.session_state.match = match

                st.session_state.page = "result"
                st.rerun()

            except Exception as e:
                st.error("❌ Error in prediction")
                st.write(e)

# PAGE 2 - RESULT
elif st.session_state.page == "result":

    st.title("📊 Prediction Result")

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#0f172a,#1e293b);
        padding:35px;
        border-radius:18px;
        text-align:center;
        box-shadow:0px 0px 20px rgba(0,0,0,0.3)
    ">
        <h2 style="color:white;">🏏 {st.session_state.player}</h2>
        <h4 style="color:#38bdf8;">{st.session_state.match}</h4>
        <h1 style="color:#22c55e;font-size:50px;">
            {st.session_state.result} Runs
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button("⬅ Back", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()