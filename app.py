import streamlit as st

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="AI Playlist Generator", page_icon="🎧", layout="wide")

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import uuid
import gdown
import time

# ==========================================
# ❤️ Initialize Session State
# ==========================================
if "liked_songs" not in st.session_state:
    st.session_state.liked_songs = []
if "playlist_df" not in st.session_state:
    st.session_state.playlist_df = pd.DataFrame()



# ==========================================
# 🎧 DQN Model Definition
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# ⚙️ Load Model & Dataset
# ==========================================
@st.cache_resource
def load_model_and_data():
    csv_path = "spotify_clean_subset.csv"
    model_path = "trained_dqn_playlist_final_v2.pth"
    gdrive_id = "1q8BfoClD5mjMTD1658Vf7rHJ9Mu7ihCa"

    # ✅ Load dataset
    if not os.path.exists(csv_path):
        st.error("❌ 'spotify_clean_subset.csv' file is missing in your app directory.")
        st.stop()
    df = pd.read_csv(csv_path)
    features = ['tempo', 'energy', 'valence', 'acousticness', 'popularity']

    # ✅ Download model if missing
    if not os.path.exists(model_path):
        with st.spinner("⏳ Downloading trained model from Google Drive... This may take a minute."):
            try:
                url = f"https://drive.google.com/uc?id={gdrive_id}"
                gdown.download(url, model_path, quiet=False)
                time.sleep(2)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                st.stop()
    else:
        st.info("📁 Model found locally — skipping download.")

    # ✅ Load the model safely
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        q_net = DQN(len(features), len(df))
        if 'q_net_state_dict' in checkpoint:
            q_net.load_state_dict(checkpoint['q_net_state_dict'], strict=False)
        else:
            st.warning("⚠️ 'q_net_state_dict' missing in checkpoint — using random weights.")
        q_net.eval()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    st.success("✅ Model and dataset loaded successfully.")
    return q_net, df, features

# ✅ Load model and data
q_net, df, features = load_model_and_data()

# ==========================================
# 🎨 Spotify API Setup
# ==========================================
client_id = "b166c71602014354b5c135a15c185f74"
client_secret = "fd57730a9ef541d8974c78205125d50b"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

def get_spotify_data(track_name, artist_name):
    try:
        query = f"{track_name} {artist_name}"
        result = sp.search(q=query, limit=1, type="track")
        if result["tracks"]["items"]:
            track = result["tracks"]["items"][0]
            return {
                "cover_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                "preview_url": track["preview_url"],
                "spotify_url": track["external_urls"]["spotify"]
            }
    except:
        pass
    return {"cover_url": None, "preview_url": None, "spotify_url": None}

# ==========================================
# 🧠 Playlist Generator
# ==========================================
def generate_playlist(q_net, df, start_genre=None, mood=None, playlist_len=10, top_k=20):
    if start_genre and start_genre != "Any":
        df = df[df['genre'].str.lower() == start_genre.lower()]
    if mood and mood != "Any":
        if mood == "Energetic":
            df = df[df['energy'] > 0.7]
        elif mood == "Calm":
            df = df[df['energy'] < 0.4]
        elif mood == "Happy":
            df = df[df['valence'] > 0.6]
        elif mood == "Sad":
            df = df[df['valence'] < 0.3]

    df = df.reset_index(drop=True)
    if df.empty:
        st.error("⚠️ No songs match your filters.")
        return pd.DataFrame()

    idx = random.randint(0, len(df) - 1)
    cur_song = df.iloc[idx]
    playlist = [cur_song]
    used = {idx}

    for _ in range(playlist_len - 1):
        state = torch.tensor(cur_song[features].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state).flatten()
            q_values += torch.rand_like(q_values) * 0.01
        top_indices = torch.topk(q_values, min(top_k, len(df))).indices.cpu().numpy()
        candidates = [i for i in top_indices if i not in used]
        if not candidates:
            break
        next_idx = random.choice(candidates)
        cur_song = df.iloc[next_idx]
        playlist.append(cur_song)
        used.add(next_idx)

    return pd.DataFrame(playlist).drop_duplicates(subset=['track_name']).reset_index(drop=True)

# ==========================================
# 🎨 Streamlit UI
# ==========================================

st.markdown("<h1 style='text-align:center;color:#1DB954;'>🎧 AI Playlist Generator</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Personalized Mood-Based Playlists using Reinforcement Learning + Spotify API</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ Playlist Settings")
playlist_len = st.sidebar.slider("Playlist length", 5, 20, 10)
top_k = st.sidebar.slider("Exploration (Top-K Sampling)", 5, 50, 20)
start_genre = st.sidebar.selectbox("Preferred Genre", ["Any"] + sorted(df['genre'].unique().tolist()))
mood = st.sidebar.selectbox("Select Mood 🎭", ["Any", "Happy", "Calm", "Energetic", "Sad"])

if st.button("✨ Generate Playlist"):
    st.session_state.playlist_df = generate_playlist(q_net, df, start_genre, mood, playlist_len, top_k)
    st.success("✅ Playlist Generated Successfully!")

# ==========================================
# 🎶 Display Playlist + Likes
# ==========================================
if not st.session_state.playlist_df.empty:
    st.markdown("### 🎧 Generated Playlist")
    for i, row in st.session_state.playlist_df.iterrows():
        meta = get_spotify_data(row['track_name'], row['artist_name'])
        cover = meta["cover_url"] or "https://via.placeholder.com/100x100?text=No+Cover"
        song_key = f"{row['track_name']}_{i}"

        with st.container():
            cols = st.columns([1, 3, 1])
            with cols[0]:
                st.image(cover, width=90)
            with cols[1]:
                st.markdown(f"**{row['track_name']}**  \n*{row['artist_name']}*  \n🎵 *{row['genre']}*")
                if meta["preview_url"]:
                    st.audio(meta["preview_url"], format="audio/mp3")
                elif meta["spotify_url"]:
                    track_id = meta["spotify_url"].split("/")[-1]
                    st.markdown(f"""
                    <iframe src="https://open.spotify.com/embed/track/{track_id}"
                            width="100%" height="80" frameborder="0"
                            allowtransparency="true" allow="encrypted-media"></iframe>
                    """, unsafe_allow_html=True)
            with cols[2]:
                if st.button("❤️ Like", key=f"like_{song_key}"):
                    entry = {"track": row['track_name'], "artist": row['artist_name'], "spotify_url": meta["spotify_url"]}
                    if entry not in st.session_state.liked_songs:
                        st.session_state.liked_songs.append(entry)
                        st.toast(f"💖 Added {row['track_name']} to Liked Songs!")
                if meta["spotify_url"]:
                    st.markdown(f"[🔗 Open in Spotify]({meta['spotify_url']})")

# ==========================================
# 💖 Liked Songs
# ==========================================
if st.session_state.liked_songs:
    st.markdown("---")
    st.markdown("### 💖 My Liked Songs")
    for s in st.session_state.liked_songs:
        st.markdown(f"• **[{s['track']}]({s['spotify_url']})** — *{s['artist']}*")

    if st.button("🧹 Clear Liked Songs"):
        st.session_state.liked_songs.clear()
        st.success("✅ Liked songs cleared!")

# ==========================================
# 🔍 Search Section
# ==========================================
st.markdown("---")
st.markdown("## 🔍 Search Songs on Spotify")

query = st.text_input("Search a song or artist", placeholder="e.g., Blinding Lights or Arijit Singh")
if st.button("🔎 Search Spotify"):
    if query.strip():
        results = sp.search(q=query, limit=5, type="track")
        for track in results["tracks"]["items"]:
            st.markdown(f"**{track['name']}** — *{track['artists'][0]['name']}*")
            cover = track["album"]["images"][0]["url"] if track["album"]["images"] else None
            if cover:
                st.image(cover, width=100)
            st.markdown(f"""
            <iframe src="https://open.spotify.com/embed/track/{track['id']}"
                    width="100%" height="80" frameborder="0"
                    allowtransparency="true" allow="encrypted-media"></iframe>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a search query.")

# ==========================================
# 📝 Footer
# ==========================================
st.markdown("---")
st.markdown(
    "<center><h4>🎶 Built with ❤️ using PyTorch + Streamlit + Spotify API<br>SR University — AI Playlist Project</h4></center>",
    unsafe_allow_html=True
)
