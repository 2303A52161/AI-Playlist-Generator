import streamlit as st
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
    # Load CSV
    csv_path = "spotify_clean_subset.csv"
    if not os.path.exists(csv_path):
        st.error("❌ 'spotify_clean_subset.csv' file is missing.")
        st.stop()

    df = pd.read_csv(csv_path)
    features = ['tempo', 'energy', 'valence', 'acousticness', 'popularity']

    # Load model
    model_path = "trained_dqn_playlist_final_v2.pth"
    gdrive_id = "1V8B5GRRX3rOY4BucMjukPiwTK_oLdDoa"  
    if not os.path.exists(model_path):
        st.warning("Downloading model from Google Drive ⏳ ...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")

    checkpoint = torch.load(model_path, map_location="cpu")
    q_net = DQN(len(checkpoint['features']), len(df))
    q_net.load_state_dict(checkpoint['q_net_state_dict'], strict=False)
    q_net.eval()

    st.success("✅ Model and dataset loaded successfully.")
    return q_net, df, features



# Load model and data once
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
# 🧠 Helper Functions
# ==========================================
def get_state(song):
    return torch.tensor(song[features].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)

def generate_playlist(q_net, df, start_genre=None, mood=None, playlist_len=10, top_k=20):
    """
    Generate a playlist using the trained DQN model with genre & mood filters.
    Prevents duplicate songs and index-out-of-bound errors.
    """

    # 🎧 Apply genre filter
    if start_genre and start_genre != "Any":
        genre_subset = df[df['genre'].str.lower() == start_genre.lower()]
        if genre_subset.empty:
            st.warning(f"⚠️ No tracks found for genre '{start_genre}'. Using full dataset.")
        else:
            df = genre_subset.reset_index(drop=True)

    # 🎭 Apply mood filter
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
    n_songs = len(df)
    if n_songs == 0:
        st.error("⚠️ No songs match the selected filters.")
        return pd.DataFrame()

    # 🎵 Start with a random song
    idx = random.randint(0, n_songs - 1)
    cur_song = df.iloc[idx]
    playlist = [cur_song]
    used_indices = {idx}

    for _ in range(playlist_len - 1):
        # Prepare state vector safely
        state = torch.tensor(
            cur_song[features].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(0)

        # Predict Q-values and add slight randomness for exploration
        with torch.no_grad():
            q_values = q_net(state).flatten()
            q_values += torch.rand_like(q_values) * 0.01  # adds variety

        k = min(top_k, n_songs)
        topk_indices = torch.topk(q_values, k).indices.cpu().numpy()

        # ⚡ Only keep valid indices
        topk_indices = [i for i in topk_indices if 0 <= i < n_songs]
        candidates = [i for i in topk_indices if i not in used_indices]

        # Handle out-of-bound or exhausted candidate cases
        if not candidates:
            remaining = list(set(range(n_songs)) - used_indices)
            if not remaining:
                break  # no more songs left
            next_idx = random.choice(remaining)
        else:
            next_idx = random.choice(candidates)

        # ✅ Double-check that index is valid before accessing
        if 0 <= next_idx < n_songs:
            next_song = df.iloc[next_idx]
            playlist.append(next_song)
            used_indices.add(next_idx)
            cur_song = next_song
        else:
            continue  # skip invalid index safely

    # ✅ Remove duplicates by track name
    playlist_df = pd.DataFrame(playlist).drop_duplicates(subset=['track_name']).reset_index(drop=True)
    return playlist_df



# ==========================================
# 🎨 Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Playlist Generator", page_icon="🎧", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1DB954;'>🎧 AI Playlist Generator</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Personalized Mood-Based Playlists using Reinforcement Learning + Spotify API</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ Playlist Settings")
playlist_len = st.sidebar.slider("Playlist length", 5, 20, 10)
top_k = st.sidebar.slider("Exploration (Top-K Sampling)", 5, 50, 20)
start_genre = st.sidebar.selectbox("Preferred Genre", ["Any"] + sorted(df['genre'].unique().tolist()))
mood = st.sidebar.selectbox("Select Mood 🎭", ["Any", "Happy", "Calm", "Energetic", "Sad"])

# Generate Button
if st.button("✨ Generate Playlist"):
    st.session_state.playlist_df = generate_playlist(q_net, df, start_genre=start_genre, mood=mood, playlist_len=playlist_len, top_k=top_k)
    st.success("✅ Playlist Generated Successfully!")

# ==========================================
# 🎶 Display Generated Playlist (Fixed Likes)
# ==========================================
if not st.session_state.playlist_df.empty:
    st.markdown("### 🎧 Generated Playlist")
    
    for i, row in st.session_state.playlist_df.iterrows():
        meta = get_spotify_data(row['track_name'], row['artist_name'])
        cover = meta["cover_url"] or "https://via.placeholder.com/100x100?text=No+Cover"
        
        # Unique key based on song + index
        song_key = f"{row['track_name']}_{i}"
        
        with st.container():
            cols = st.columns([1, 3, 1])
            
            with cols[0]:
                st.image(cover, width=90)
            
            with cols[1]:
                st.markdown(f"**{row['track_name']}**  \n*{row['artist_name']}*  \n🎵 *{row['genre']}*")
                
                # Play audio preview or Spotify embed
                if meta["preview_url"]:
                    st.audio(meta["preview_url"], format="audio/mp3")
                elif meta["spotify_url"]:
                    track_id = meta["spotify_url"].split("/")[-1]
                    embed_html = f"""
                    <iframe src="https://open.spotify.com/embed/track/{track_id}"
                            width="100%" height="80" frameborder="0"
                            allowtransparency="true" allow="encrypted-media"></iframe>
                    """
                    st.markdown(embed_html, unsafe_allow_html=True)
                else:
                    st.info("🎧 No preview or embed available.")
            
            with cols[2]:
                like_button = st.button("❤️ Like", key=f"like_{song_key}")
                if like_button:
                    liked_entry = {
                        "track": row['track_name'],
                        "artist": row['artist_name'],
                        "spotify_url": meta["spotify_url"]
                    }
                    # Prevent duplicates
                    if liked_entry not in st.session_state.liked_songs:
                        st.session_state.liked_songs.append(liked_entry)
                        st.toast(f"💖 Added {row['track_name']} to Liked Songs!")
                
                if meta["spotify_url"]:
                    st.markdown(f"[🔗 Open in Spotify]({meta['spotify_url']})")


# ==========================================
# 💖 Display Liked Songs
# ==========================================
if st.session_state.liked_songs:
    st.markdown("---")
    st.markdown("### 💖 My Liked Songs")
    for song in st.session_state.liked_songs:
        st.markdown(f"• **[{song['track']}]({song['spotify_url']})** — *{song['artist']}*")

    if st.button("🧹 Clear Liked Songs"):
        st.session_state.liked_songs.clear()
        st.success("Cleared liked songs list!")

# ==========================================
# 🔍 Spotify Search Section
# ==========================================
st.markdown("---")
st.markdown("## 🔍 Search Songs on Spotify")

search_query = st.text_input("Type a song or artist name", placeholder="e.g., Blinding Lights or Arijit Singh")

if st.button("🔎 Search Spotify"):
    if search_query.strip():
        results = sp.search(q=search_query, limit=5, type="track")
        tracks = results["tracks"]["items"]

        if len(tracks) == 0:
            st.warning("No songs found. Try another search!")
        else:
            for track in tracks:
                track_name = track["name"]
                artist = track["artists"][0]["name"]
                album = track["album"]["name"]
                url = track["external_urls"]["spotify"]
                cover = track["album"]["images"][0]["url"] if track["album"]["images"] else "https://via.placeholder.com/150"

                embed_html = f"""
                <iframe src="https://open.spotify.com/embed/track/{track['id']}"
                        width="100%" height="80" frameborder="0"
                        allowtransparency="true" allow="encrypted-media"></iframe>
                """

                with st.container():
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.image(cover, width=100)
                    with cols[1]:
                        st.markdown(f"**{track_name}** — *{artist}*  \n💿 {album}")
                        st.markdown(embed_html, unsafe_allow_html=True)
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
