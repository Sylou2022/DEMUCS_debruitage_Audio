import torch
from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import streamlit as st
import soundfile as sf
import base64
import tempfile
import uuid

# Charger le modèle pré-entraîné de Demucs
model = pretrained.get_model('mdx_extra')

# Configuration de la page Streamlit
st.set_page_config(page_title="Traitement Audio", page_icon="🎵", layout="wide")

# Ajout de l'image d'accueil avec style
def add_background(image_path):
    with open(image_path, "rb") as file:  # Ouvrir en mode binaire
        image_data = base64.b64encode(file.read()).decode()  # Encoder en base64
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{image_data});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background("image_audio.jpeg")  # Remplacez par le chemin vers votre image

# Afficher le titre et une description sur la page d'accueil
st.title("🎵 Traitement Audio : Détection et Débruitage")
st.markdown(
    "<p style='color: white; font-size: 18px;'>"
    "Bienvenue dans notre application de traitement audio. "
    "Téléchargez un fichier audio pour détecter ou réduire les bruits."
    "</p>",
    unsafe_allow_html=True
)

# Ajouter des onglets dans une barre latérale
tab = st.sidebar.radio("Navigation", ["Accueil", "Correction rapide", "Détectage de Bruit", "Débruitage"])

if tab == "Accueil":
    st.write("")  # Pour afficher uniquement l'image en arrière-plan

elif tab == "Débruitage":
    st.header("Débruitage")
    uploaded_file = st.file_uploader("Téléchargez un fichier audio (MP3 ou WAV)", type=["mp3", "wav"])

    if uploaded_file is not None:
        try:
            waveform, sr = torchaudio.load(uploaded_file, backend="soundfile")
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'audio : {str(e)}")
            st.stop()

        # Ajouter une dimension pour le lot
        waveform = waveform.unsqueeze(0)  # Pour [1, canaux, longueur]

        # Appliquer le modèle pour séparer les sources
        with st.spinner("Isolation des sources..."):
            sources = apply_model(model, waveform, split=True)

        # Sélectionner la source vocale (index 3 dans 'mdx_extra')
        vocal_source = sources[0, 3, :, :]
        vocal_source = torch.mean(vocal_source, dim=0)  # Conversion mono

        # Sauvegarder la voix isolée sans bruit
        temp_dir = tempfile.gettempdir()
        isolated_voice_path = f"{temp_dir}/voix_isolee_{uuid.uuid4().hex}.wav"
        sf.write(isolated_voice_path, vocal_source.cpu().numpy(), sr)

        # Jouer et proposer de télécharger la voix isolée
        st.subheader("Voix isolée")
        st.audio(isolated_voice_path, format="audio/wav", start_time=0)

        with open(isolated_voice_path, "rb") as file:
            st.download_button(
                label="Télécharger la voix isolée",
                data=file,
                file_name="voix_isolee.wav",
                mime="audio/wav",
            )

elif tab == "Détectage de Bruit":
    st.header("Détectage de Bruit")
    uploaded_file = st.file_uploader("Téléchargez un fichier audio (MP3 ou WAV)", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Charger le fichier audio
        waveform, sr = torchaudio.load(uploaded_file, backend="soundfile")
        waveform = waveform.unsqueeze(0)

        # Appliquer le modèle pour séparer les sources
        with st.spinner("Extraction du bruit..."):
            sources = apply_model(model, waveform, split=True)

        # Extraire la source de bruit
        noise_source = sources[0, 2, :, :].cpu().numpy()
        noise_source = np.mean(noise_source, axis=0)

        # Calcul de l'énergie RMS
        frame_size = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=noise_source, frame_length=frame_size, hop_length=hop_length).flatten()

        # Détection des intervalles de bruit
        threshold = np.mean(rms) + 1.5 * np.std(rms)
        in_noise = False
        noise_intervals = []

        for i, energy in enumerate(rms):
            if energy > threshold and not in_noise:
                start_time = i * hop_length / sr
                in_noise = True
            elif energy <= threshold and in_noise:
                end_time = i * hop_length / sr
                noise_intervals.append((start_time, end_time))
                in_noise = False

        # Ajouter le dernier intervalle si bruit continu jusqu'à la fin
        if in_noise:
            noise_intervals.append((start_time, len(noise_source) / sr))

        # Affichage graphique des segments de bruit
        st.subheader("Segments de Bruit Détectés")
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(waveform.squeeze().numpy(), sr=sr, alpha=0.5, color="blue", ax=ax)
        for start, end in noise_intervals:
            ax.axvspan(start, end, color="red", alpha=0.3)
        ax.set_xlabel("Temps (secondes)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Détection des intervalles de bruit continu")
        st.pyplot(fig)

        # Message indiquant le nombre d'intervalles détectés
        if noise_intervals:
            st.success(f"Nombre de segments de bruit détectés : {len(noise_intervals)}.")
        else:
            st.info("Aucun bruit continu détecté.")

        # Sauvegarder le bruit isolé
        noise_path = "bruit_isole.wav"
        sf.write(noise_path, noise_source, sr)

        # Jouer et proposer de télécharger le bruit isolé
        st.subheader("Bruit isolé")
        st.audio(noise_path, format="audio/wav", start_time=0)

        with open(noise_path, "rb") as file:
            st.download_button(
                label="Télécharger le bruit isolé",
                data=file,
                file_name="bruit_isole.wav",
                mime="audio/wav",
            )

elif tab == "Correction rapide":
    st.header("Correction rapide")

    # Permet à l'utilisateur de télécharger un fichier audio
    uploaded_file = st.file_uploader("Téléchargez un fichier audio (MP3 ou WAV)", type=["mp3", "wav"])

    if uploaded_file is not None:
        try:
            # Charger le fichier audio
            waveform, sr = torchaudio.load(uploaded_file, backend="soundfile")

            # Ajouter une dimension pour le lot
            waveform = waveform.unsqueeze(0)  # Pour [1, canaux, longueur]

            # Appliquer le modèle pour séparer les sources
            with st.spinner("Séparation des sources..."):
                sources = apply_model(model, waveform, split=True)

            # Sélectionner la source vocale (index 3 dans 'mdx_extra')
            vocal_source = sources[0, 3, :, :]
            vocal_source = torch.mean(vocal_source, dim=0)  # Conversion mono

            # Sauvegarder la voix isolée sans bruit
            temp_dir = tempfile.gettempdir()
            isolated_voice_path = f"{temp_dir}/voix_isolee_{uuid.uuid4().hex}.wav"
            sf.write(isolated_voice_path, vocal_source.cpu().numpy(), sr)

            # Appliquer un filtre passe-bas pour lisser le signal
            voice, sr = sf.read(isolated_voice_path)
            sos = signal.butter(10, 1500, 'lp', fs=sr, output='sos')
            filtered_voice = signal.sosfilt(sos, voice)

            # Sauvegarder le fichier filtré
            filtered_voice_path = f"{temp_dir}/voix_filtre_{uuid.uuid4().hex}.wav"
            sf.write(filtered_voice_path, filtered_voice, sr)

            # Afficher et proposer de télécharger la voix filtrée
            st.subheader("Voix filtrée")
            st.audio(filtered_voice_path, format="audio/wav", start_time=0)

            with open(filtered_voice_path, "rb") as file:
                st.download_button(
                    label="Télécharger la voix filtrée",
                    data=file,
                    file_name="voix_filtre.wav",
                    mime="audio/wav",
                )
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'audio : {str(e)}")
            st.stop()
