import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
from EKF import *
from RTS import *


def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = AudioSegment(y.tobytes(), frame_rate=sr,
                        sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


def extract_amplitude_from_mp3(file_path):
    """
    Extraire l'amplitude d'un fichier MP3 en fonction du temps

    Args:
        file_path (str): Chemin vers le fichier MP3

    Returns:
        tuple: (temps_en_secondes, amplitude, taux_echantillonnage)
    """
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    # Charger le fichier audio avec pydub
    audio = AudioSegment.from_mp3(file_path)

    # Extraire les données audio en forme d'onde
    samples = np.array(audio.get_array_of_samples())

    # Convertir en float32 et normaliser entre -1 et 1
    # if audio.sample_width == 2:  # 16-bit audio
    #     samples = samples / 32768.0
    # elif audio.sample_width == 1:  # 8-bit audio
    #     samples = samples / 128.0

    # Si l'audio est stéréo, prendre la moyenne des canaux
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)

    # Créer l'axe du temps
    duration_seconds = len(samples) / audio.frame_rate
    time_axis = np.linspace(0, duration_seconds, len(samples))

    return time_axis, samples, audio.frame_rate


def plot_amplitude(time_axis, amplitude, sample_rate):
    """
    Afficher l'amplitude en fonction du temps

    Args:
        time_axis (numpy.ndarray): Axe temporel en secondes
        amplitude (numpy.ndarray): Valeurs d'amplitude
        sample_rate (int): Taux d'échantillonnage
    """
    plt.figure(figsize=(12, 6))

    # Tracer l'amplitude complète
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, amplitude, linewidth=0.5)
    plt.title("Amplitude en fonction du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Tracer un zoom sur une petite partie (première seconde)
    plt.subplot(2, 1, 2)
    # Limiter à 1 seconde ou moins si l'audio est plus court
    zoom_end = min(1.0, time_axis[-1])
    zoom_indices = time_axis <= zoom_end
    plt.plot(time_axis[zoom_indices], amplitude[zoom_indices], linewidth=0.5)
    plt.title(
        f"Zoom sur la première seconde (taux d'échantillonnage: {sample_rate} Hz)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_filtered(time_axis, noisy_amplitude, filtered_amplitude, real_amplitude):
    """
    Afficher l'amplitude en fonction du temps

    Args:
        time_axis (numpy.ndarray): Axe temporel en secondes
        amplitude (numpy.ndarray): Valeurs d'amplitude
        sample_rate (int): Taux d'échantillonnage
    """
    plt.figure(figsize=(12, 6))

    # Tracer l'amplitude complète
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, noisy_amplitude, linewidth=0.5, label="Noisy dialogue")
    plt.plot(time_axis, real_amplitude, linewidth=0.5, label="True dialogue")
    plt.title("Amplitude en fonction du temps")
    plt.legend()
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Tracer un zoom sur une petite partie (première seconde)
    plt.subplot(2, 1, 2)
    # Limiter à 1 seconde ou moins si l'audio est plus court
    plt.plot(time_axis, noisy_amplitude, linewidth=0.5, label="Noisy dialogue")
    plt.plot(time_axis, filtered_amplitude,
             linewidth=0.5, label="Filtered dialogue")
    plt.title("Amplitude en fonction du temps")
    plt.legend()
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Chemin vers votre fichier MP3
    file_path1 = "audio/rain.mp3"  # Remplacez par le chemin de votre fichier
    filepath2 = "audio/dialogue.mp3"
    try:
        # Extraire l'amplitude
        time_axis1, amplitude1, sample_rate1 = extract_amplitude_from_mp3(
            file_path1)
        time_axis2, amplitude2, sample_rate2 = extract_amplitude_from_mp3(
            filepath2)

        # Afficher quelques informations
        print(f"Fichier: {file_path1}")
        print(f"Durée: {time_axis1[-1]:.2f} secondes")
        print(f"Taux d'échantillonnage: {sample_rate1} Hz")
        print(f"Nombre d'échantillons: {len(amplitude1)}")

        print(f"Fichier: {filepath2}")
        print(f"Durée: {time_axis2[-1]:.2f} secondes")
        print(f"Taux d'échantillonnage: {sample_rate2} Hz")
        print(f"Nombre d'échantillons: {len(amplitude2)}")

        # Afficher l'amplitude
        # plot_amplitude(time_axis1, amplitude1, sample_rate1)
        # plot_amplitude(time_axis2, amplitude2, sample_rate2)
        amplitude_noisy = amplitude2 + amplitude1[:len(amplitude2)]/4

        n_points_filtering = len(amplitude2)
        max = np.max(amplitude_noisy)
        real_amplitude = amplitude2[:n_points_filtering] / max / 100
        amplitude_noisy = amplitude_noisy[:n_points_filtering] / max/100

        time_axis2 = time_axis2[:n_points_filtering]

        sigma_e2_est = 0.00001
        sigma_w2_est = 0.000001
        sigma_a2_est = 0.000001
        AR_order_est = 2
        q = 20
        n_points = len(amplitude_noisy)

        means, covs, all_state_means, all_states_covs = Extended_Kalman_Filter(
            sigma_w2_est, sigma_a2_est, sigma_e2_est, q, AR_order_est, n_points, amplitude_noisy, plot_progress=True)

        # smoothed_means, smoothed_covs, means,covs = RTS_smoother(
        #     all_state_means, all_states_covs, q, AR_order_est, sigma_w2_est, sigma_a2_est, n_points, plot_progress=True)

        ### We keep only the part of the signal that is not noisy ie s>0.000005###
        amplitude_sup = np.abs(means) > 0.0001
        means[amplitude_sup] *= 3

        plot_filtered(time_axis2, amplitude_noisy, means, real_amplitude)

        # we multiply by 5 to get a higher amplitude,
        means *= max*100
    ### Change the previous line to obtain exactly the same amplitude as the original signal!!!###
        amplitude_noisy *= max*100
        real_amplitude = real_amplitude * max*100
        write("outputs/noiseless.wav", sample_rate2, real_amplitude)
        write("outputs/filtered.wav", sample_rate2, means)
        write("outputs/noisy.wav", sample_rate2, amplitude_noisy)

    except Exception as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()
