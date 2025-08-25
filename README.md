# Spoken Digit Recognition in PyTorch

This repository contains a complete, end-to-end workflow for building and comparing two high-performance spoken digit recognition models. The project demonstrates a full machine learning lifecycle, from a high-accuracy baseline to a real-time application with microphone integration, implemented entirely in PyTorch within Google Colab notebooks.

The model achieves **99.33% accuracy** on the clean test set while the robust model demonstrates strong performance on live microphone audio with **98% accuracy**.

---

## Repository Structure & Models

This project is organized into two separate notebooks, each building a distinct model with a specific purpose.

### 1. The Baseline Model
-   **Notebook:** [`1_baseline_Digit_recognition.ipynb`](https://colab.research.google.com/drive/1JulNqWlnbPrhhWJyba6S1eT_VdcbDmnT?usp=sharing)
-   **Purpose:** To establish a high-performance "sanity check" model and demonstrate the maximum achievable accuracy on the source data.
-   **Method:** This notebook trains a custom CNN on the clean, upsampled *Free Spoken Digit Dataset*. No data augmentation is used.
-   **Final Result:** The baseline model achieves a state-of-the-art **99.33% accuracy** on the clean test set, proving the validity of the data pipeline and model architecture in a perfect environment.

### 2. The Real-Time Model
-   **Notebook:** [`2_real_time_Digit_recognition.ipynb`](https://colab.research.google.com/drive/17EJQJf-BP_g2p5oAxdBhmtc9OVtlOXYp?usp=sharing)
-   **Purpose:** To build the final, robust model and deploy it in a live application with microphone integration.
-   **Method:** This notebook retrains the custom CNN from scratch but enables a gentle and effective data augmentation strategy (`SpecAugment`). This prepares the model for the "air gap" problem—the domain shift between clean data and real-world microphone audio.
-   **Final Result:** The real-time model achieves an excellent **98.00% accuracy** on the clean test set, demonstrating that it retains expert knowledge while gaining the robustness needed for live inference. This notebook contains the final application.

---

## The Core Challenge: The "Air Gap" Problem

While achieving high accuracy on a clean dataset is a good start, the primary challenge of this project was bridging the domain gap between the pristine training data and the noisy, distorted audio from a live microphone. Our iterative process revealed several key insights:

-   **Sample Rate Mismatch:** The 8kHz training data had to be upsampled to 48kHz to match the domain of modern browser-based microphones.
-   **The Failure of Over-Augmentation:** Aggressive audio-level augmentations (like heavy reverb and noise) completely destroyed the signal in the clean FSDD dataset, preventing the model from learning.
-   **The Success of Gentle Augmentation:** The final, successful real-time model uses a targeted and gentle augmentation strategy—`SpecAugment`—which regularizes the model effectively without corrupting the core audio features.

---

## Audio Features and Model Choice

The success of this project hinges on two key technical decisions: the choice of audio features and the model architecture.

### Audio Feature: Log-Mel Spectrogram
-   **What it is:** A log-mel spectrogram is a 2D "image" of sound. It represents how the energy across different frequencies changes over time. Crucially, the frequency axis is warped onto the *Mel scale*, which mimics human auditory perception by giving more importance to lower frequencies. The energy is also converted to a logarithmic (decibel) scale.
-   **Why it's relevant:** For a Convolutional Neural Network (CNN), this is the perfect feature. A CNN is an expert image processor. The log-mel spectrogram provides a rich, image-like representation where the characteristic patterns of spoken digits (like formants) appear as textures and shapes. This allows the CNN to leverage its powerful pattern-recognition capabilities to "see" the difference between digits.

### Model Type: Custom Lightweight CNN
-   **What it is:** Our final model is a custom-built Convolutional Neural Network with three convolutional blocks followed by a classifier head. This architecture is lightweight and tailored specifically for this 10-class classification problem.
-   **Why it's relevant:** A custom CNN provides the perfect balance:
    -   ***Efficient:*** It has a small footprint and trains quickly.
    -   ***Effective:*** It is powerful enough to learn the intricate patterns in the spectrograms, achieving near-perfect accuracy.
    -   ***Appropriate:*** It correctly treats the problem as an image classification task, which is the most effective approach for this type of isolated-word recognition.

---

## How to Use This Repository

1.  **Open `1_baseline_Digit_recognition.ipynb`**
    -   Run the single cell to train the high-accuracy baseline model. This notebook serves as a "sanity check" and can be used for performance comparisons.

2.  **Open `2_real_time_Digit_recognition.ipynb`**
    -   Run the first large cell ("The Definitive Training Pipeline") to train the final, robust model. This will save its weights to `best_model.pth`.
    -   After training is complete, run the last cell ("The Definitive Real-Time Inference App") to launch the application with microphone integration and test the model with your own voice.

---

## Technologies Used

-   **Framework:** PyTorch
-   **Core Libraries:**
    -   `torchaudio` for audio processing and transformations.
    -   `scikit-learn` for performance evaluation metrics.
    -   `numpy` and `pandas` for data manipulation.
    -   `matplotlib` and `seaborn` for plotting.
-   **Tools:**
    -   `noisereduce` for real-time noise suppression.
-   **Environment:** Google Colab
