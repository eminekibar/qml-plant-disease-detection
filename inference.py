"""
Pi Inference Modülü — TFLite Backbone + PennyLane Quantum Devre
===============================================================

Inference akışı:
    1. Kamera frame'i → TFLite feature extractor → 128D özellik vektörü
    2. L2 normalizasyon → birim norm vektör (2^7 = 128 amplitüd)
    3. PennyLane 7-qubit AmplitudeEmbedding + StronglyEntanglingLayers
       → 21D quantum çıktı (PauliX + PauliY + PauliZ, her qubit)
    4. Numpy softmax sınıflandırıcı → hastalık sınıfı

Neden AmplitudeEmbedding?
    AngleEmbedding 4 qubit'e 4 sayı kodlarken AmplitudeEmbedding 7 qubit'e
    2^7=128 sayı kodlar. Kayıpsız özellik aktarımı sağlar, ayrı projeksiyon
    katmanına gerek kalmaz.

Kurulum (Pi'de):
    sudo apt install python3-numpy
    pip3 install --break-system-packages tflite-runtime pennylane RPi.GPIO
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

try:
    from ai_edge_litert import interpreter as tflite   # yeni Pi OS (Python 3.13+)
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite   # eski Pi OS
    except ImportError:
        import tensorflow.lite as tflite               # PC testi fallback

import pennylane as qml


N_OBSERVABLES = 3   # PauliX + PauliY + PauliZ


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    is_diseased: bool


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Eğitimle aynı normalizasyon: birim L2 normu."""
    return v / (np.linalg.norm(v) + 1e-8)


class PlantDiseaseClassifier:
    """
    Pi üzerinde çalışan kuantum-hibrit sınıflandırıcı.

    Bileşenler:
        - TFLite feature extractor  : görüntü → 128D özellik
        - L2 normalizasyon          : 128D → birim norm (2^7 amplitüd)
        - PennyLane quantum devre   : AmplitudeEmbedding + StronglyEntanglingLayers
                                      → 21D quantum çıktı
        - Numpy softmax             : 21D → sınıf olasılıkları
    """

    def __init__(
        self,
        feature_extractor_path: str,
        quantum_weights_path: str,
        class_names_path: str,
    ):
        for path in (feature_extractor_path, quantum_weights_path, class_names_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dosya bulunamadı: {path}")

        # --- TFLite feature extractor ---
        self.interpreter = tflite.Interpreter(model_path=feature_extractor_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape       = self.input_details[0]["shape"]
        self.input_height = int(input_shape[1])
        self.input_width  = int(input_shape[2])
        self.input_dtype  = self.input_details[0]["dtype"]

        # --- Quantum weights ---
        data = np.load(quantum_weights_path)
        self.q_weights         = data["quantum_weights"].astype(np.float64)
        self.classifier_kernel = data["classifier_kernel"].astype(np.float64)
        self.classifier_bias   = data["classifier_bias"].astype(np.float64)
        self._n_layers         = int(data["n_layers"])
        self._n_qubits         = int(data["n_qubits"])
        self._feature_dim      = int(data["feature_dim"])
        n_classes              = self.classifier_bias.shape[0]

        expected_amp = 2 ** self._n_qubits
        assert self._feature_dim == expected_amp, (
            f"feature_dim ({self._feature_dim}) != 2^n_qubits ({expected_amp}). "
            "Model ile inference.py uyumsuz."
        )

        # --- PennyLane quantum devre ---
        self._dev = qml.device("default.qubit", wires=self._n_qubits)
        n_qubits  = self._n_qubits

        @qml.qnode(self._dev)
        def _circuit(features, weights):
            # normalize=False: L2 norm _l2_normalize() ile dışarıda uygulanır
            qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=False)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return (
                [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
                [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
                [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            )

        self._circuit = _circuit

        # --- Sınıf isimleri ---
        with open(class_names_path, encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        assert len(self.class_names) == n_classes, (
            f"class_names.txt ({len(self.class_names)}) ile "
            f"classifier ağırlığı ({n_classes}) uyuşmuyor."
        )

        output_dim = self._n_qubits * N_OBSERVABLES
        print(
            f"[INFO] classifier yüklendi "
            f"(input={self.input_width}x{self.input_height}, "
            f"feature_dim={self._feature_dim}, "
            f"n_qubits={self._n_qubits}, n_layers={self._n_layers}, "
            f"output_dim={output_dim}, n_classes={n_classes})"
        )

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr) -> np.ndarray:
        """BGR OpenCV frame → TFLite input tensor"""
        import cv2
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))

        if self.input_dtype == np.float32:
            return np.expand_dims(resized.astype(np.float32), axis=0)
        # INT8 quantized model
        scale, zero_point = self.input_details[0]["quantization"]
        tensor = np.round(resized / scale + zero_point).astype(np.int8)
        return np.expand_dims(tensor, axis=0)

    def _extract_features(self, frame_bgr) -> np.ndarray:
        """TFLite backbone → (feature_dim,) float64 özellik vektörü"""
        tensor = self._preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        if self.output_details[0]["dtype"] != np.float32:
            scale, zero_point = self.output_details[0]["quantization"]
            raw = (raw.astype(np.float32) - zero_point) * scale

        return raw.astype(np.float64)

    def _quantum_inference(self, features_norm: np.ndarray) -> np.ndarray:
        """L2-normalize edilmiş özellik → PennyLane quantum çıktı (output_dim,)"""
        result = self._circuit(features_norm, self.q_weights)
        return np.array(result, dtype=np.float64)

    def _classify(self, quantum_output: np.ndarray) -> tuple[int, float]:
        """Softmax sınıflandırıcı → (sınıf_indeksi, güven)"""
        logits = quantum_output @ self.classifier_kernel + self.classifier_bias
        probs  = _softmax(logits)
        idx    = int(np.argmax(probs))
        return idx, float(probs[idx])

    # ------------------------------------------------------------------
    # Ana predict metodu
    # ------------------------------------------------------------------

    def predict(self, frame_bgr) -> Prediction:
        """
        Tam inference pipeline:
            frame_bgr → TFLite(128D) → L2 norm → PennyLane(21D) → softmax → Prediction
        """
        features_raw  = self._extract_features(frame_bgr)
        features_norm = _l2_normalize(features_raw)
        quantum_out   = self._quantum_inference(features_norm)
        idx, conf     = self._classify(quantum_out)
        class_name    = self.class_names[idx]

        return Prediction(
            class_name=class_name,
            confidence=conf,
            is_diseased=self.is_diseased(class_name),
        )

    @staticmethod
    def is_diseased(class_name: str) -> bool:
        return "healthy" not in class_name.lower()
