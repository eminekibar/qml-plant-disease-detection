"""
PC Test Scripti
===============
Raspberry Pi kamerası olmadan inference pipeline'ını test eder.
Kullanım:
    python test_pc.py gorsel.jpg
    python test_pc.py gorsel.jpg --top 5
"""

import argparse
import os
import sys

import cv2
import numpy as np

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Test edilecek görsel dosyası")
    p.add_argument("--model",   default=os.path.join(MODEL_DIR, "feature_extractor.tflite"))
    p.add_argument("--weights", default=os.path.join(MODEL_DIR, "quantum_weights.npz"))
    p.add_argument("--labels",  default=os.path.join(MODEL_DIR, "class_names.txt"))
    p.add_argument("--top",     type=int, default=3, help="Kaç sınıf gösterilsin")
    return p.parse_args()


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"HATA: Görsel bulunamadı: {args.image}")
        sys.exit(1)

    # Model yükle
    from inference import PlantDiseaseClassifier
    clf = PlantDiseaseClassifier(args.model, args.weights, args.labels)

    # Görseli yükle
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"HATA: Görsel okunamadı: {args.image}")
        sys.exit(1)

    print(f"\nGörsel : {args.image}  ({frame.shape[1]}x{frame.shape[0]})")
    print("-" * 50)

    # Tam pipeline'ı çalıştır (inference.py içindeki predict)
    pred = clf.predict(frame)

    print(f"Sonuç  : {pred.class_name}")
    print(f"Güven  : {pred.confidence * 100:.1f}%")
    print(f"Durum  : {'🔴 Hastalıklı' if pred.is_diseased else '🟢 Sağlıklı'}")

    # Top-N için internal pipeline tekrar çalıştır
    if args.top > 1:
        import pennylane as qml
        features_raw  = clf._extract_features(frame)
        features_norm = features_raw / (np.linalg.norm(features_raw) + 1e-8)
        q_out         = clf._quantum_inference(features_norm)
        logits        = q_out @ clf.classifier_kernel + clf.classifier_bias
        probs         = softmax(logits)
        top_idx       = np.argsort(probs)[::-1][:args.top]

        print(f"\nTop-{args.top}:")
        for rank, idx in enumerate(top_idx, 1):
            bar = "█" * int(probs[idx] * 30)
            print(f"  {rank}. {clf.class_names[idx]:<45s} {probs[idx]*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
