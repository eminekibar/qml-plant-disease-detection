"""
Raspberry Pi Ana Çalıştırma Döngüsü
===================================

Akış:
    Picamera2 -> TFLite inference -> GPIO uyarısı -> SQLite log

Tasarım hedefleri:
    - Pi 3 üzerinde gereksiz yük bindirmemek
    - Her karede buzzer çalmamak
    - Log klasörünü kontrolsüz büyütmemek
    - Demo sırasında okunabilir konsol çıktısı vermek
"""

from __future__ import annotations

import argparse
import os
import time
import signal
import sys

from camera_controller import CameraController
from gpio_controller import AlertController
from inference import PlantDiseaseClassifier
import logger


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.tflite")
DEFAULT_QUANTUM_WEIGHTS_PATH   = os.path.join(MODEL_DIR, "quantum_weights.npz")
DEFAULT_CLASS_NAMES_PATH       = os.path.join(MODEL_DIR, "class_names.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default=DEFAULT_FEATURE_EXTRACTOR_PATH)
    parser.add_argument("--quantum-weights", default=DEFAULT_QUANTUM_WEIGHTS_PATH)
    parser.add_argument("--labels",         default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--poll-interval", type=float, default=0.75)
    parser.add_argument("--cooldown-seconds", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--housekeeping-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true", help="Tek kare tahmin yapıp çıkar")
    parser.add_argument("--save-all", action="store_true", help="Tüm loglara görsel ekle")
    parser.add_argument("--display", action="store_true", help="OpenCV penceresi açar")
    parser.add_argument("--dry-run-gpio", action="store_true", help="Gerçek GPIO yerine print kullanır")
    parser.add_argument("--button", action="store_true", help="Butona basınca tahmin yap (GPIO 24)")
    parser.add_argument("--button-pin", type=int, default=24, help="Buton GPIO pin (BCM)")
    return parser.parse_args()


def should_log_event(
    current_class: str,
    last_logged_class: str | None,
    now: float,
    last_log_at: float,
    cooldown_seconds: float,
) -> bool:
    """
    Aynı etiketi her döngüde veritabanına basmak istemiyoruz.
    Bu nedenle ya sınıf değişmeli ya da belli bir zaman geçmiş olmalı.
    """
    if last_logged_class is None:
        return True
    if current_class != last_logged_class:
        return True
    return (now - last_log_at) >= cooldown_seconds


def maybe_show_frame(frame, title: str) -> bool:
    import cv2

    cv2.imshow("Plant Disease Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print(f"[INFO] pencere kapatma isteği alındı: {title}")
        return False
    return True


def main() -> None:
    args = parse_args()

    def handle_exit_signal(signum, frame):
        print(f"[INFO] Sistem sinyali alındı ({signum}). Temizce kapatılıyor...")
        # KeyboardInterrupt hatası fırlatarak aşağıdaki 'except' bloğuna düşmesini sağlıyoruz
        raise KeyboardInterrupt 

    # Sistemden gelen kapatma (SIGTERM) sinyalini yakala
    signal.signal(signal.SIGTERM, handle_exit_signal)

    classifier = PlantDiseaseClassifier(args.model, args.quantum_weights, args.labels)
    alerts = AlertController(
        confidence_threshold=args.min_confidence,
        enabled=not args.dry_run_gpio,
    )
    camera = CameraController(frame_size=(args.frame_width, args.frame_height))

    last_alert_at = 0.0
    last_log_at = 0.0
    last_logged_class = None
    last_housekeeping_at = 0.0

    # Buton kurulumu
    button_gpio = None
    if args.button and not args.dry_run_gpio:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(args.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            button_gpio = args.button_pin
            print(f"[INFO] Buton aktif (GPIO {button_gpio}) — basmak için bekliyor")
        except Exception as e:
            print(f"[WARN] Buton kurulamadı: {e}")

    try:
        alerts.setup()
        camera.start()

        while True:
            # Buton modu: basılana kadar bekle
            if button_gpio is not None:
                print("[INFO] Butona bas → tahmin yap  (Ctrl+C ile çık)")
                import RPi.GPIO as GPIO
                while GPIO.input(button_gpio) == GPIO.HIGH:
                    time.sleep(0.05)
                time.sleep(0.05)  # debounce
                print("[INFO] Buton algılandı, tahmin yapılıyor...")

            frame = camera.capture_frame()
            prediction = classifier.predict(frame)
            now = time.time()

            print(
                "[INFO] "
                f"prediction={prediction.class_name} "
                f"confidence={prediction.confidence:.3f} "
                f"diseased={prediction.is_diseased}"
            )

            # Sesli/görsel uyarıyı sürekli tetiklemek yerine cooldown ile sınırlıyoruz.
            if (now - last_alert_at) >= args.cooldown_seconds:
                alerts.alert_prediction(
                    is_diseased=prediction.is_diseased,
                    confidence=prediction.confidence,
                )
                last_alert_at = now

            if should_log_event(
                current_class=prediction.class_name,
                last_logged_class=last_logged_class,
                now=now,
                last_log_at=last_log_at,
                cooldown_seconds=args.cooldown_seconds,
            ):
                save_frame = args.save_all or prediction.is_diseased
                logger.log_event(
                    class_name=prediction.class_name,
                    confidence=prediction.confidence,
                    is_diseased=prediction.is_diseased,
                    frame=frame if save_frame else None,
                )
                last_logged_class = prediction.class_name
                last_log_at = now

            if (now - last_housekeeping_at) >= args.housekeeping_seconds:
                deleted_by_age = logger.purge_old_records()
                deleted_by_size = logger.purge_storage_budget()
                print(
                    "[INFO] housekeeping "
                    f"age_deleted={deleted_by_age} size_deleted={deleted_by_size}"
                )
                last_housekeeping_at = now

            if args.display:
                import cv2

                display_frame = frame.copy()
                text = f"{prediction.class_name} ({prediction.confidence:.2f})"
                color = (0, 0, 255) if prediction.is_diseased else (0, 255, 0)
                cv2.putText(
                    display_frame,
                    text,
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                if not maybe_show_frame(display_frame, text):
                    break

            if args.once:
                break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("[INFO] kullanıcı isteği ile durduruldu")
    finally:
        camera.stop()
        alerts.cleanup()
        logger.close()

        if args.display:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
