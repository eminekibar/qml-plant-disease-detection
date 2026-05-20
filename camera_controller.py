"""
Picamera2 Kamera Sarmalayıcısı
==============================

Bu modülün amacı Picamera2 kurulum ayrıntılarını main döngüsünden ayırmaktır.

Önemli tercih:
    main stream formatı olarak "RGB888" seçildi.

Sebep:
    Picamera2 kılavuzuna göre Python tarafında bu format OpenCV'nin beklediği
    (B, G, R) piksel düzenine karşılık gelir. Böylece ek format karmaşası
    yaşamadan inference pipeline'ına geçebiliriz.
"""

from __future__ import annotations

import time

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class CameraController:
    def __init__(
        self,
        frame_size: tuple[int, int] = (640, 480),
        warmup_seconds: float = 2.0,
        buffer_count: int = 4,
    ):
        self.frame_size = frame_size
        self.warmup_seconds = warmup_seconds
        self.buffer_count = buffer_count
        self.picam2 = None

    def start(self) -> None:
        if Picamera2 is None:
            raise RuntimeError(
                "Picamera2 import edilemedi. Pi üzerinde python3-picamera2 kurulmalı."
            )

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": self.frame_size, "format": "RGB888"},
            buffer_count=self.buffer_count,
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Kamera sensörünün pozlama ve beyaz ayar için kısa bir ısınma süresine
        # ihtiyacı olur. İlk birkaç frame'i hemen kullanmak kararsız sonuç verebilir.
        time.sleep(self.warmup_seconds)

    def capture_frame(self):
        if self.picam2 is None:
            raise RuntimeError("Kamera başlatılmadan capture_frame çağrıldı.")
        return self.picam2.capture_array("main")

    def stop(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
