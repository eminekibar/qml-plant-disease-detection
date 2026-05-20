"""
GPIO Kontrolcüsü - Buzzer + LED
===============================

Pin atamaları (BCM):
    GPIO 17 -> Kırmızı LED   : hastalıklı / kritik
    GPIO 27 -> Yeşil LED     : sağlıklı
    GPIO 22 -> Sarı LED      : düşük güven / belirsiz
    GPIO 23 -> Aktif buzzer  : sesli uyarı

Burada amaç sadece pin aç/kapa yapmak değil; düşük güven ile yüksek güven
sonuçlarını ayrıştırıp demoda yanlış alarm hissini azaltmak.
"""

from __future__ import annotations

import time

try:
    import RPi.GPIO as GPIO
    _HAVE_REAL_GPIO = True
except (ImportError, RuntimeError):
    GPIO = None
    _HAVE_REAL_GPIO = False


class AlertController:
    def __init__(
        self,
        red_pin: int = 17,
        green_pin: int = 27,
        yellow_pin: int = 22,
        buzzer_pin: int = 23,
        confidence_threshold: float = 0.20,
        enabled: bool = True,
    ):
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.yellow_pin = yellow_pin
        self.buzzer_pin = buzzer_pin
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled and _HAVE_REAL_GPIO

        if not self.enabled:
            print("[GPIO] simülasyon modunda çalışılıyor")

    def setup(self) -> None:
        if not self.enabled:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self._pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    @property
    def _pins(self) -> tuple[int, int, int, int]:
        return (self.red_pin, self.green_pin, self.yellow_pin, self.buzzer_pin)

    def _set(self, pin: int, state: bool) -> None:
        if self.enabled:
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        else:
            print(f"[GPIO] pin={pin} state={'HIGH' if state else 'LOW'}")

    def all_off(self) -> None:
        for pin in self._pins:
            self._set(pin, False)

    def beep(self, count: int, on_ms: int, off_ms: int) -> None:
        for _ in range(count):
            self._set(self.buzzer_pin, True)
            time.sleep(on_ms / 1000)
            self._set(self.buzzer_pin, False)
            if off_ms > 0:
                time.sleep(off_ms / 1000)

    def alert_prediction(self, is_diseased: bool, confidence: float) -> None:
        """
        Karar mantığı:
            - Güven düşükse sarı LED ile belirsizlik göster
            - Hastalıklıysa kırmızı LED + buzzer
            - Sağlıklıysa yeşil LED
        """
        self.all_off()

        low_confidence = confidence < self.confidence_threshold

        if is_diseased:
            self._set(self.red_pin, True)
            self.beep(count=3, on_ms=200, off_ms=100)
        else:
            self._set(self.green_pin, True)

        if low_confidence:
            self._set(self.yellow_pin, True)

        time.sleep(2.0)
        self.all_off()

    def cleanup(self) -> None:
        self.all_off()
        if self.enabled:
            GPIO.cleanup()
