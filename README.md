# QML PLANT DISEASE DETECTION

Bu proje; tarımsal sürdürülebilirlik ve akıllı tarım uygulamaları kapsamında, bitki yapraklarındaki patolojik durumları uç cihazlar üzerinde gerçek zamanlı olarak tespit etmek amacıyla geliştirilmiş, **Kuantum-Klasik Hibrit Yapay Zekâ (Hybrid Quantum-Classical Machine Learning)** tabanlı bir uç cihaz çıkarım (Edge Inference) sistemidir. 

Sistem, klasik evreşimli sinir ağları (CNN) ile kuantum hesaplama katmanlarını entegre ederek uç cihaz segmentinde yüksek doğruluklu ve optimize edilmiş kararlar üretmektedir.

---

## Mimarî ve Pipeline Akışı

Sistemin uçtan uca veri işleme ve çıkarım hattı şu aşamalardan oluşmaktadır:

1. **Görüntü Yakalama Modülü:** `picamera2` sarmalayıcısı aracılığıyla, Raspberry Pi kamera sensöründen gelen ham görüntüler OpenCV ile tam uyumlu ve ek dönüşüm katmanı gerektirmeyecek şekilde `RGB888` formatında yakalanır.
2. **Klasik Özellik Çıkarımı (Feature Extraction):** Yakalanan kareler, gömülü sistem optimizasyonu için TensorFlow Lite (TFLite) platformuna optimize edilmiş MobileNetV2 omurgasından geçirilerek 128 boyutlu klasik bir özellik vektörüne dönüştürülür.
3. **Kuantum Durum Kodlama (Quantum State Embedding):** Geleneksel Angle Embedding yöntemlerinin aksine, 128 boyutlu klasik özellik uzayını kayıpsız bir şekilde kuantum sistemine aktarmak amacıyla **7-Qubit Amplitude Embedding** mimarisi kullanılmıştır. Bu sayede $2^7 = 128$ amplitüd değeri kuantum durum vektörüne doğrudan haritalanır.
4. **Kuantum Devresi Varyasyonel Katmanları:** Kodlanan kuantum durumları, PennyLane kütüphanesi üzerinde kurgulanmış 4 derinlikli `StronglyEntanglingLayers` mimarisinden geçirilerek parametrik olarak işlenir.
5. **Kuantum Ölçüm ve Sınıflandırma (Softmax Classification):** Her qubit üzerinden alınan PauliX, PauliY ve PauliZ beklenti değerleri (Expectation Values) ile 21 boyutlu bir kuantum çıktı matrisi ($N_{qubits} \times 3$) üretilir. Bu çıktılar, klasik bir Numpy Softmax katmanı ile işlenerek nihai hastalık sınıfı ve güven (confidence) skoru hesaplanır.

---

## Depo (Repository) Yapısı

```text
├── models/
│   ├── feature_extractor.tflite  # Klasik CNN Öznitelik Çıkarıcı (128D)
│   ├── quantum_weights.npz       # Eğitilmiş Kuantum Katman Ağırlıkları
│   └── class_names.txt           # 38 Farklı Hastalık Sınıf Etiketleri
├── camera_controller.py          # Picamera2 Görüntü Pipeline Yapılandırması
├── inference.py                  # TFLite Backbone + PennyLane QNode Entegrasyonu
├── gpio_controller.py            # Donanımsal LED ve Buzzer Uyanış Mantığı
├── logger.py                     # SQLite Kayıt Yönetimi & Depolama Bütçesi
├── dashboard.py                  # Flask Web Tabanlı Canlı Kontrol Paneli
├── main.py                       # Raspberry Pi Ana Çalıştırma ve Çıkarım Döngüsü
├── test_pc.py                    # Donanımsız Bilgisayarlar İçin Çıkarım Test Scripti
└── plant_disease_train.ipynb     # Colab Ortamında Kuantum-Klasik Eğitim Hattı

```

---

## Donanım Kurulumu ve GPIO Haritası

Raspberry Pi üzerinde fiziksel geri bildirimlerin kararlı çalışması ve yanlış alarmların önlenmesi için Broadcom (BCM) pin şeması üzerinden debounced bir sinyal hattı kurgulanmıştır.

| Bileşen / Çıktı Türü | GPIO Pin (BCM) | Karar/Tetikleme Mantığı |
| --- | --- | --- |
| **Kırmızı LED** | `GPIO 17` | Yaprakta patoloji/hastalık tespiti durumunda aktif olur. |
| **Yeşil LED** | `GPIO 27` | Bitkinin sağlıklı (healthy) olarak sınıflandırılması durumunda aktif olur. |
| **Sarı LED** | `GPIO 22` | Çıkarım güven skoru belirlenen eşik değerin altında kaldığında belirsizliği gösterir. |
| **Aktif Buzzer** | `GPIO 23` | Kritik hastalık durumlarında 3 kez kesikli sesli uyarı sinyali üretir. |
| **Fiziksel Buton** | `GPIO 24` | İsteğe bağlı tetiklemeli çıkarım (Button Mode) için dahili Pull-Up direnciyle çalışır. |

---

## Kurulum ve Bağımlılıklar

Raspberry Pi OS mimarisinde kararlı bir çalışma ortamı sağlamak için sistem kütüphaneleri ve Python paketleri uyumlu şekilde kurulmalıdır.

### 1. Gerekli Sistem Paketlerinin Kurulması

```bash
sudo apt update
sudo apt install python3-numpy python3-opencv python3-picamera2 -y

```

### 2. Python Paketlerinin Kurulması (PEP 668 Uyumlu)

```bash
pip3 install tflite-runtime pennylane Flask --break-system-packages

```

---

## Kullanım Senaryoları

### 1. Canlı Çıkarım Döngüsünü Başlatmak (Raspberry Pi)

Sistem varsayılan olarak kamera akışını sürekli polleyerek çıkarım yapar:

```bash
python3 main.py --min-confidence 0.65 --poll-interval 0.75

```

### 2. Tetiklemeli Tahmin Modu (Button Mode)

Kameranın sürekli çalışmasını engellemek ve sadece fiziksel butona basıldığında tahmin üretmek için:

```bash
python3 main.py --button --button-pin 24

```

### 3. PC Üzerinde Donanımsız Test Çalıştırması

Uç donanım (Pi) ve kamera sensörü olmadan klasik bir bilgisayarda pipeline doğrulaması yapmak için:

```bash
python3 test_pc.py test_yaprak_gorseli.jpg --top 5

```

---

## Veritabanı ve İzleme Paneli (Dashboard)

Sistem tarafından üretilen tüm kararlar ve patolojik analiz sonuçları, lokal veri bütünlüğünü korumak adına SQLite formatında `plant_log.db` veritabanına kaydedilir.

* **Depolama Bütçesi (Housekeeping):** Sınırlı disk alanına sahip uç cihazlar için, veritabanı logları otomatik olarak 7 gün ile sınırlandırılmış ve görsel klasörü boyutu maksimum 1024 MB olarak bütçelenmiştir. Bu sınır aşıldığında sistem en eski kayıtlardan başlayarak otomatik temizlik tetikler.
* **Web Arayüzü:** Ağ üzerindeki diğer cihazlardan sistemi takip etmek amacıyla, Flask tabanlı hafif bir dashboard arayüzü sunulmaktadır. Canlı tespitler `/` rotası üzerinden, ham JSON çıktıları ise `/api/events` uç noktası üzerinden izlenebilir.

```bash
python3 dashboard.py

```

---

## Eğitim Süreci (Phase 1 & Phase 2)

Model, hibrit mimarinin kararlılığı için Google Colab ortamında iki aşamalı bir pipeline ile eğitilmiştir:

* **Phase 1 (Klasik Sınıflandırma):** MobileNetV2 omurgasının son 30 katmanı serbest bırakılarak veri kümesi üzerinde ince ayar (fine-tune) yapılmış ve 128 boyutlu öznitelik çıkarıcı `feature_extractor.tflite` modeli üretilmiştir.
* **Phase 2 (Kuantum Optimizasyonu):** Çıkarılan öznitelikler L2 normalizasyonundan geçirilerek cache'lenmiş, ardından Adam optimizasyon yöntemi ve kosinüs öğrenme oranı (Cosine LR) stratejisi kullanılarak PennyLane varyasyonel devre ağırlıkları (`quantum_weights.npz`) optimize edilmiştir.

```
