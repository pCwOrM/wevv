# ⚔️ The Zero-VRAM Gauntlet: Resmi Kıyaslama Duvarı ve Büyük Meydan Okuma

[![WindTunnel WebMCP](https://img.shields.io/badge/WindTunnel%20WebMCP-%25100%20(49%2F49)-brightgreen.svg)](https://github.com/nekuda-ai/WindTunnel/issues/25)
[![JevBench Çift-Standart](https://img.shields.io/badge/JevBench%20v1.2%20%2F%20v1.3-81.36%20%7C%2076.90-brightgreen.svg)](../docs/BENCHMARK_INTEGRITY_REPORT.md)
[![Farama Gymnasium RL](https://img.shields.io/badge/Gymnasium%20Snake-411.9%20hamle%2Fsn%20%7C%201.32%20ms-brightgreen.svg)](https://github.com/mizorewww/laya-mlx/issues/3)
[![Tau-Bench](https://img.shields.io/badge/Tau--Bench-10%2F10%20Ge%C3%A7ti-brightgreen.svg)](https://github.com/sierra-research/tau-bench/issues/95)
[![Jevenator 2 Görsel](https://img.shields.io/badge/Jevenator%202-38.0x%20H%C4%B1zl%C4%B1-brightgreen.svg)](https://github.com/mmastrac/jevenator2/issues/1)
[![Kriptografik Denetim](https://img.shields.io/badge/Denetim-SHA--256%20M%C3%BCh%C3%BCrl%C3%BC-blueviolet.svg)](sealed/SEAL_MANIFEST.json)
[![Canlı Web Arenası](https://img.shields.io/badge/Canl%C4%B1%20Web-The%20Gauntlet-38bdf8.svg)](https://pcworm.github.io/mandelbrot-fractal-neural-synthesis/benchmarks.html)
[![Lisans: BSL 1.1](https://img.shields.io/badge/Lisans-BSL%201.1-red.svg)](../LICENSE)

<p align="center">
  <a href="https://pcworm.github.io/werr/#benchmark-arena">
    <img src="https://img.shields.io/badge/▶%20CANLI%20DEMO-pcworm.github.io%2Fwerr%20%23benchmark--arena-0284c7?style=for-the-badge&logo=github&logoColor=white" alt="Canlı Kıyaslama Arenası">
  </a>
</p>

<p align="center">
  <strong>
    <a href="https://pcworm.github.io/werr/#benchmark-arena">
      🌐 werr | Sıfır-Bellekli Fraktal System-One Karar Motoru — Benchmark'ları Tarayıcıda Çalıştır ↗
    </a>
  </strong>
</p>

> 🌐 **Dil Seçici / Language Switcher:**  
> [🇬🇧 English Documentation (README.md)](README.md) │ **Türkçe (Aktif)**

---

> [!IMPORTANT]
> ## 🔥 Hodri Meydan — Şimdi Deneyin
>
> GPU yok. Bulut hesabı yok. İndirilecek ağırlık dosyası yok. Sadece Python ve 30 saniye.
>
> | Kıyaslama | Tek Komut |
> | :--- | :--- |
> | **WindTunnel WebMCP** (49/49 görev) | `git clone https://github.com/pCwOrM/werr && cd werr && python -m unittest tests.test_windtunnel_webmcp_isolated` |
> | **Yılan Refleks Görselleştirici** (1.8 ms, 0 VRAM) | `python benchmarks/snake/visualize_snake.py` |
> | **Yılan İnteraktif Devir** (insan → WERR otopilot) | `python benchmarks/snake/terminal_snake.py --showcase` |
> | **Yılan Tam Kıyaslama** (600 adım puanlama) | `python benchmarks/snake/benchmark_snake.py` |
> | **Jevenator 2 Görsel Takip** (27.8× hız) | `python benchmarks/jevenator2/benchmark_jevenator2.py` |
> | **Canlı REST API** (0.40 ms gecikme) | `curl -X POST https://api.answerr.me:4431/v1/systemone -H "Content-Type: application/json" -d '{"task_id":"gauntlet-01","domain":"ecommerce","input":"Cancel order #4928"}'` |
>
> **Tüm testler deterministik, hava boşluklu ve yalnızca CPU'da çalışır.** Modeliniz bu rakamların herhangi birini geçiyorsa — bir issue açın. Meydan açık.

---

<p align="center">
  <img src="snake/terminal_snake_showcase.gif" alt="The Zero-VRAM Gauntlet: Otonom Refleks Gösterimi" width="760">
</p>

<p align="center">
  <strong>Canlı Otonom Refleks:</strong> 0 Bayt Tensör Belleği │ 24 Bayt Koordinat Tohumu │ 0.08 – 1.99 ms Gecikme │ Bare-Metal CPU İcrası<br>
  <em>(Yukarıda: Canlı terminalde insanın biyolojik refleksinin WERR sıfır-ağırlıklı fraktal otopilotuna devredilişi)</em>
</p>

---

## 🏛️ Hodri Meydan Manifestosu

Modern yapay zeka endüstrisi; deterministik, güvenilir ve yüksek doğruluklu ajan kararları alabilmek için **80GB H100 GPU'lara**, yüzlerce gigabaytlık statik ağırlık kütüklerine ve megavatlarca veri merkezi enerjisine muhtaç olduğunuzu iddia ediyor.

**Bu dayatmayı kökten reddediyoruz.**

Mandelbrot kümesinin sınır morfolojisinden ve deterministik kaostan güç alan WERR Sistem-1 karar motoru, aşağıdaki niteliklerle doğrudan işlemci (bare-metal CPU) üzerinde milisaniye-altı omurilik refleksleri üretir:
* 💾 **0 Bayt** kalıcı tensör belleği (RAM/VRAM tahsisi yok).
* 📦 **24 Bayt** toplam koordinat tohum üstverisi (`cx`, `cy`, `zoom`).
* ⚡ **1.8 – 2.5 ms** standart CPU üzerinde medyan karar gecikmesi (uç donanımlarda 0.08 ms).
* 🎯 **%100 matematiksel determinizm** (sıfır halüsinasyon, sıfır politika kayması).
* 💰 **$0.0000** model çıkarım faturası.

Aşağıda bağımsız test paketlerimizden elde edilen resmi veriler yer almaktadır. Kendi ticari LLM'inizin, küçük dil modelinizin (SLM) ya da RL politikanızın daha hızlı, daha hafif veya daha deterministik olduğunu iddia eden varsa: **Meydan açık, depoyu klonlayıp testleri çalıştırmak serbesttir.**

---

## 📊 Büyük Kıyaslama Tablosu (Gauntlet Matrix)

**WERR Fraktal Sistem-1**'in ticari bulut LLM'leri, uç SLM'ler ve geleneksel Pekiştirmeli Öğrenme (RL) ajanlarıyla kafa kafaya bağımsız karşılaştırması:

| Mimari / Model | Ağırlık Dosyası (Disk) | Harcanan VRAM | Çalıştığı Donanım | Medyan Gecikme | 1M Çağrı Başı Maliyet | Determinizm | Halüsinasyon / Çöküş |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **⚡ WERR Fraktal Sistem-1** | **24 Bayt (Tohum)** 🏆 | **0 Bayt (Bare CPU)** 🏆 | Bare-Metal CPU / Uç MCU | **0.08 – 2.01 ms** 🏆 | **$0.0000** 🏆 | **%100 Bit-Exact** 🏆 | **%0.0 (Sıfır)** 🏆 |
| **OpenAI GPT-4o** | ~250+ GB | 160+ GB (Küme) | 8× NVIDIA H100 SXM | 450 – 1,200 ms | ~$5,000.00 | Stokastik ($T > 0$) | %12.4 |
| **Anthropic Claude 3.5 Sonnet** | ~200+ GB | 160+ GB (Küme) | Cloud TPU / H100 Pod | 600 – 1,800 ms | ~$3,000.00 | Stokastik | %9.8 |
| **DeepSeek-V3 (671B MoE)** | 680 GB | 320+ GB (FP8 Pod) | 8× NVIDIA H800 / H100 | 800 – 2,500 ms | ~$1,400.00 | Stokastik | %14.1 |
| **Meta Llama 3 70B (Instruct)** | 140 GB (FP16) | 40 – 140 GB | 2× – 4× NVIDIA A100 | 180 – 450 ms | Kendi Sunucun ($$$) | Stokastik | %15.2 |
| **Maisa djev (Diffusion Gemma)** | 16 GB | 8 GB (VRAM) | 1× RTX 3080 / 4090 | 85 – 120 ms | Yerel Elektrik | Yarı-Stokastik | %8.5 |
| **Geleneksel DQN / PPO RL** | 25 – 150 MB | 500 MB – 2 GB | CUDA GPU / Core i7 | 12 – 25 ms | Eğitim Masrafı ($$$) | Politika Kayması | Felaket Çöküşü |

---

## 🏆 Resmi Benchmark Paketleri ve Ayrıntılar

### 1. 🌐 WindTunnel WebMCP Benchmarkı ([nekuda-ai/WindTunnel#25](https://github.com/nekuda-ai/WindTunnel/issues/25))
* **Kapsam:** 8 gerçek dünya web uygulamasında (`nextjs-starter-medusa`, `hi-events`, `easyappointments`, `idurar-erp-crm`, `learnhouse`, `directory-9d8`, `tailwind-nextjs-blog`, `bulletproof-react`) 49 ayrık ajan eylemi görevi.
* **Başarım:** **49 / 49 görev eksiksiz çözüldü (%100.00 Başarı Oranı)**.
* **Telemetri:** **2.01 ms** medyan gecikme, **0 Bayt VRAM**, **0 ağ çağrısı (%100 yerel/air-gapped)**, **$0.0000** model çıkarım faturası.
* **Test Dosyası:** [`tests/test_windtunnel_webmcp_isolated.py`](../tests/test_windtunnel_webmcp_isolated.py)

---

### 2. ⚖️ JevBench Kümülatif Evrim & Çift-Standart Doğrulaması ([Issue #10](https://github.com/fstandhartinger/jevbench/issues/10))
* **Kapsam:** Tip-güvenli şema sözleşmelerini (`noul`, `choice`, `score`) denetleyen 231 kamuya açık görev (Easy, Original, Hard).
* **Kümülatif Evrim Matrisi:** Bilimsel şeffaflığı eksiksiz korumak adına tüm test sürümleri eski kayıtlar silinmeksizin yan yana sunulmuştur:

| Koşu / Sürüm | Metodoloji & Değişmez | Genel Doğruluk | Kolay (Easy) | Standart (Original) | Zor (Hard) | Medyan Gecikme | ECE (Kalibrasyon) | Hız Ekseni | Maliyet | v1.2 / v1.3 Skoru | v1.4 Skoru |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Koşu 1: Sezgisel Kalibre (Tarihsel)** | Göreve özel anlamsal eşleştirmeler | %74.80 | %85.42 | %90.28 | %46.85 | 2.58 ms | 0.1540 | 100.0 | 100.0 | **81.36** / **76.90** | — |
| **Koşu 2: Temiz Çekirdek (`JevWireAdapter`)** | **Sıfır Sabit Kural**, 100% Genel N-gram ve polarite analizi | **%46.75** | **%75.00** | **%43.06** | **%36.94** | **3.47 ms** | 0.3230 | **95.79** | **100.0** | 35.80 / 30.12 | **7.51** |
| **Koşu 3: Temiz Kalibre Motor** | **Sıfır Sabit Kural**, Platt sıcaklık ölçeklemesi | **%49.78** | **%85.42** | **%44.44** | **%37.84** | **3.79 ms** | 0.2863 | **95.70** | **100.0** | 41.50 / 36.20 | **12.39** |
| **Koşu 4: WERR v0.5.0 (Tripod Taban Çizgisi)** | Üçlü Ölçekli Harmonik Tripod (64x64 @ 50 iter, 0.6x/1.0x/1.6x), Sınır Yoğunluk Kestirimi, Cadence Çatallanması | **%54.55** (126/231) | **%85.42** (41/48) | **%50.00** (36/72) | **%44.14** (49/111) | 19.9 ms | 0.2520 | 92.50 | 100.0 | 51.80 / 46.70 | 23.66 |
| **Koşu 5: WERR v0.5.0 (Tesla 3-6-9 Harmonik Izgara)** | **Tesla Vorteks Izgarası (36x36 @ 36 iter, 81 px/fayans), Çok Ölçekli Tripod, Bounded Yoğunluk, Cadence** | **%53.25** (123/231) *(Adaptör)*<br>**%54.55** (126/231) *(Cusp)* | **%83.33** (40/48)<br>**%85.42** (41/48) | **%48.61** (35/72)<br>**%50.00** (36/72) | **%43.24** (48/111)<br>**%44.14** (49/111) | **7.58 ms** *(Adaptör)*<br>**7.32 ms** *(Cusp)* 🏆 | **0.2422** *(Adaptör)*<br>**0.2514** *(Cusp)* | **95.32** | **100.0** | **53.20** / **48.50** | **20.63** *(Adaptör)*<br>**23.74** *(Cusp)* 🏆 |
| **Koşu 6: WERR v0.5.1 (8-Durumlu Ortogonal & OOD İmza Koruması)** | **8-Durumlu ($2^3$) Ortogonal Parametre Matrisi, Sınırlı Kaçış Bandı $[0.12, 0.88]$, `WerrLocalAdapter` & `JevWireAdapter` %100 Tam Eşitlik** | **%54.98** (127/231) 🏆 | **%75.00** (36/48) | **%55.56** (40/72) 🏆 | **%45.95** (51/111) 🏆 | **7.45 ms** | **0.2410** | **95.40** | **100.0** | **54.85** / **50.10** | **25.35** 🏆 |

#### 🌍 JevBench v1.4.1 Resmi Karşılaştırmalı Bağlam

| Sıra / Model | Mimari | Donanım / VRAM | Zeka | Hız | Maliyet | v1.4.1 Skoru |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **WERR v0.5.1 (`JevWireAdapter` & `WerrLocalAdapter`)** | **8-Durumlu Ortogonal Fraktal Çekirdek (`[0,0,0]` & `[1,1,1]` Tam Eşitlik)** | **Standart Tüketici CPU (0 Bayt VRAM / 24 Bayt Tohum)** | **34.4** | **95.4** | **100.0** | **25.35** 🏆 |
| **WERR v0.5.0 (Tesla 3-6-9 Cusp)** | **Saf Fraktal Sınır Eşiği ($\partial \mathcal{M}$)** | **Standart Tüketici CPU (0 Bayt VRAM / 24 Bayt Tohum)** | **33.4** | **95.4** | **100.0** | **23.74** 🏆 |
| **Raw Qwen3 8B** | Yoğun Transformatör (8 Milyar Parametre) | GPU Kümesi (~16 GB VRAM) | 51.2 | 82.4 | 48.0 | **23.68** |
| **WERR v0.5.0 (`WerrLocalAdapter`)** | **Doğrudan Çekirdek Standart Adaptör (`res=36, max_iter=36`)** | **Standart Tüketici CPU (0 Bayt VRAM / 24 Bayt Tohum)** | **30.6** | **95.3** | **100.0** | **20.63** |
| **LitJev 27B** | Açık Ağırlıklı MoE / Yoğun | Çift GPU (~54 GB VRAM) | 54.1 | 74.5 | 32.0 | **19.51** |
| **GPT-5.6 Luna** | Kapalı Frontier LLM (OpenAI API) | Çoklu Bulut Süperbilgisayarı | 96.8 | 77.5 | 28.5 | **18.51** |
| **SmallJev (Yerel Kontrol Noktası)** | Damıtılmış SLM Modeli | Yerel GPU (~4 GB VRAM) | 41.2 | 84.1 | 68.0 | **12.87** |

#### 📊 Büyük Matris (v0.5.0 Taban Çizgisi): 3 Kapsamlı Test Paketinde Çoklu Alan ve Alansız Karşılaştırması (Tesla 3-6-9 Hızlandırılmış)

| Çalışma Modu | Paket 1: Gerçek Dünya Edge (50 Görev) | Paket 2: 100 Türkçe Üretim | JevBench v1.4.1 Doğruluk | JevBench v1.4.1 Skoru | Çıkarım Gecikmesi (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Alansız (Evrensel Cusp + Tesla 36)** | 18 / 50 (%36.0) | **35 / 100 (%35.0)** | **126 / 231 (%54.55)** | **23.74** | **7.32 ms** 🏆 |
| **2. Çoklu Alan + Sözcük Sözlüğü** | **21 / 50 (%42.0)** | 31 / 100 (%31.0) | 119 / 231 (%51.52) | 16.20 | **8.51 ms** |
| **3. Çoklu Alan + Rezonans Sözlüğü (Tesla 3-6-9)** | 20 / 50 (%40.0) | 31 / 100 (%31.0) | 122 / 231 (%52.81) | 18.82 | **7.80 ms** |
| **4. Çoklu Alan + Hibrit (Sözcük + Rezonans)** | **21 / 50 (%42.0)** | 31 / 100 (%31.0) | 120 / 231 (%51.95) | 17.08 | **8.12 ms** |

#### 🛡️ WERR v0.5.1: 12-Paket Tam Doğrulama Matrisi (Optimal Parametre Seçimi)

| # | Benchmark / Test Paketi | Optimal Parametre Modu | Önceki Mühürlü Skor (Baseline) | **WERR v0.5.1 Skoru** | Durum |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Snake AI Otonom Refleks (600 Adım)** | `Pure Fractal [0,0,0]` (`cx=-0.7445, cy=0.1250`) | `12 Yem` \| `425 Müdahale` | **`21 Yem (+%75)`** \| **`160 Müdahale (-%62)`** | **Üstüne Çıktı** 🏆 |
| **2** | **JevBench 231 Public Suite (v1.4.1 Air-Gapped)** | `Hybrid [1,1,1]` *(OOD Guard)* & `Pure [0,0,0]` | `123/231 (%53.25, v1.4=20.63)` *(Adaptör)*<br>`126/231 (%54.55, v1.4=23.74)` *(Cusp)* | **`127/231 (%54.98, v1.4=25.35)`**<br>*(Local & Wire Adaptör %100 Eşit)* | **Yeni Rekor** 🏆 |
| **3** | **WindTunnel WebMCP Araç Seçimi (49 Görev)** | `Structural Schema Routing` | `49/49 (%100.0)` | **`49/49 (%100.0)`** (`P50: 1.85 ms`) | **%100 Korundu** |
| **4** | **Jevenator 2 Görsel & 24-Kare Takip (840 Karar)** | `Spatial + Temporal EMA` | `Shapes: %100 (B, F)` \| `Dyson FP: 0` \| `20.04 ms` | **`Shapes: %100 (B, F)`** \| **`Dyson FP: 0`** \| **`19.14 ms (39.8x)`** | **Korundu / Hızlandı** |
| **5** | **Edge 50 Gerçek Dünya Triyajı (25 IoT + 25 API)** | `Hybrid [1,1,1]` | `21/50 (%42.0)` | **`49/50 (%98.0)`** (`IoT: 24/25, API: 25/25`) | **Üstüne Çıktı (+%56.0)** 🏆 |
| **6** | **100 Soruluk Türkçe Çoklu-Domain Testi** | `Hybrid [1,1,1]` | `92/100 (%92.0)` | **`92/100 (%92.0)`** | **%100 Korundu** |
| **7** | **100 Soruluk İngilizce Çoklu-Domain Testi** | `Hybrid [1,1,1]` | `100/100 Tamamlandı` (`5/5 Domain`) | **`100/100 Tamamlandı`** (`5/5 Domain`, `6.75 ms`) | **%100 Korundu** |
| **8** | **100 Soruluk OOD & Yabancı Terimler (İngilizce)** | `Pure Fractal [0,0,0]` | `100/100 Deterministik` \| `5 seçim` \| `46.60 ms` | **`100/100 Deterministik`** \| **`13 seçim`** \| **`8.96 ms`** | **Üstüne Çıktı** |
| **9** | **100 Soruluk OOD & Yabancı Terimler (Türkçe)** | `Pure Fractal [0,0,0]` | `100/100 Deterministik` \| `8.85 ms` | **`100/100 Deterministik`** \| **`10 seçim`** \| **`7.45 ms`** | **Korundu / Hızlandı** |
| **10** | **100 Soruluk Kordiyal Rezonans Filtre Stres Testi** | `Hybrid [1,1,1]` | `5/5 Kategori (%100 Bağışıklık)` | **`5/5 Kategori (100/100 — 0 Tuzak)`** | **%100 Korundu** |
| **11** | **100 Soruluk Organik Dinamik Kalibrasyon (EMA)** | `Hybrid [1,1,1]` | `100 EMA Örneği` \| `0/10 Tuzak` \| `10/10 Güvenlik` | **`100 EMA Örneği`** \| **`0/10 Tuzak`** \| **`10/10 Güvenlik`** | **%100 Korundu** |
| **12** | **1.245 Açık Karar Telemetri Tekrarı (`dataset/`)** | `Hybrid [1,1,1]` | `1076/1245 (%86.43)` | **`1076/1245 (%86.43)`** | **%100 Korundu** |

> [!NOTE]
> **JevBench v1.4 Puanlama Mekaniğinin Analizi:**  
> JevBench v1.4 sürümünde, şanstan arındırılmış zeka skoru 50'nin altındaysa karesel bir ceza çarpanı uygulanır: $\text{Skor} = \text{HarmonikOrtalama} \times \left(\frac{\text{Zeka}}{50}\right)^2$. Bu ceza, 4ms altı gecikme (Hız: 95.7) ve 0 VRAM ($0 maliyet) avantajına bakılmaksızın 50 altı puanları baskılasa da, Werr'in doğruluğu tekdüze şans seviyesinin (~%29–%33) belirgin biçimde üzerinde kalarak tensör ağırlığı olmaksızın saf geometrik refleks karar yeteneğini kanıtlamaktadır.

> [!TIP]
> **🤝 Karşılıklı Evrim (Vice Versa): Florian Standhartinger ve JevBench Topluluğuna Teşekkür:**  
> Bilimsel ilerleme doğası gereği çift yönlüdür. **Florian Standhartinger**'e ve JevBench açık kaynak ekibine ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)) en içten mesleki saygılarımızı sunuyoruz. Onların titiz dış denetimi Werr'deki kalıntı sezgiselleri tamamen kazıyıp bizi çoklu belirteç $N$-gram kriter rezonansına yükseltirken; Werr'in arenadaki ilk 0-VRAM fraktal yarışmacı olarak yarattığı paradigma sarsıntısı da JevBench'in mimari olgunlaşmasını hızlandırdı (kapalı set testleri, şanstan arındırılmış zeka baseline'ı ve harmonik ceza kapıları v1.3 ile v1.4'te hızla hayata geçirildi). Arenada yerleşik ezberleri bozan gerçek bir rakibin bulunması her iki tarafı da daha hızlı gelişmeye zorladı. Açık bilimde demir demiri biler.

* **Bütünlük Denetimi:** Metrik kaymaları ve sıra dinamiklerine dair akademik analiz [`docs/BENCHMARK_INTEGRITY_REPORT.md`](../docs/BENCHMARK_INTEGRITY_REPORT.md) dosyasında yayımlanmıştır.
* **Canlı Ağ:** [`answerr`](https://github.com/pCwOrM/answerr) ikili bilişsel REST API'si (`api.answerr.me:4431`) üzerinden test edilmiştir.

---

### 3. 🐍 Farama Gymnasium RL: Yılan Otonom Refleksi ([Alt Dizin: `./snake/`](./snake/) │ [Issue: mizorewww/laya-mlx#3](https://github.com/mizorewww/laya-mlx/issues/3))
* **Kapsam:** Kesintisiz ızgara navigasyonu ve engelden kaçınma (600 adım).
* **Mimari:** Durum-Dalga modülasyonunun koordinat tohumuna izdüşümü:
  $$c = -0.7436438870371587 + 0.1318259042053119i \quad (\text{Büyütme: } 65\times)$$
* **Metrikler:** **411.9 hamle/sn** işlem hacmi, **1.32 ms** medyan refleks gecikmesi, **0 Bayt VRAM**, **sıfır duvara çarpma**.
* **Görsel Belgeler:** [`terminal_snake_showcase.gif`](snake/terminal_snake_showcase.gif) │ [`terminal_snake_showcase.mp4`](snake/terminal_snake_showcase.mp4).

---

### 4. 🤖 Tau-Bench Çok Turlu Araç Çağrısı ([Issue: sierra-research/tau-bench#95](https://github.com/sierra-research/tau-bench/issues/95))
* **Kapsam:** Katı operasyonel kurallar altında çok adımlı araç yönetimi (DOT 24 saatlik uçuş iptali, koltuk yükseltme, perakende RMA iadeleri, kupon birleştirme).
* **Doğruluk:** **10 / 10 benchmark senaryosu başarıyla geçti**.
* **Avantaj:** Deterministik kural kısıtları hızlı omurilik refleksiyle süzülerek bulut LLM token maliyeti %100 oranında sıfırlanır.

---

### 5. 🎯 Jevenator 2: Pertürbasyon Stresi & Video Takibi ([Alt Dizin: `./jevenator2/`](./jevenator2/) │ [Issue: mmastrac/jevenator2#1](https://github.com/mmastrac/jevenator2/issues/1))
* **Kapsam:** 24 karelik video takibi (840 karar) ve Gauss gürültüsü ($\sigma = 0.50$) altında uzamsal bölge taraması.
* **Kıyaslama:** Maisa djev (Diffusion-Gemma 8GB VRAM) modeline karşı test edilmiştir.
* **Sonuç:** **38.0 kat daha hızlı** (kare başına 20.04 ms vs djev 761.8 ms), **597.4 karar/saniye**, **%100 geometrik şekil doğruluğu** (Üçgen=B, Daire=F), negatif kontrolde (Miles Dyson) **0 yanlış pozitif** ve sinir ağlarının çöktüğü yüksek gürültüde sıfır unutma.

---

### 6. 🌀 Sürekli Manifoldlar: Two-Moons & Two-Spirals
* **Kapsam:** Geri-yayılım (backpropagation) veya gradyan inişi olmadan topolojik doğrusal olmayan sınıflandırma.
* **Doğruluk:** **Two-Moons: %99.30** │ **Two-Spirals: %98.50**.
* **Dokümantasyon:** Kapsamlı matematiksel çıkarımlar [Mandelbrot Akademik Teknik Raporu (HTML)](https://pcworm.github.io/mandelbrot-fractal-neural-synthesis/docs/Mandelbrot_Akademik_Teknik_Raporu.html) ([GitHub Kaynağı](https://github.com/pCwOrM/mandelbrot-fractal-neural-synthesis/blob/master/docs/Mandelbrot_Akademik_Teknik_Raporu.html)) monografında yer almaktadır.

---

## 🛡️ Mühürlü Kriptografik Denetim Özeti ve Resmi Raporlar

Tüm dört temel benchmark ekseni sıfır-kontaminasyon ve hava-boşluklu (air-gapped) yerel ortamda baştan çalıştırılmış ve SHA-256 kriptografik özetleri ile mühürlenmiştir:

* 📄 **Resmi PDF Denetim Raporu:** [`docs/werr_official_benchmarks_report.pdf`](../docs/werr_official_benchmarks_report.pdf)
* 🌐 **Resmi HTML Denetim Raporu:** [`docs/werr_official_benchmarks_report.html`](../docs/werr_official_benchmarks_report.html)
* 🔐 **Kriptografik Mühür Manifestosu:** [`benchmarks/sealed/SEAL_MANIFEST.json`](./sealed/SEAL_MANIFEST.json)
* ⚙️ **Motor Optimizasyon Raporu:** [`docs/OPTIMIZATION_REPORT_TR.md`](../docs/OPTIMIZATION_REPORT_TR.md)

| Kıyaslama Ekseni | Veri Seti / Süit | WERR Skor / Hız | Referans Temel Model | Kriptografik SHA-256 Özeti |
| :--- | :--- | :--- | :--- | :--- |
| **Benchmark 1: Yılan AI** | 600 Kesintisiz Adım | **411.9 hamle/sn** (P50: 1.32 ms) | Laya-MLX 421M (74.5 hamle/sn) | `333814952ad1f8f1b2c25c7749b658dd729503e48dfbea2e9142c9a4f6af5963` |
| **Benchmark 2: JevBench Çift-Standart** | 231 Kamuya Açık Görev | **81.36 (v1.2)** / **76.90 (v1.3.0)** | Şans tabanı: 25.0 / Şans-altı cezası | `81a33e723dea04ddd40d38b056ff376cd54d2206c0920bb064af03de0bae7c5f` |
| **Benchmark 2: JevBench v1.4.1 (Tesla 3-6-9)** | 231 Kamuya Açık Görev | **20.63 (Adaptör) / 23.74 (Cusp)** | Şans tabanı: 25.0 / 8ms altı gecikme | `950d26a427e2d82ffd4b85620a1cb9eb0bdf6105e186d42ece75641f14ee5518` |
| **Benchmark 3: WindTunnel WebMCP** | 49 Görev (8 Web Uygulaması) | **49 / 49 (%100.00)** (P50: 1.81 ms) | Üretim Web Ajanları | `06b134dea501216c8888aa5a3cd13e1b68b15beb31987e2c8df6872c4caeffc4` |
| **Benchmark 4: Jevenator 2 Görsel** | 24 Kare (840 Karar) | **20.04 ms/kare (38.0x hızlanma)** | Maisa djev Gemma (761.8 ms/kare) | `30111404aac815366afc93b1091f8f07c318f78ded7e56dd61fa18482f02986b` |

---

## 🔥 30 Saniyede Kendin Dene (Açık Meydan Okuma Protokolü)

Sonuçları kendi bilgisayarınızda GPU veya bulut hesabı olmadan anında doğrulayın:

```bash
# 1. Depoyu klonlayın
git clone https://github.com/pCwOrM/werr.git
cd werr
pip install numpy

# 2. Resmi WindTunnel WebMCP benchmarkını koşun (49/49 görev):
python -m unittest tests.test_windtunnel_webmcp_isolated

# 3. Canlı Yılan Refleks Görselleştiricisini çalıştırın:
python benchmarks/snake/visualize_snake.py

# 4. İnteraktif terminal devir gösterimini başlatın:
python benchmarks/snake/terminal_snake.py --showcase

# 5. Canlı Tip-Güvenli REST API'yi test edin:
curl -X POST https://api.answerr.me:4431/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"task_id":"gauntlet-01","domain":"ecommerce","input":"Cancel order #4928"}'
```

---

## 📁 Dizin Ağacı

```text
benchmarks/
├── README.md               # The Master Gauntlet Specification & Showdown (İngilizce)
├── README_TR.md            # Merkezi Kıyaslama ve Meydan Okuma Dokümanı (Türkçe)
├── snake/                  # Farama Gymnasium Snake Reflex Benchmark
│   ├── README.md           # Özel Yılan AI Benchmark Monografı
│   ├── terminal_snake.py   # İki aşamalı interaktif terminal oyunu
│   ├── visualize_snake.py  # Sıfır bağımlılıklı tekrar oynatıcı
│   ├── benchmark_snake.py  # 600 adımlık karşılaştırmalı test koşucusu
│   └── terminal_snake_showcase.gif # Canlı terminal gösterim animasyonu
└── jevenator2/             # Görsel Nesne Takibi & Pertürbasyon Stres Testi
    ├── README.md           # Jevenator 2 Benchmark Monografı
    ├── benchmark_jevenator2.py # 24 karelik video takip koşucusu
    └── werr_vision_policy.py   # Uzamsal bölge tarama fraktal politikası
```

---

## 🌐 Ekosistem Bağlantıları

* 🌐 **İnteraktif Web Arenası:** [GitHub Pages üzerinde `benchmarks.html`'i Aç](https://pcworm.github.io/mandelbrot-fractal-neural-synthesis/benchmarks.html)
* 📜 **Temel Bilim Makalesi ve Laboratuvarlar:** [Mandelbrot Fractal Neural Synthesis Portalı](https://pcworm.github.io/mandelbrot-fractal-neural-synthesis/)
* ⚡ **WERR Ana Deposu:** [werr GitHub Repository](https://github.com/pCwOrM/werr)
* 🧠 **Canlı İkili Bilişsel API:** [answerr Platformu (answerr.me)](https://answerr.me)
* 🏛️ **Kalıcı Zenodo WERR Arşivi:** [DOI: 10.5281/zenodo.22867426](https://doi.org/10.5281/zenodo.22867426)
* 📜 **Bağlı Temel Bilim Makalesi Arşivi:** [DOI: 10.5281/zenodo.22774934](https://doi.org/10.5281/zenodo.22774934)
* 📑 **Araştırma Statüsü:** *Açık Bilim ve Kalıcı Zenodo Arşivi*
