## 🏭 Fabrika Enerji Yönetimi – Q-Learning ile Işık & Fan Kontrolü

Bu projede, 6×6’lık bir fabrika alanında bulunan 4 odanın ışık ve havalandırma sistemlerini pekiştirmeli öğrenme (Q-Learning) kullanarak otomatik olarak yönetmeyi amaçladık.
Ajanın amacı enerji tasarrufu sağlarken aynı zamanda oda konforunu korumaktır.

Basit bir yapıya sahip olsa da, gerçek fabrika otomasyonlarının temel mantığını örnekleyen bir simülasyon ortaya çıkmaktadır.

## 🎯 Projenin Amacı

* Gereksiz enerji tüketimini azaltmak.

* Oda içi sıcaklığı belirli bir aralıkta tutmak.

* Aktivite olduğunda ışık/fanın doğru şekilde açılmasını sağlamak.

* Pahalı enerji saatlerinde tasarruf etmek.

* Çevresel değişimlere göre kendi kendine optimal politika öğrenmek.

## 🔧 Q-Learning Mantığı

Ajan, her durumda yapacağı eylemlerin değerini Q tablosunda saklar.
Zaman içinde aldığı ödüllere göre bu tablo güncellenir.

Kullanılan güncelleme formülü:
```bash
Q(s, a) = Q(s, a) + α * (r + γ * max Q(s' , a') - Q(s, a))
```
* s → mevcut durum

* a → seçilen eylem

* r → alınan anlık ödül

* s' → yeni durum

* α → öğrenme oranı

* γ → geleceğe verilen önem

## 📦 Gerekli Kütüphaneler

Projenin çalışması için aşağıdaki Python kütüphanelerinin yüklü olması gerekir:
```bash
pip install numpy matplotlib imageio pillow
```
Kullanılan kütüphanelerin görevleri:

* numpy → Q tablosu, matematiksel işlemler

* matplotlib → grafik çizimi, grid görselleştirme

* imageio → GIF oluşturma

* pillow (PIL) → görsel işleme desteği

## 🧩 Kod Yapısının Genel Taslağı

* FactoryRoomEnv

  * reset()

  * step()

* Durumları tablo indeksine çevirme

* Q-learning eğitimi

* 4 odanın ayrı ayrı eğitilmesi

* 6×6 grid görselleştirmesi

* GIF oluşturma

* PNG grafikleri


## 🏠 Ortam Tasarımı (Environment)

Fabrika 6×6 bir grid olarak modellenmiştir.
Bu gridde sadece 4 hücre oda olarak kullanılmaktadır:
```bash
(1, 1)
(1, 4)
(4, 1)
(4, 4)
```
Diğer tüm hücreler sabit gri gösterilir ve herhangi bir hesaplama içermez.

## 🔥 Sıcak Oda Mantığı

* Oda 1 ve Oda 2 → fan kapalıyken sıcaklığı hızlı yükselen odalar
* Oda 3 ve Oda 4 → normal sıcaklık değişimli odalar

Bu sayede ajan gerçekten fan açmayı öğrenebiliyor.

## 🌡️ Durum Uzayı (State)

Her oda için durum şu 4 bilgiyi içeriyor:

1. Aktivite (0/1)

2. Sıcaklık (0 = soğuk, 1 = konfor, 2 = sıcak)

3. Saat (0 = gündüz, 1 = gece)

4. Enerji fiyatı (0 = ucuz, 1 = pahalı)

Toplam durum sayısı:
```bash
2 × 3 × 2 × 2 = 24
```
## ⚡ Aksiyon Uzayı (6 Eylem)
```bash
0 → ışık kapalı, fan kapalı
1 → ışık açık
2 → fan düşük
3 → ışık + fan düşük
4 → fan yüksek
5 → ışık + fan yüksek
```
| Renk     | Anlam      |
| -------- | ---------- |
| Gri      | kapalı     |
| Sarı     | ışık       |
| Lacivert | fan        |
| Mavi     | ışık + fan |

## 🏆 Ödül Fonksiyonu

Ajanın kararını yönlendiren temel ödül yapısı:

✔ Aktivite varken ışık açıksa → +3

✔ Sıcak bir oda + fan açıksa → +3

✔ Oda boş ama cihazlar açıksa → –3

✔ Pahalı enerji saatinde oda boş & her şey kapalı → +2

Enerji tüketimi ayrıca maliyet olarak düşünülür:

* ışık = 1 birim

* düşük fan = 1.5 birim

* yüksek fan = 3 birim

* pahalı saatlerde ×2 çarpanı

Bu maliyet ödülden düşülür → gereksiz tüketim cezalandırılır.

## 🧠 Eğitim Süreci

Her oda bağımsız olarak eğitilir.

Parametreler:
| Parametre   | Değer               |
| ----------- | ------------------- |
| Episode     | 800                 |
| Adım sayısı | 40                  |
| Alfa        | 0.1                 |
| Gamma       | 0.95                |
| Epsilon     | 1 → 0 lineer azalma |

Son episode’da sıcaklık ve enerji maliyeti kaydedilir.

## 📦 Çıktılar

Proje çalıştığında 3 dosya üretir:

🎞️ 1) factory.gif

4 odanın zaman içinde aldığı aksiyonları gösterir.

🌡️ 2) temperature.png

Son episode boyunca sıcaklık değişimini gösterir.

⚡ 3) energy.png

Enerji maliyetinin zaman içindeki değişimi.



