# 🏥 Heart-Disease-Prediction-ML


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje, bir hastanın tıbbi ölçümlerine (yaş, kolesterol, tansiyon vb.) bakarak kalp hastalığı olup olmadığını tahmin eden bir **Makine Öğrenmesi** çalışmasıdır. Projede sınıflandırma başarısı yüksek olan **Lojistik Regresyon** algoritması kullanılmıştır.

---

## 🚀 Model Performansı
* **Doğruluk Oranı (Accuracy):** %85+ (Veri setine göre değişkenlik gösterebilir)
* **Algoritma:** Logistic Regression
* **Veri İşleme:** StandardScaler (Ölçeklendirme)

---

## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **Pandas:** Veri manipülasyonu.
* **Scikit-Learn:** Makine öğrenmesi modeli ve veri ölçeklendirme.
* **Matplotlib / Seaborn:** Veri görselleştirme (Opsiyonel).

---

## 📊 Veri Seti Hakkında
Proje, Kaggle üzerindeki popüler [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) kullanılarak geliştirilmiştir. 
* **Örnek Sayısı:** 1025+ satır
* **Özellik Sayısı:** 13 tıbbi parametre (Girdi) + 1 Hedef Sütun (Çıktı)

---

## 📈 Model Akış Şeması
1. **Veri Keşfi:** Eksik verilerin kontrol edilmesi.
2. **Ölçeklendirme:** Sayısal verilerin (örneğin kolesterol) standartlaştırılması.
3. **Eğitim:** Lojistik Regresyon modelinin eğitilmesi.
4. **Test:** Hata matrisi (Confusion Matrix) ile yanlış tahminlerin analizi.

---

## 💻 Örnek Tahmin Bloğu
```python
# Yeni bir hasta verisi tahmini
yeni_hasta = [[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.2, 1, 0, 2]]
tahmin = model.predict(scaler.transform(yeni_hasta))
# Çıktı: 1 (Hastalık Riski) veya 0 (Sağlıklı)
```
---
## Geliştirici: İbrahim Türköz
