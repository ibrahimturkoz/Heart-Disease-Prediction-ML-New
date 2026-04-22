import pandas as pd # Veri analizi ve dosya işlemleri için pandas kütüphanesini içe aktarır.
from sklearn.model_selection import train_test_split # Veriyi eğitim ve doğrulama seti olarak bölmek için kullanılır.
from sklearn.preprocessing import StandardScaler # Sayısal verileri (yaş, kolesterol vb.) aynı ölçeğe getirmek için kullanılır.
from sklearn.linear_model import LogisticRegression # İkili sınıflandırma (Hasta/Değil) için temel ve etkili algoritmayı çağırır.
from sklearn.metrics import accuracy_score, confusion_matrix # Modelin ne kadar doğru bildiğini ölçmek için metrikleri yükler.

# 1. VERİYİ YÜKLEME
df = pd.read_csv('heart.csv') # Kaggle'dan indirilen kalp hastalığı veri setini bir tablo olarak okur.

# 2. VERİ ÖN İŞLEME
# 'target' sütunu hastanın durumunu (1: Hasta, 0: Sağlıklı) gösterir.
X = df.drop('target', axis=1) # Hedef sütunu (target) dışındaki tüm tıbbi özellikleri X değişkenine atar.
y = df['target'] # Tahmin edilmek istenen sonuç sütununu y değişkenine atar.

# 3. VERİYİ BÖLME
# Verinin %20'si test, %80'i eğitim amacıyla rastgele şekilde bölünür.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. STANDARTLAŞTIRMA (Ölçeklendirme)
# Kolesterol (250) ve cinsiyet (1) gibi farklı aralıktaki sayıları birbiriyle uyumlu hale getirir.
scaler = StandardScaler() # Ölçeklendirme işlemini yapacak nesneyi oluşturur.
X_train = scaler.fit_transform(X_train) # Eğitim verilerini öğrenir ve standart bir ölçeğe sokar.
X_test = scaler.transform(X_test) # Test verilerini, eğitimde öğrenilen kurallara göre dönüştürür.

# 5. MODEL EĞİTİMİ
model = LogisticRegression() # Lojistik Regresyon modelini tanımlar.
model.fit(X_train, y_train) # Modelin verilerdeki örüntüleri (hastalık belirtilerini) öğrenmesini sağlar.

# 6. TAHMİN VE BAŞARI ANALİZİ
y_pred = model.predict(X_test) # Modelin daha önce görmediği test verileri üzerinde tahmin yapmasını sağlar.
basari_orani = accuracy_score(y_test, y_pred) # Tahminlerin gerçek sonuçlarla ne kadar eşleştiğini hesaplar.

print(f"Modelin Kalp Hastalığı Tahmin Başarısı: %{basari_orani * 100:.2f}") # Başarıyı yüzde olarak ekrana yansıtır.

# 7. KARMAŞIKLIK MATRİSİ (Confusion Matrix)
# Modelin kaç kişiye yanlışlıkla hasta dediğini veya kaç hastayı kaçırdığını gösterir.
print("\n--- Hata Matrisi ---") # Bölüm başlığını yazdırır.
print(confusion_matrix(y_test, y_pred)) # Doğru ve yanlış tahminlerin sayısal tablosunu gösterir.

# 8. TEKİL TEST (Yeni Hasta Örneği)
# Örnek veriler: Yaş=55, Cinsiyet=1, Göğüs Ağrısı=2, Kan Basıncı=130, Kolesterol=250...
yeni_hasta_verisi = [[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.2, 1, 0, 2]] # Örnek bir hasta profili oluşturur.
yeni_hasta_scaled = scaler.transform(yeni_hasta_verisi) # Veriyi modelin tanıması için ölçeklendirir.
tahmin = model.predict(yeni_hasta_scaled) # Modelin bu verilerle tahmin yapmasını sağlar.

if tahmin[0] == 1: # Eğer sonuç 1 ise:
    print("\nSonuç: Kalp Hastalığı Riski Yüksek!") # Kullanıcıya riskli olduğunu söyler.
else: # Eğer sonuç 0 ise:
    print("\nSonuç: Kalp Sağlığı Yerinde Görünüyor.") # Kullanıcıya sağlıklı olduğunu söyler.