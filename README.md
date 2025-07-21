# Göğüs Kanseri Tanılama Projesi

Bu proje, göğüs kanseri verisi üzerinde lojistik regresyon kullanarak tümörlerin iyi huylu (B) veya kötü huylu (M) olarak sınıflandırılmasını sağlamaktadır. Model, scikit-learn kütüphanesi ile eğitilmiş ve Tkinter tabanlı basit bir tahmin arayüzü içermektedir.

## Özellikler

- Verisetini keşif ve ön işleme adımları
- Standart ölçeklendirme (StandardScaler)
- Hiperparametre optimizasyonu (GridSearchCV)
- Model eğitimi, değerlendirme ve karışıklık matrisi görselleştirmesi
- Kaydedilmiş model (`breast_cancer_classification_model.pkl`)
- Tkinter ile kullanıcıdan ölçüm verisi alarak anlık tahmin

## Gereksinimler

- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- tkinter (Python ile birlikte gelir)

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

## Dosya Yapısı

- `breast-cancer.csv` : Ham veri dosyası (Göğüs kanseri ölçüm değerleri ve etiketler)
- `kanser_tespit.ipynb` : Veri keşfi, model eğitimi ve değerlendirme adımlarını içeren Jupyter Notebook
- `breast_cancer_classification_model.pkl` : Eğitilmiş modelin pickle dosyası
- `README.md` : Proje açıklamaları ve kullanım talimatları

## Kullanım

1. Depoyu klonlayın veya indirin:
   ```bash
   git clone <repository_url>
   cd gogus_kanseri_tespiti
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```

3. Model eğitimi ve değerlendirme için Jupyter Notebook'u açın:
   ```bash
   jupyter notebook kanser_tespit.ipynb
   ```

4. Eğitilmiş modeli kullanarak tahmin yapmak için terminal veya IDE üzerinde aşağıdaki Python kodunu çalıştırabilirsiniz:
   ```python
   import pandas as pd
   import joblib
   from sklearn.preprocessing import StandardScaler

   # Modeli ve verisetini yükle
   model = joblib.load('breast_cancer_classification_model.pkl')
   df = pd.read_csv('breast-cancer.csv')
   data = df.drop(['diagnosis', 'id'], axis=1)

   # Ölçeklendirici hazırla
   scaler = StandardScaler()
   scaler.fit(data)

   # Kullanıcıdan ölçüm değerleri alarak tahmin fonksiyonunu kullanın
   def predict_cancer(input_features: dict) -> str:
       user_df = pd.DataFrame([input_features])
       user_df = user_df[data.columns]
       user_scaled = scaler.transform(user_df)
       pred = model.predict(user_scaled)[0]
       return 'Kötü Huylu (M)' if pred == 1 else 'İyi Huylu (B)'

   # Örnek kullanım
   sample = {col: float(input(f"{col}: ")) for col in data.columns}
   print("Tahmin:", predict_cancer(sample))
   ```

5. Tkinter tabanlı GUI arayüzü ile çalıştırmak için:
   ```bash
   python -c "import kanser_tespit; kanser_tespit.get_user_input(data);"
   ```
   veya notebook içindeki ilgili hücreyi çalıştırın.

## Sosyal Medya

- 🐦 X: https://x.com/eminarukk
- ▶️ YouTube: https://youtube.com/@eminaruk
- 🌐 Web Sitesi: https://eminaruk.com


