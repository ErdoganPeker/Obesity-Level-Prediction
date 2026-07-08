# ⚕️ Obesity Level Prediction — Obezite Seviyesi Tahmini

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat&logo=tensorflow&logoColor=white)

> Bireylerin yaşam tarzı, beslenme alışkanlıkları ve demografik özelliklerine göre obezite durumunu ve seviyesini tahmin eden çok sınıflı makine öğrenmesi ve derin öğrenme projesi.

---

## ✨ Özellikler

- Obezite veri setinin kapsamlı Keşifsel Veri Analizi (EDA)
- Obeziteyi etkileyen faktörlerin istatistiksel ve görsel incelemesi
- Çok sınıflı (multi-class) obezite seviyesi sınıflandırması
- Kategorik ve sayısal değişkenler için özellik mühendisliği
- Çoklu ML modeli karşılaştırması ve ensemble yöntemler
- Derin öğrenme modeli ile geliştirilmiş sınıflandırma
- Sınıf bazlı precision, recall ve F1-Score değerlendirmesi

---

## 🛠️ Teknoloji Yığını

| Kategori | Araçlar |
|---|---|
| ML | scikit-learn (Random Forest, SVM, KNN, Decision Tree, Logistic Regression) |
| Ensemble | XGBoost, Gradient Boosting |
| DL | TensorFlow / Keras |
| Veri İşleme | Pandas, NumPy |
| Görselleştirme | Matplotlib, Seaborn |
| Ortam | Jupyter Notebook |

---

## 📂 Proje Yapısı

```
Obesity-Level-Prediction/
├── Obesity Level Prediction.ipynb   # Ana analiz ve model notebook'u
└── .gitignore
```

---

## 📊 Obezite Seviyeleri (Çok Sınıflı Hedef)

| Sınıf | Açıklama |
|---|---|
| Insufficient_Weight | Yetersiz kilo |
| Normal_Weight | Normal kilo |
| Overweight_Level_I | Hafif kilolu I |
| Overweight_Level_II | Hafif kilolu II |
| Obesity_Type_I | Obezite Tip I |
| Obesity_Type_II | Obezite Tip II |
| Obesity_Type_III | Obezite Tip III (Morbid) |

**Kullanılan Özellikler:**
- Yaş, boy, kilo, cinsiyet
- Yüksek kalorili yiyecek tüketimi (FAVC)
- Günlük öğün sayısı (NCP), su tüketimi (CH2O)
- Fiziksel aktivite sıklığı (FAF)
- Alkol tüketimi, sigara kullanımı
- Ulaşım yöntemi (MTRANS)
- Aile obezite geçmişi (family_history)

---

## 🚀 Başlangıç

### Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow jupyter
```

### Çalıştırma

```bash
git clone https://github.com/ErdoganPeker/Obesity-Level-Prediction.git
cd Obesity-Level-Prediction
jupyter notebook "Obesity Level Prediction.ipynb"
```

---

## 👤 Geliştirici

**Erdoğan Yasin Peker**
[GitHub](https://github.com/ErdoganPeker) · [LinkedIn](https://www.linkedin.com/in/erdogan-yasin-peker-b107ba24b/)
