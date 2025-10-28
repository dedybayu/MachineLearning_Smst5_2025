# k-Nearest Neighbors (kNN)

## Apa itu kNN?

Algoritma *k-Nearest Neighbors* (k-NN) merupakan salah satu metode paling sederhana dalam pembelajaran terbimbing (*supervised learning*). Alih-alih membangun model matematis yang kompleks, k-NN bekerja dengan prinsip **“belajar dari tetangga terdekat”**. Algoritma ini hanya menyimpan seluruh data latih, lalu untuk data baru, mencari sejumlah *k* titik data yang paling dekat dengannya dan menentukan kelas berdasarkan informasi dari tetangga tersebut. Meskipun sederhana, kNN dapat digunakan untuk kebutuhan klasifikasi maupun regresi.

* **Untuk klasifikasi**, keputusan ditentukan dengan **voting mayoritas** dari label tetangga.
* **Untuk regresi**, nilai prediksi adalah **rata-rata** dari nilai tetangga terdekat.

Pada modul ini, kita hanya akan berfokus pada pemanfaatan kNN sebagai algoritma klasifikasi.

\
![](https://images.gitbook.com/__img/dpr=2,width=760,onerror=redirect,format=auto,signature=1720767645/https%3A%2F%2Ffiles.gitbook.com%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%2FYHOv2XH8FJMJSMsnhwNa%2Fuploads%2FJrgyhrdMhZbd0JMoR25C%2Fimage.png%3Falt%3Dmedia%26token%3D185f9f1d-3769-41af-b6f0-ce0c2d98e6dd)Ilustrasi cara kerja kNN dengan  (Muller, 2023)Selanjutnya, dikarenakan keputusan label berdasarkan hasil voting, ada pendapat dan *best practice* yang menyatakan bahwa nilai  yang aman adalah nilai ganjil. Hal ini dikarenakan jika jumlah  berjumlah genap ada potensi untuk jumlah vote berimbang.

***

### Jarak Pada kNN <a href="#jarak-pada-knn" id="jarak-pada-knn"></a>

Seperti yang telah dijelaskan sebelumnya, kNN menggunakan fungsi jarak untuk menentukan tetangga terdekatnya. Lalu, bagaimana cara menentukan jaraknya? Kita dapat menggunakan fungsi jarak seperti,

1. Euclidean distance (umum digunakan)
2. Manhattan distance
3. Minkowski distance
4. Cosine distance

Penggunaan jenis jarak bergantung dengan kasus yang akan diselesaikan. Namun, dikarenakan terkadang fitur memiliki skala yang berbeda, kita mungkin membutuhkan **proses standardisasi sebelum melakukan perhitungan jarak**.

***

### Analisis Performa kNN <a href="#analisis-performa-knn" id="analisis-performa-knn"></a>

Untuk mengetahui performa dari model kNN, kita dapat melakukan analisis terhadap dua hal, pertama adalah decision boundaries, kedua performa berdasarkan metric klasifikasi untuk setiap nilai . Untuk decision boundaries, cara ini dapat dilakukan dengan amatan visual jika fitur yang dibandingkan tidak lebih dari 3. Decision boundaries akan lebih "halus" jika jumlah tetangga yang digunakan semakin banyak. Akan tetapi, perhatikan jumlah fitur / kompleksitas dari data yang digunakan.

> * Gunakan beberapa (sedikit) tetangga pada model yang kompleks
> * Gunakan banyak tetangga pada model yang sederhana

Sebagai contoh,

```python
# Decision Boundaries - Introduction to Machine Learning
# Andreas C. Müller, Sarah Guido (2023)
# pip install mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn


X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
  # the fit method returns the object self, so we can instantiate
  # and fit in one line
  clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
  mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title("{} neighbor(s)".format(n_neighbors))
  ax.set_xlabel("feature 0")
  ax.set_ylabel("feature 1")
  axes[0].legend(loc=3)
```

Akan menghasilkan,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2Fpe7Y3djaTbnzbkEiPogA%2Fimage.png?alt=media&#x26;token=2c0ad13c-081c-4bc8-9817-8800c18d2601" alt=""><figcaption></figcaption></figure>

Dapat dilihat pada gambar sebelah kiri, penggunaan 1 tetangga akan menjadikan decision boundary akan mengikuti pola data dengan "kaku". Jumlah tetangga yang lebih banyak akan menjadikan batas keputusan menjadi lebih halus. Namun apakah dengan cara ini performa dapat lebih baik?

Kita perlu melakukan analisis lebih lanjut untuk siap jumlah tetangga yang digunakan.

Contoh,

```python
# Introduction to Machine Learning
# Andreas C. Müller, Sarah Guido (2023)
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []

# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
  # build the model
  clf = KNeighborsClassifier(n_neighbors=n_neighbors)
  clf.fit(X_train, y_train)
  # record training set accuracy
  training_accuracy.append(clf.score(X_train, y_train))
  # record generalization accuracy
  test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
```

Hasilnya,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FbS44T4XD1TYucMSH774c%2Fimage.png?alt=media&#x26;token=26511f2b-4d43-4560-ae05-c0dbd8936fd1" alt="" width="432"><figcaption></figcaption></figure>

Dari hasil grafik dapat dilihat bahwa, jika tetangga yang digunakan hanya 1, performa data training akan sangat baik akan tetapi bertolak belakang dengan data testing. Hal ini menunjukkan fenomena overfitting dimana model tidak dapat mengeneralisasi data dengan cukup baik. Akan tetapi jika 10 tetangga digunakan, maka kompleksitas model akan menjadi lebih sederhana sehingga performa (akurasi) malah menurun. Dari grafik kita dapat mengetahui bahwa jumlah tetangga yang dapat mengakomodasi performa dengan cukup baik dari sisi training dan testing adalah 6 tetangga dilihat dari grafik performa training dan testing yang hampir berdekatan.






# Naive Bayes

## Apa itu Naive Bayes?

Sebelum kita mempelajari tentang model naïve bayes, ada baiknya kita memahami apa itu kaidah bayes. Konsep utama dari kaidah bayes adalah mencari nilai probabilitas (peluang) dari sebuah kejadian berdasarkan nilai probabilitas kejadian lain yang diketahui. Secara matematis, kaidah bayes dapat digambarkan dengan menggunakan Persamaan berikut.

$$
P(A|B)=\frac{P(A)P(B|A)}{P(B)}
$$

Dimana $$A$$ dan $$B$$ adalah sebuah kejadian dan nilai $$P(A)$$ merupakan peluang kejadian $$A$$ dan $$P(B)$$ merupakan peluang kejadian $$B$$. $$P(A|B)$$ adalah peluang kejadian $$A$$ setelah kita mengetahui kejadian $$B$$, begitu juga sebaliknya untuk nilai peluang $$P(B|A)$$.

Sebagai contoh, kita akan merencanakan untuk pergi untuk melakukan kegiatan piknik. Kondisi saat ini didapati cuaca sedang berawan. Kita juga mendapatkan fakta bahwa,

* 50% dari hari hujan diawali dengan cuaca berawan
* Keadaan berawan sering terjadi pada pagi hari dengan peluang 40%
* Kondisi umum saat ini cenderung kering dan jarang terjadi hujan. Hujan hanya terjadi 3 hari sepanjang 30 hari.

Kemudian kita ingin mengetahui peluang terjadinya hujan pada siang hari ini sebelum kita memutuskan untuk pergi piknik.

Berdasarkan fakta-fakta tersebut, kita dapat memformulasikannya ke dalam bentuk bayes, yaitu,

* Menentukan kejadian $$A$$ dan kejadian $$B$$. Pada kasus ini, kita berasumsi bahwa kejadian $$A$$ adalah hujan, sedangkan kejadian $$B$$ adalah kondisi berawan. Sehingga didapatkan $$P(Hujan)=P(A)$$ dan $$P(Berawan)=P(B)$$.
* 50% dari hari hujan diawali cuaca berawan → $$P(Berawan│Hujan)=P(B│A)=50%=0.5$$.
* Keadaan berawan sering terjadi di pagi hari dengan peluang 40% → $$P(B)=40%=0.4$$.
* Kondisi kering dan jarang terjadi hujan dapat kita modelkan menjadi P(A) dengan peluang $$3/30=10%=0.1$$ → $$P(A)=0.1$$.
* Sehingga untuk menghitung peluang terjadinya hujan pada siang hari, kita dapat modelkan menjadi $$P(A|B)$$.

Didapatkan nilai $$P(A|B)$$ adalah,

$$
P(A|B)=\frac{P(A)P(B|A)}{P(B)}=\frac{0.1\*0.5}{0.4}=0.125=12.5%
$$

Dengan demikian, didapatkan peluang terjadinya hujan pada siang hari adalah 12.5%.

Lalu apa hubungan contoh tersebut dengan model Naïve Bayes?

Pada model naïve bayes kita dapat menghitung peluang sebuah label pada kelas dalam kasus klasifikasi berdasarkan peluang setiap fitur untuk label tersebut. Sehingga, untuk menghitung peluang label $$y$$ berdasarkan fitur-fitur $$x\_n$$ pada naïve bayes dilakukan dengan Persamaan berikut.

$$
P(y|x\_1, x\_2, . . . ,x\_n)=\frac{P(y)P(x\_1,x\_2,...,x\_n)}{P(x\_1,x\_2, ...,x\_n)}
$$

Setelah mendapatkan peluang dari label $$y$$ berdasarkan fitur-fitur $$x\_n$$, maka proses terakhir dari model naïve bayes adalah membandingkan peluang untuk setiap label yang dimiliki. Label dengan peluang terbesar adalah hasil prediksi dari model naïve bayes.

Lalu mengapa naïve bayes dikatakan naif? Hal ini dikarenakan naïve bayes tidak memperhitungkan hubungan atau korelasi antar fitur. Dalam praktiknya, terdapat beberapa jenis model naïve bayes berdasarkan sumber data yang digunakan. Jika data yang digunakan bersifat diskrit, maka model naïve bayes yang paling cocok digunakan adalah multinomial naïve bayes. Jika data kita berupa data kategorikal, maka kita dapat menggunakan model categorical naïve bayes. Sedangkan jika data yang digunakan bersifat kontinu, maka kita dapat menggunakan model gaussian naïve bayes. Akan tetapi perlu dicatat, pada model gaussian naïve bayes, model berasumsi bahwa data pada fitur terdistrubusi secara normal (distribusi gaus)









# Praktikum 1

⬇️ Download Dataset ⬇️

{% file src="<https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FQTyyBdSBRJWE4J8SUjjr%2Firis.csv?alt=media&token=f0ebca2e-8dc4-41cd-a0ce-37741bfdcb87>" %}

## Langah 1 - Load Data

{% code lineNumbers="true" %}

```python
# Load data
import pandas as pd
data = pd.read_csv('iris.csv')
data.head()
```

{% endcode %}

## Langkah 2 - Eksplorasi Data

Cek struktur data informasi deskriptif data

{% code lineNumbers="true" %}

```python
data.info()
data.describe()
data['species'].value_counts()
```

{% endcode %}

## Langkah 3 - Visualisasi Data

Lakukan visualisasi data untuk mengetahui distribusi dan korelasi setiap fitur terhadap label.

{% code lineNumbers="true" %}

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, hue='species')
plt.show()
```

{% endcode %}

Hasilnya,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FEgFZjUxWFjmtw4no0tZw%2Fimage.png?alt=media&#x26;token=7cf79f18-9d90-4dbe-b48a-fd0e694e658f" alt="" width="563"><figcaption></figcaption></figure>

## Langkah 4 - Preprocessing

Pada tahap ini, kita akan memisahkan antara label dengan fitur yang akan digunakan. Selain itu, untuk alasan pembelajaran, kita akan melakukan standardisasi dari fitur yang akan digunakan.&#x20;

> Pada kasus Iris Dataset, seluruh fitur sudah dalam satuan yang sama yaitu cm.

{% code lineNumbers="true" %}

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.iloc[:, :-1]   # semua kolom kecuali label
y = data.iloc[:, -1]    # kolom label terakhir

# Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

{% endcode %}

## Langkah 5 - Buat Model kNN

{% code lineNumbers="true" %}

```python
from sklearn.neighbors import KNeighborsClassifier

# Tentukan nilai K (misalnya 3)
knn = KNeighborsClassifier(n_neighbors=3)

# Latih model
knn.fit(X_train, y_train)
```

{% endcode %}

## Langkah 6 - Evaluasi Model kNN

Pada tahap ini kita akan melakukan evaluasi terhadap model kNN yang telah dibuat sebelumnya. Metrik utama yang akan digunakan adalah akurasi dan detail analisis menggunakan confusion metrics.

{% code lineNumbers="true" %}

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = knn.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))
```

{% endcode %}

Di dapatkan hasil

```
Akurasi: 1.0
Confusion Matrix:
 [[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]
Laporan Klasifikasi:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       1.00      1.00      1.00        13
   virginica       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

```

Dapat dilihat bahwa model dapat bekerja dengan performa sempurna pada data testing. Namun apakah memang nilai $$k$$ adalah nilai yang terbaik? Selanjutnya kita akan mengevaluasi setiap nilai $$k$$.

{% code lineNumbers="true" %}

```python
acc = []
for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc.append(model.score(X_test, y_test))

plt.plot(range(1, 11), acc, marker='o')
plt.title('Nilai K vs Akurasi')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.show()

```

{% endcode %}

Hasilnya,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FqR4EymR2iB1y4ksmzlr1%2Fnikai%20k.jpg?alt=media&#x26;token=24364bd1-d54f-4f93-bd74-dff0d150f0c7" alt="" width="438"><figcaption></figcaption></figure>

Dapat dilihat bahwa performa data test mendapatkan nilai sempurna pada $$k=3$$ dilanjutkan dengan $$k=6$$ hingga $$k=10$$. Meskipun demikian, perlu dicatat bahwa kNN tidak pernah menyimpan bobot hasil training,  proses klasifikasi dilakukan secara langsung sesuai dengan jumlah tetangga sehingga hasil ini masih perlu dibandingkan dengan hasil data training. Gambar di bawah ini merupakan perbandingan performa akurasi antara data training dan data testing.

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FGCeIdDmhgMHAcyBP9Veg%2Fneigh.jpg?alt=media&#x26;token=4d7e0cb9-62fd-45e8-9a4d-49f75f0f798c" alt=""><figcaption></figcaption></figure>











# Praktikum 2

## Intro

Pada percobaan ini kita akan menggunakan data dummy (sintentis) untuk membuat sebuah model Naive Bayes. Untuk membuat data dummy, kita dapat menggunakan fungsi `make_classification` dari library scikit-learn. Selanjutnya, kita akan membuat model Multinomial Naive Bayes dengan menggunakan `MultinomialNB` dan model Gaussian Naive Bayes menggunakan `GaussianNB`.

## Langkah 1 - Buat Dataset Dummy

{% code lineNumbers="true" %}

```python
import numpy as np
from sklearn.datasets import make_classification

# Membuat data dummy
# Hasil dari make_classification berupa data fitur X dan label y
# Label y akan berupa data yang sudah di encode (angka)
X,y = make_classification(n_samples=30, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, shuffle=False)

# Secara defalt, make_classfication menghasilkan nilai float
# Kita perlu merubah dalam bentuk diskrit

# Absolutekan nilai
X = np.absolute(X)

# Bulatkan nilai ke 2 angka dibelakang koma
# Kalikan dengan 100 supaya tidak ada lagi koma
X = np.round(X, 2) * 100

# Ubah ke dalam bentuk integer
X = X.astype(int)

# Cek Hasil
print(X)
print(y)
```

{% endcode %}

Parameter yang digunakan pada fungsi `make_classification` adalah,

* `n_samples`: jumlah sampel yang diinginkan
* `n_features`: jumlah fitur yang digunakan
* `n_classes`: jumlah kelas
* `n_informative`: jumlah fitur yang memiliki korelasi dengan kelas
* `n_redundant`: jumlah fitur yang tidak memiliki korelasi dengan kelas
* `n_repeated`: jumlah fitur yang diulang

## Langkah 2 (Opsional) - Membuat Data Frame

Agar data lebih mudah untuk dibaca, maka kita akan membuat DataFrame dengan menggunakan library Pandas berdasarkan data dummy yang telah dibuat sebelumnya.

{% code lineNumbers="true" %}

```python
import pandas as pd

# Reshape label y menjadi 2D
# Hal ini dilakukan karena kita akan menggabungkannya dengan data fitur X
y_new = y.reshape(len(y), 1)

# Gabungkan fitur X dan label y dalam data array
data = np.concatenate((X, y_new), axis=1)

# Definisikan nama kolom
nama_kolom = ['Fitur 1', 'Fitur 2', 'Label']

# Buat Data Frame
df = pd.DataFrame(data, columns=nama_kolom)

# Cek Data Frame
df.head()
```

{% endcode %}

## Langkah 3 (Opsional) - Labeling

&#x20;Dikarenakan label masih berbetuk encoding angka, untuk mempermudah pembacaan data, kita dapat mengubah bentuknya dalam bentuk kategorial

{% code lineNumbers="true" %}

```python
# Definisikan nama label
labels = {
    1 : 'Kelas A',
    0 : 'Kelas B'
}

# Copy Data Frame untuk menyimpan Data Frame baru
# dengan label yang mudah untuk dibaca
df_label = df.copy()

# Ubah label dengan fungsi mapping dari Pandas
# pada Data Frame df_label
df_label['Label'] = df_label['Label'].map(labels)

# Cek Data Frame df_label
df_label.head()
```

{% endcode %}

## Langkah 4 - Visualisasi Data

{% code lineNumbers="true" %}

```python
import matplotlib.pyplot as plt

# Definisikan warna untuk setiap kelas
colors = {
    'class_a': 'MediumVioletRed',
    'class_b': 'Navy'
}

# Kelompokkan label berdasarkan nama kelas
gb = df_label.groupby(['Label'])
class_a = gb.get_group('Kelas A')
class_b = gb.get_group('Kelas B')

# Plot
plt.scatter(x=class_a['Fitur 1'], y=class_a['Fitur 2'], c=colors['class_a'])
plt.scatter(x=class_b['Fitur 1'], y=class_b['Fitur 2'], c=colors['class_b'])
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
plt.legend(['Kelas A', 'Kelas B'])
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
plt.show()
```

{% endcode %}

Hasilnya,

<figure><img src="https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FAxwI4d6uKunra3Uu0CBV%2Fimage.png?alt=media&#x26;token=16a0d2df-2cab-4d50-ab0e-8cd897f09cef" alt=""><figcaption></figcaption></figure>

## Langkah 5 - Model Multinomial Naive Bayes

Selanjutnya buat model naive bayes dengan jenis multinomoial. Sejatinya, ***model multinomial digunakan untuk fitur yang bersifat diskrit*** (e.g. jumlah kata untuk klasifikasi teks). Akan tetapi kita akan mencoba menggunakan model ini untuk konteks data kontinu ***hanya sebagai pembelajaran***.

{% code lineNumbers="true" %}

```python
from sklearn.naive_bayes import MultinomialNB # class untuk model MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # evaluasi model berdasarkan akurasi

# Inisiasi obyek MultinomialNB
mnb = MultinomialNB()

# Kita dapat langsung menggunakan fitur X dan label y
# hasil dari proses pembuatan data dummy

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=30)

# Fit model
# Label y harus dalam bentuk 1D atau (n_samples,)
mnb.fit(X_train, y_train)

# Prediksi dengan data training
y_train_pred = mnb.predict(X_train)

# Evaluasi akurasi training
acc_train = accuracy_score(y_train, y_train_pred)

# Prediksi test data
y_test_pred = mnb.predict(X_test)

# Evaluasi model dengan metric akurasi
acc_test = accuracy_score(y_test, y_test_pred)

# Print hasil evaluasi
print(f'Hasil akurasi data train: {acc_train}')
print(f'Hasil akurasi data test: {acc_test}')
```

{% endcode %}

Hasilnya,

```
Hasil akurasi data train: 0.6190476190476191
Hasil akurasi data test: 0.7777777777777778
```

## Langkah 6 - Model Gaussian Naive Bayes

Model Gaussian lebih cocok digunakan untuk data kontinu yang kita miliki, hal ini dikarenakan model ini menggunakan distribusi gaussian (normal) yang secara alami memiliki rentang dengan nilai kontinu.

{% code lineNumbers="true" %}

```python
from sklearn.naive_bayes import GaussianNB # class untuk model GaussianNB

# Inisiasi obyek Gaussian
gnb = GaussianNB()

# Kita menggunakan split data training dan testing
# yang sama dengan model multinomial

# Fit model
# Label y harus dalam bentu 1D atau (n_samples,)
gnb.fit(X_train, y_train)

# Prediksi dengan data training
y_train_pred_gnb = gnb.predict(X_train)

# Evaluasi akurasi training
acc_train_gnb = accuracy_score(y_train, y_train_pred_gnb)

# Prediksi test data
y_test_pred_gnb = gnb.predict(X_test)

# Evaluasi model dengan metric akurasi
acc_test_gnb = accuracy_score(y_test, y_test_pred_gnb)

# Print hasil evaluasi
print(f'Hasil akurasi data train (Gaussian): {acc_train_gnb}')
print(f'Hasil akurasi data test (Gaussian): {acc_test_gnb}')
```

{% endcode %}

Hasilnya,

```
Hasil akurasi data train (Gaussian): 0.6666666666666666
Hasil akurasi data test (Gaussian): 0.5555555555555556
```

Meskipun hasilnya tidak jauh berbeda dengan model multimodal, secara teoritis kita telah menerapkan langkah yang benar dalam membuat sebuah model klasifikasi dengan menggunakan Naive Bayes.











# Praktikum 3

## Intro

Pada percobaan ini, kita akan menggunakan nilai multinomial untuk melakukan klasifikasi dengan Naive Bayes. Nilai multinomial adalah data yang nilainya didapatkan dari proses menghitung. Sehingga, pada konteks fitur, nilai multinomial fitur berdasarkan proses perhitungan (counting) probabilitas kemunculan fitur tersebut dalam sebuah data. Contoh klasik fitur multinomial adalah perhitungan jumlah kata pada klasifikasi teks.Pada percobaan ini, kasus klasifikasi teks diberikan untuk mempermudah pemahaman terhadap algoritma Naive Bayes tipe Multinomial.

Kita akan menggunakan data `spam.csv` yang berisi data teks sms dengan label **spam** dan **ham**. Spam adalah sms sampah, sedangkan ham adalah sebaliknya.

{% file src="<https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2FAmbExXlgeOySgkTA0OfS%2Fspam.csv?alt=media&token=14477551-ac5c-4bf7-84ab-70f53a129452>" %}

## Langkah 1 - Load Data

Pada tahap ini kita akan *loading* data ke dalam data frame dan melakukan inspeksi sederhana untuk memastikan apakah kita perlu proses pra pengolahan data sebelum melakukan ekstraksi fitur dan permodelan.

{% code lineNumbers="true" %}

```python
import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1') # spesifiksi encoding diperlukan karena data tidak menggunakan UTF-8

df.head()
```

{% endcode %}

| Text | v1   | v2                                                | Unnamed: 2 |
| ---- | ---- | ------------------------------------------------- | ---------- |
| 0    | ham  | Go until jurong point, crazy.. Available only ... | NaN        |
| 1    | ham  | Ok lar... Joking wif u oni...                     | NaN        |
| 2    | spam | Free entry in 2 a wkly comp to win FA Cup fina... | NaN        |
| 3    | ham  | U dun say so early hor... U c already then say... | NaN        |
| 4    | ham  | Nah I don't think he goes to usf, he lives aro... | NaN        |

Terdapat 3 kolom yang tidak bermanfaat untuk proses selanjutnya, maka kita perlu membuang kolom tersebut. Selain itu, untuk memudahkan pembacaan data, kita juga akan mengubah nama kolom **v1** yang berupa label dan **v2** yang berupa teks sms.

## Langkah 2 - Preprocessing

Beberapa hal yang akan dilakukan pada tahap ini yaitu,

1. Drop kolom yang tidak digunakan
2. Ubah nama kolom v1 (label) dan v2 (teks sms)
3. Inspeksi Data
4. Encode label
5. Memisahkan fitur dengan label

### Langkah 2a - Drop Kolom

{% code lineNumbers="true" %}

```python
# Drop 3 kolom terakhir dengan fungsi iloc
df = df.drop(df.iloc[:,2:], axis=1)

# Cek data
df.head()
```

{% endcode %}

| Text | v1   | v2                                                |
| ---- | ---- | ------------------------------------------------- |
| 0    | ham  | Go until jurong point, crazy.. Available only ... |
| 1    | ham  | Ok lar... Joking wif u oni...                     |
| 2    | spam | Free entry in 2 a wkly comp to win FA Cup fina... |
| 3    | ham  | U dun say so early hor... U c already then say... |
| 4    | ham  | Nah I don't think he goes to usf, he lives aro... |

### Langkah 2b - Inspeksi Data

{% code lineNumbers="true" %}

```python
# Cek Jumlah Data Per Kelas
print(df['Labels'].value_counts())
print('\n')

# Cek Kelengkapan Data
print(df.info())
print('\n')

# Cek Statistik Deskriptif
print(df.describe())
```

{% endcode %}

Hasilnya,

```
ham     4825
spam     747
Name: Labels, dtype: int64


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Labels  5572 non-null   object
 1   SMS     5572 non-null   object
dtypes: object(2)
memory usage: 87.2+ KB
None


       Labels                     SMS
count    5572                    5572
unique      2                    5169
top       ham  Sorry, I'll call later
freq     4825                      30
```

### Langkah 2c - Encoding Label

{% code lineNumbers="true" %}

```python
# Data untuk label
new_labels = {
    'spam': 1,
    'ham': 0
}

# Encode label
df['Labels'] = df['Labels'].map(new_labels)

# Cek data
df.head()
```

{% endcode %}

| Text | Labels | SMS                                               |
| ---- | ------ | ------------------------------------------------- |
| 0    | 0      | Go until jurong point, crazy.. Available only ... |
| 1    | 0      | Ok lar... Joking wif u oni...                     |
| 2    | 1      | Free entry in 2 a wkly comp to win FA Cup fina... |
| 3    | 0      | U dun say so early hor... U c already then say... |
| 4    | 0      | Nah I don't think he goes to usf, he lives aro... |

### Langkah 2d - Pisahkan Fitur dengan Label

{% code lineNumbers="true" %}

```python
X = df['SMS'].values
y = df['Labels'].values
```

{% endcode %}

## Langkah 3 - Ekstraksi Fitur

Ekstraksi fitur untuk setiap SMS akan menggunakan konsep Bag of Words. Kita dapat menggunakan fungsi `CountVectorizer` dari scikit-learn. Akan tetapi untuk mencegah **leaking information** kita akan melakukan split data terlebih dahulu, baru melakukan transformasi terhadap data training dan testing.

{% code lineNumbers="true" %}

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Inisiasi CountVectorizer
bow = CountVectorizer()

# Fitting dan transform X_train dengan CountVectorizer
X_train = bow.fit_transform(X_train)

# Transform X_test
# Mengapa hanya transform? Alasan yang sama dengan kasus pada percobaan ke-3
# Kita tidak menginginkan model mengetahui paramter yang digunakan oleh CountVectorizer untuk fitting data X_train
# Sehingga, data testing dapat tetap menjadi data yang asing bagi model nantinya
X_test = bow.transform(X_test)
```

{% endcode %}

Cek fitur dari proses `CountVectorizer`.

{% code lineNumbers="true" %}

```python
print(len(bow.get_feature_names()))
print(f'Dimensi data: {X_train.shape}')
```

{% endcode %}

## Langkah 4 - Training dan Evaluasi Model

Kita akan menggunakan algoritma Multinomial Naive Bayes. Fungsi `MultinomialNB` dari scikit-learn dapat digunakan pada kasus ini.

{% code lineNumbers="true" %}

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Inisiasi MultinomialNB
mnb = MultinomialNB()

# Fit model
mnb.fit(X_train, y_train)

# Prediksi dengan data training
y_pred_train = mnb.predict(X_train)

# Evaluasi akurasi data training
acc_train = accuracy_score(y_train, y_pred_train)

# Prediksi dengan data training
y_pred_test = mnb.predict(X_test)

# Evaluasi akurasi data training
acc_test = accuracy_score(y_test, y_pred_test)

# Print hasil evaluasi
print(f'Hasil akurasi data train: {acc_train}')
print(f'Hasil akurasi data test: {acc_test}')py
```

{% endcode %}

Hasilnya,

```
Hasil akurasi data train: 0.9946152120260264
Hasil akurasi data test: 0.9775784753363229
```







# Tugas 1

{% file src="<https://3041032130-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F5CvtE8Xh9b75jKUaRr5Y%2Fuploads%2Fv7RHnCSAbQRjwJX7o36o%2Fvoice.csv?alt=media&token=d5bad3d3-e5c4-4515-803c-bb4ea198fda9>" %}

1. Buatlah model klasifikasi dengan menggunakan kNN untuk mengklasifikasikan jenis suara `male` dan `female` pada dataset `voice.csv`.
2. Lakukan percobaan untuk mengetahui fitur-fitur yang paling optimal untuk digunakan. Fitur apa saja yang Anda gunakan untuk mendapatkan hasil terbaik?
3. Berdasarkan fitur yang telah Anda pilih pada soal nomor 2, berapa nilai  yang terbaik? Lampirkan grafika analisis dan alasan Anda.








# Tugas 2

* Buatlah model klasfikasi Multinomial Naive Bayes dengan ketentuan,
  1. Menggunakan data `spam.csv`
  2. Fitur `CountVectorizer` dengan mengaktifkan **stop\_words**
  3. Evaluasi hasilnya
* Buatlah model klasfikasi Multinomial Naive Bayes dengan ketentuan,
  1. Menggunakan data `spam.csv`
  2. Fitur `TF-IDF` dengan mengaktifkan **stop\_words**
  3. Evaluasi hasilnya dan bandingkan dengan hasil pada Tugas no 2.
  4. Berikan kesimpulan fitur mana yang terbaik pada kasus data `spam.csv`

