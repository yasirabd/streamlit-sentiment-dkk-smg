import streamlit as st
from PIL import Image


def display_modeling():
    text = """
    # Modeling
    Tahap ini melakukan training model *machine learning* dari masukan data yang sudah dilakukan 
    fitur ekstraksi.
    """
    st.markdown(text, unsafe_allow_html=True)
    st.info("Data untuk training yang digunakan sebanyak `7967 data` dan data untuk testing sebanyak `1992 data`")

    text = """
    Terdapat beberapa metode *machine learning* yang digunakan untuk pelatihan data yaitu `Naive Bayes`, `Support Vector Machine (SVM)`,
    `Random Forest`, `Convolutional Neural Network (CNN)`, dan `Long Short-Term Memory (LSTM)`. Hasil akurasi terbaik dirangkum pada tabel
    berikut.
    """
    st.markdown(text, unsafe_allow_html=True)


    text = """
    Pembahasan masing-masing metode dijelaskan di bawah.

    ## Naive Bayes

    ### Deskripsi
    Metode Naïve Bayes adalah metode yang paling populer digunakan dalam pengklasifikasian [1]. 
    Naïve Bayes Classifier merupakan sebuah metode klasifikasi dengan probabilitas sederhana yang mengaplikasikan 
    Teorema Bayes dengan asumsi ketidaktergantungan (independen) yang tinggi.[3]

    Kelebihan
    - Tidak membutuhkan data training yang banyak untuk memperkirakan parameter [2]
    - Dalam penelitian [3] diungkapkan bahwa metode Naïve Bayes Classifier memiliki beberapa kelebihan antara lain, 
    sederhana, cepat dan berakurasi tinggi

    Kekurangan
    - Sangat sensitif pada fitur yang terlalu banyak, sehingga membuat akurasi menjadi rendah [4]
    - Harus mengasumsi bahwa antar fitur tidak terkait (*independent*) [5]

    ---

    1. Natalius, Samuel., 2011, Metoda Naïve Bayes Classifier dan Penggunaannya pada Klasifikasi Dokumen.
    2. Stern, M., Beck, J., & Woolf, B. P..2009. Applying Naive Bayes Data Mining Technique for Classification of Agricultural Land Soils
    3. Nurhuda, F., Sihwi, S. W., & Doewes, A. (2013). Analisis sentimen masyarakat terhadap calon presiden Indonesia 2014 berdasarkan opini dari Twitter menggunakan metode Naive Bayes classifier. 
    4. Utami, L. D., & Wahono, R. S. (2015). Integrasi metode information gain untuk seleksi fitur dan adaboost untuk mengurangi bias pada analisis sentimen review restoran menggunakan algoritma naive bayes. 
    5. Pamungkas, D. S., Setiyanto, N. A., & Dolphina, E. (2015). Analisis sentiment pada sosial media twitter menggunakan naive bayes classifier terhadap kata kunci “kurikulum 2013”. 


    ### Modeling dengan Naive Bayes
    Model Naive Bayes dilakukan training dengan data input berbeda, yaitu hasil feature extraction 
    menggunakan TF-IDF dan hasil feature extraction menggunakan Word2Vec.

    Untuk meningkatkan performa model, dilakukan *hyperparameter tuning* menggunakan `GridSearchCV`. 
    Berikut ini adalah tabel *hyperparameter*.

    | Hyperparameter  | Kombinasi        |
    |-----------------|------------------|
    | `var_smoothing` | `range(1, 1e-9)` |
    <br>

    ##### Naive Bayes dengan TF-IDF
    Hasil kombinasi terbaik untuk model Naive Bayes dengan masukan data hasil feature extraction TF-IDF sebagai berikut.

    | Hyperparameter  | Kombinasi Terbaik     |
    |-----------------|-----------------------|
    | `var_smoothing` | `0.1873817422860384`  |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **meningkat 14.4%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score Naive Bayes di bawah. 

    ##### Naive Bayes dengan Word2Vec
    Hasil kombinasi terbaik untuk model Naive Bayes dengan masukan data hasil feature extraction Wor2Vec sebagai berikut.

    | Hyperparameter  | Kombinasi Terbaik        |
    |-----------------|--------------------------|
    | `var_smoothing` | `0.0006579332246575676`  |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **menurun 0.1%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score Naive Bayes di bawah. 
    """
    st.markdown(text, unsafe_allow_html=True)
    nb_acc = Image.open('images/Naive Bayes Accuracy.png')
    st.image(nb_acc, caption='Tabel Accuracy Score Naive Bayes', width=400)

    text = """
    ### Pembahasan
    Berdasarkan hasil modeling menggunakan Naive Bayes, diketahui:
    - Model terbaik adalah **model TF-IDF Naive Bayes** setelah dilakukan *tuning* dengan akurasi pada test **69.68%**.
    - Model dengan feature extraction TF-IDF memiliki akurasi lebih tinggi dari Word2Vec karena banyaknya kata-kata 
    yang tidak terdapat vector-nya pada model Word2Vec. Misalkan: kata bahasa jawa; engko, mlebu, dan ngeles.
    - Model TF-IDF Naive Bayes mendapatkan akurasi tertinggi dengan melakukan *tuning* parameter `var_smoothing` 
    dengan nilai terbaik `0.1873817422860384`.
    """
    st.markdown(text, unsafe_allow_html=True)

    text = """
    ## Support Vector Machine
    Support Vector Machine (SVM) merupakan salah satu metode *machine learning* populer yang termasuk 
    *supervised learning*. Metode SVM populer untuk *classification, regression*, dan *outliers detection*.[1]

    ### Deskripsi
    Berdasarkan dokumentasi scikit-learn, SVM memiliki kelebihan dan kelemahan sebagai berikut:

    Kelebihan dari SVM:
    - Efektif untuk *high dimensional spaces*.
    - Tetap efektif pada kasus dimana jumlah dimensi lebih besar dari jumlah sampel.
    - *Memory efficient*, karena menggunakan subset dari *training points* pada *decision function* / *(support vectors)*.
    - *Versatile*, bisa menggunakan fungsi kernel berbeda seperti: Linear, Polynomial, dan Radial.

    Adapun kekurangan SVM:
    - Ketika jumlah fitur lebih banyak daripada jumlah sampel, maka penting untuk menghindari *over-fitting* perlu
    dilakukan pemilihan *kernel function* dan *regularization*.
    - SVM tidak mengakomodir *probability estimate*, sehingga proses ini menggunakan 5-fold cross-validation.

    Metode SVM mencari *hyperplane* terbaik untuk penentuan kelas data.
    """
    st.markdown(text, unsafe_allow_html=True)

    svm = Image.open('images/svm.png')
    st.image(svm, caption='SVM, gambar dari Wikipedia', width=300)
    
    text = """
    Berdasarkan gambar di atas, model SVM digambarkan dengan garis berwarna merah. Salah satu kelebihan
    SVM mampu melakukan *soft margin classification*, yaitu menjaga keseimbangan antara menjaga wilayah dalam garis 
    titik-titik tetap lebar dan membatasi jumlah data yang melewati garis model. Hal ini yang menjadikan SVM 
    sebagai metode yang fleksibel.

    Metode SVM juga memiliki performa yang bagus untuk *sentiment analysis* bahasa Indonesia pada kasus sentiment analysis
    untuk mengetahui kepuasan pelanggan terhadap pelayanan warung dan restoran kuliner kota Tegal [2], sentiment analysis
    evaluasi dosen [3], dan Twitter sentiment analysis untuk review film. [4]

    ---

    1. Chang and Lin, LIBSVM: A Library for Support Vector Machines.
    2. Somantri and Apriliani, Support Vector Machine Berbasis Feature Selection Untuk Sentiment Analysis Kepuasan Pelanggan Terhadap Pelayanan Warung dan Restoran Kuliner Kota Tegal. DOI: 10.25126/jtiik.201855867.
    3. Santoso, Valonia, et al., Penerapan Sentiment Analysis pada Hasil Evaluasi Dosen dengan Metode Support Vector Machine. DOI: 10.26623/transformatika.v14i2.439
    4. Rahutomo, Faisal, et al., Implementasi Twitter Sentiment Analysis untuk Review Film Menggunakan Algoritma Support Vector Machine.

    
    ### Modeling dengan SVM
    Pada research ini, dilakukan training model SVM dengan data input berbeda, yaitu hasil feature extraction 
    menggunakan TF-IDF dan hasil feature extraction menggunakan Word2Vec.

    Untuk meningkatkan performa model, dilakukan *hyperparameter tuning* menggunakan `GridSearchCV`. 
    Berikut ini adalah tabel *hyperparameter*.

    | Hyperparameter            | Kombinasi                |
    |---------------------------|--------------------------|
    | `kernel`                  | `[linear, rbf, poly]`    |
    | `C`                       | `[1, 0.25, 0.5, 0.75]`   |
    | `gamma`                   | `[scale, auto, 1, 2, 3]` |
    | `decision_function_shape` | `[ovo, ovr]`             |
    <br>

    ##### SVM dengan TF-IDF
    Hasil kombinasi terbaik untuk model SVM dengan masukan data hasil feature extraction TF-IDF sebagai berikut.

    | Hyperparameter            | Kombinasi Terbaik |
    |---------------------------|-------------------|
    | `kernel`                  | `rbf`             |
    | `C`                       | `1`               |
    | `gamma`                   | `1`               |
    | `decision_function_shape` | `ovo`             |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **meningkat 0.1%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score SVM di bawah. 

    ##### SVM dengan Word2Vec
    Hasil kombinasi terbaik untuk model SVM dengan masukan data hasil feature extraction Wor2Vec sebagai berikut.

    | Hyperparameter            | Kombinasi Terbaik |
    |---------------------------|-------------------|
    | `kernel`                  | `rbf`             |
    | `C`                       | `1`               |
    | `gamma`                   | `3`               |
    | `decision_function_shape` | `ovo`             |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **menurun 0.6%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score SVM di bawah.
    """
    st.markdown(text, unsafe_allow_html=True)
    svm_acc = Image.open('images/SVM Accuracy.png')
    st.image(svm_acc, caption='Tabel Accuracy Score SVM', width=400)
    st.error("Model Word2Vec SVM setelah tuning tidak dihitung accuracy score untuk data training 5-fold karena proses sangat lama.")

    text = """
    ### Pembahasan
    Berdasarkan hasil training dan testing model, dapat kita ketahui: 
    - Model terbaik adalah **model TF-IDF SVM** setelah dilakukan *tuning hyperparameter* dengan **akurasi 74.6%**.
    - Model dengan *feature extraction* TF-IDF memiliki akurasi lebih tinggi dari Word2Vec karena banyaknya kata-kata yang tidak 
    terdapat vector-nya pada model Word2Vec. Misalkan: kata bahasa jawa; engko, mlebu, dan ngeles.
    - Hasil akurasi pada model TF-IDF SVM setelah *tuning* lebih rendah 0.1% karena dilakukan pembagian fold data secara acak.

    ## Random Forest

    ### Deksripsi
    Random Forest termasuk sebagai *ensemble method*, yaitu menggabungkan beberapa hasil prediksi dari beberapa *estimator* menggunakan
    algoritma/model tertentu. Tujuan metode ini adalah untuk meningkatkan *generalizability / robustness* dibandingkan dengan satu
    *estimator* saja. [1]

    Random Forest menggunakan pendekatan *averaging method*, yaitu dengan membuat beberapa *estimator* secara independen kemudian hasil
    prediksinya dilakukan perhitungan rata-rata. Hasil rata-rata dari gabungan beberapa *estimator* memiliki hasil lebih baik daripada
    satu *estimator* karena *variance* berkurang. [2]

    Akan tetapi, implementasi pada `scikit-learn` yang digunakan berbeda dengan original paper. Perbedaanya pada `scikit-learn` melakukan
    implementasi dengan beberapa *classifier* dengan menghitung rata-rata dari tiap hasil *probabilistic prediction*. Sedangkan pada
    paper setiap *classifier* melakukan *vote* untuk sebuah kelas. [3]

    Metode Random Forest memiliki performa bagus untuk kasus sentiment analysis pada data komentar YouTube [4], sentiment analysis pada 
    data Twitter [5], dan sentiment analysis pada data review film. [6]

    ---

    1. Dietterich, Thomas G., Ensemble Methods in Machine Learning.
    2. Breiman, Leo, Random Forests, Machine Learning, 45(1), 5-32, 2001.
    3. Scikit-learn documentation
    4. Khomsah, Siti, Sentiment Analysis On YouTube Comments Using Word2Vec and Random Forest.
    5. Bahrawi, Nfn, Sentiment Analysis Using Random Forest Algorithm-Online Social Media Based.
    6. Zamzami, Firdausi N., et al. Analisis Sentimen Terhadap Review Film Menggunakan Metode Modified Balanced Random Forest dan Mutual Information

    ### Modeling dengan Random Forest
    Pada research ini, dilakukan training model Random Forest dengan data input berbeda, yaitu hasil feature extraction 
    menggunakan TF-IDF dan hasil feature extraction menggunakan Word2Vec.

    Untuk meningkatkan performa model, dilakukan *hyperparameter tuning* menggunakan `GridSearchCV`. 
    Berikut ini adalah tabel *hyperparameter*.

    | Hyperparameter            | Kombinasi                  |
    |---------------------------|----------------------------|
    | `n_estimators`            | `[800, 1200, 1500, 1800]`  |
    | `min_samples_leaf`        | `[1, 2, 3]`                |
    | `min_samples_split`       | `[8, 10, 12]`              |
    | `bootstrap`               | `[True, False]`            |
    | `criterion`               | `[gini, entropy]`          |
    <br>

    ##### Random Forest dengan TF-IDF
    Hasil kombinasi terbaik untuk model Random Forest dengan masukan data hasil feature extraction TF-IDF sebagai berikut.

    | Hyperparameter            | Kombinasi Terbaik |
    |---------------------------|-------------------|
    | `n_estimators`            | `1800`            |
    | `min_samples_leaf`        | `1`               |
    | `min_samples_split`       | `12`              |
    | `bootstrap`               | `True`            |
    | `criterion`               | `gini`            |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **meningkat 0.6%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score Random Forest di bawah. 

    ##### Random Forest dengan Word2Vec
    Hasil kombinasi terbaik untuk model Random Forest dengan masukan data hasil feature extraction Wor2Vec sebagai berikut.

    | Hyperparameter            | Kombinasi Terbaik |
    |---------------------------|-------------------|
    | `n_estimators`            | `[800, 1200, 1500, 1800]`  |
    | `min_samples_leaf`        | `[1, 2, 3]`                |
    | `min_samples_split`       | `[8, 10, 12]`              |
    | `bootstrap`               | `[True, False]`            |
    | `criterion`               | `[gini, entropy]`          |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **menurun 0.6%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score Random Forest di bawah.

    ### Pembahasan
    """
    st.markdown(text, unsafe_allow_html=True)

    text = """
    ## Convolutional Neural Network
    *Convolutional Neural Network* (CNN) adalah salah satu jenis neural network yang biasa digunakan pada data 
    dua dimensi. CNN bisa digunakan untuk mendeteksi dan mengenali objek. Penggunanya tidak terbatas pada gambar, 
    namun bisa juga digunakan untuk memecahkan masalah dalam *natural language processing* dan *speech recognition*. 
    CNN termasuk dalam jenis Deep Neural Network karena dalamnya tingkat jaringan dan banyak diimplementasikan 
    dalam data citra. CNN memiliki dua metode; yakni klasifikasi menggunakan *feedforward* dan tahap pembelajaran 
    menggunakan *backpropagation*.

    ### Deskripsi
    CNN membentuk neuron-neuronnya ke dalam tiga dimensi (panjang, lebar, dan tinggi) dalam sebuah lapisan.

    1. Feature Learning (Convolutional Layer, Rectified Linear Unit, Pooling Layer).
    2. Classification (Flatten, Fully-connected, Softmax).

    Kelebihan dari CNN yang menggunakan dimensi > 1 akan memengaruhi keseluruhan skala dalam suatu objek. 
    CNN memiliki kemampuan mengolah informasi citra. Namun CNN, seperti metode Deep Learning lainnya, memiliki 
    kelemahan yaitu proses traning model yang lama dan relatif mahal. 


    ### Modeling dengan CNN
    Pada research ini, dilakukan training model CNN dengan data input berbeda, yaitu hasil feature extraction 
    menggunakan TF-IDF dan hasil feature extraction menggunakan Word2Vec.

    Untuk meningkatkan performa model, dilakukan *hyperparameter tuning* menggunakan `KerasTuner`. 
    Berikut ini adalah tabel *hyperparameter*.

    | Hyperparameter            | Kombinasi                                     |
    |---------------------------|-----------------------------------------------|
    | `Conv1D_filters`          | `range(min_value=64, max_value=512, step=64)` |
    | `Conv1D_kernel_size`      | `[4, 8, 16]`                                  |
    | `Dense1_units`            | `range(min_value=32, max_value=128, step=32)` |
    | `Dense2_units`            | `range(min_value=16, max_value=64, step=16)`  |
    | `Adam_learning_rate`      | `[1e-2, 1e-3, 1e-4, 1e-5]`                    |
    <br>

    ##### CNN dengan TF-IDF
    Hasil kombinasi terbaik untuk model CNN dengan masukan data hasil feature extraction TF-IDF sebagai berikut.

    | Hyperparameter            | Kombinasi Terbaik |
    |---------------------------|-------------------|
    | `Conv1D_filters`          | `320`             |
    | `Conv1D_kernel_size`      | `16`              |
    | `Dense1_units`            | `32`              |
    | `Dense2_units`            | `32`              |
    | `Adam_learning_rate`      | `1e-3`            |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **meningkat 5.52%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score CNN di bawah.

    Architecture dari model TF-IDF CNN terbaik adalah sebagai berikut.
    """
    st.markdown(text, unsafe_allow_html=True)
    tfidf_cnn_architecture = Image.open('images/TF-IDF CNN Architecture.png')
    st.image(tfidf_cnn_architecture, caption='Arsitektur TF-IDF CNN Terbaik', width=400)

    text = """
    ##### CNN dengan Word2Vec
    Hasil kombinasi terbaik untuk model CNN dengan masukan data hasil feature extraction Wor2Vec sebagai berikut.

    | Hyperparameter            | Kombinasi Terbaik |
    |---------------------------|-------------------|
    | `Conv1D_filters`          | `448`             |
    | `Conv1D_kernel_size`      | `16`              |
    | `Dense1_units`            | `64`              |
    | `Dense2_units`            | `32`              |
    | `Adam_learning_rate`      | `1e-4`            |
    <br>

    Setelah dilakukan *tuning hyperparameter*, nilai akurasi **menurun 2.26%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score CNN di bawah.

    Architecture dari model TF-IDF CNN terbaik adalah sebagai berikut.
    """
    st.markdown(text, unsafe_allow_html=True)
    word2vec_cnn_architecture = Image.open('images/Word2Vec CNN Architecture.png')
    st.image(word2vec_cnn_architecture, caption='Arsitektur Word2Vec CNN Terbaik', width=400)

    text = """
    Berikut ini adalah tabel Accuracy Score CNN.
    """
    st.markdown(text, unsafe_allow_html=True)
    cnn_acc = Image.open('images/CNN Accuracy.png')
    st.image(cnn_acc, caption='Accuracy Score CNN', width=400)

    text = """
    ### Pembahasan
    Berdasarkan hasil training dan testing model, dapat kita ketahui: 
    - Model terbaik adalah **model Word2Vec CNN** setelah dilakukan *tuning hyperparameter* dengan **akurasi 69.33%** pada data test.
    - Model dengan *feature extraction* Word2Vec memiliki hasil akurasi lebih baik dari TF-IDF karena model CNN memiliki arsitektur
    yang mendukung 3 dimensi. Selain itu, hasil feature extraction TF-IDF berukuran 2-dimensi, sehingga harus dilakukan reshape 
    menjadi 3-dimensi yang menyebabkan model CNN tidak memiliki performa yang baik.
    - Hasil akurasi pada model Wor2Vec CNN sebelum *tuning* lebih tinggi dari setelah *tuning* karena dilakukan pengacakan data pada
    proses `k-fold`. 
    """
    st.markdown(text, unsafe_allow_html=True)

    text = """
    ## Long Short-Term Memory

    ### Deskripsi
    Long Short-Term Memory (LSTM) merupakan model varian dari Recurrent Neural Network (RNN) yang dapat mengingat informasi jangka 
    panjang (*long term dependency*). Model LSTM dirancang sebagai solusi dari masalah *vanishing gradient* yang ditemui pada RNN 
    konvensional [1]. Dalam LSTM terdapat tiga gerbang yaitu *input gate*, *forget gate*, dan *output gate*. Sel memori dan tiga gerbang 
    dirancang untuk dapat membaca, menyimpan, dan memperbarui informasi terdahulu.
    """
    st.markdown(text, unsafe_allow_html=True)
    lstm = Image.open('images/LSTM3-chain.png')
    st.image(lstm, caption='Struktur LSTM, source dari colah.github.io', width=600)

    text = """
    Kelebihan:
    - LSTM tepat digunakan untuk data yang berbentuk *sequence*.
    - LSTM memiliki *performance* yang lebih baik pada *sentiment classification* ketika jumlah training data lebih banyak [2].
    - LSTM lebih baik dalam mempelajari *context-sensitive* daripada RNN [3].

    Kekurangan:
    - LSTM membutuhkan waktu lebih lama dan lebih banyak memori untuk dilatih.
    - Dropout jauh lebih sulit untuk diterapkan di LSTM.
    - LSTM sensitif terhadap inisialisasi bobot acak yang berbeda.

    ---

    Referensi:

    1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. 
    2. Murthy, D., Allu, S., Andhavarapu, B., Bagadi, M., & Belusont, M. (2020). Text based sentiment analysis using LSTM. 
    3. Gers, F. A., & Schmidhuber, J. (2001). LSTM recurrent networks learn simple context free and context sensitive languages.

    ### Modeling dengan LSTM
    Pada research ini, dilakukan training model LSTM dengan input data berbeda, yaitu hasil feature extraction menggunakan 
    TF-IDF dan hasil feature extraction menggunakan Word2Vec.

    Untuk meningkatkan performa model, dilakukan hyperparameter tuning menggunakan `Keras Tuner` untuk mendapatkan nilai 
    *hyperparameter* pada model. Berikut ini adalah tabel *hyperparameter*.

    | Hyperparameter   | Kombinasi            |
    |------------------|----------------------|
    | `lstm_out`       | `[32, 48, 64]`       |
    | `dropout`        | `[0.2, 0.5, 0.8]`    |
    | `learning_rate`  | `[0.1, 0.01, 0.001]` |
    
    ##### LSTM dengan TF-IDF
    Hasil kombinasi terbaik untuk model LSTM dengan masukan data hasil feature extraction TF-IDF sebagai berikut.

    | Hyperparameter   | Kombinasi Terbaik |
    |------------------|-------------------|
    | `lstm_out`       | `32`              |
    | `dropout`        | `0.8`             |
    | `learning_rate`  | `0.001`           |
    <br>

    Setelah dilakukan *tuning*, nilai akurasi **meningkat 2.41%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score LSTM di bawah.

    ##### LSTM dengan Word2Vec
    Hasil kombinasi terbaik untuk model LSTM dengan masukan data hasil feature extraction Word2Vec sebagai berikut.

    | Hyperparameter   | Kombinasi Terbaik |
    |------------------|-------------------|
    | `lstm_out`       | `48`              |
    | `dropout`        | `0.2`             |
    | `learning_rate`  | `0.01`            |
    <br>

    Setelah dilakukan *tuning*, nilai akurasi **meningkat 1.11%** pada data test. Hasil akurasi skor dapat 
    dilihat pada tabel Accuracy Score LSTM di bawah.
    """
    st.markdown(text, unsafe_allow_html=True)
    lstm_acc = Image.open('images/LSTM Accuracy.png')
    st.image(lstm_acc, caption='Accuracy Score LSTM', width=400)

    text = """
    ### Pembahasan
    Berdasarkan hasil training dan testing model, dapat kita ketahui: 
    - Model terbaik adalah **model TF-IDF LSTM** setelah dilakukan *tuning hyperparameter* dengan **akurasi 71.94%** pada data test.
    - Model dengan *feature extraction* TF-IDF memiliki hasil akurasi lebih baik dari Word2Vec karena terdapat beberapa kata yang 
    *out-of-vocabulary* (oov) dari *vocabulary pre-trained* FastText. 
    - Hasil akurasi pada model TF-IDF LSTM dan Word2Vec LSTM memiliki akurasi cukup baik sekitar 70%, menunjukkan LSTM mampu melakukan 
    analisis sentimen dengan cukup baik. 
    """
    st.markdown(text, unsafe_allow_html=True)