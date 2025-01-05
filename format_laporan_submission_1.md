# Laporan Proyek Machine Learning - Randa Andriana Putra

## Domain Proyek

Memahami tingkat stres di kalangan siswa sangat penting karena dampaknya yang signifikan terhadap kesehatan mental, kinerja akademik, dan interaksi sosial. Stres yang tinggi dapat menyebabkan efek merugikan pada kesejahteraan keseluruhan siswa, mempengaruhi kemampuan mereka untuk berkonsentrasi dan berhasil secara akademis. Penelitian menunjukkan bahwa sebagian besar siswa mengalami tingkat stres yang tinggi, dengan studi menunjukkan bahwa sekitar 60% siswa di beberapa wilayah melaporkan stres tinggi akibat tekanan akademik.

**Rubrik/Kriteria Tambahan (Opsional)**: 
Stres siswa harus dianggap serius karena dapat mengakibatkan konsekuensi jangka panjang seperti kelelahan, penurunan kualitas hidup, dan perkembangan gangguan kesehatan mental. Stres kronis dapat mengganggu keterlibatan akademik dan meningkatkan risiko masalah psikologis seperti kecemasan dan depresi. Institusi pendidikan dan keluarga sering menghadapi tantangan dalam mengidentifikasi dan menangani faktor-faktor penyebab stres ini, yang dapat mencakup tuntutan akademis, harapan keluarga, dan tekanan sosial.

Teknologi dan analisis data memainkan peran penting dalam memahami pola stres di kalangan siswa. Pendekatan prediktif menggunakan pembelajaran mesin dapat membantu mengantisipasi stres sebelum berkembang menjadi masalah yang lebih serius. Dengan menganalisis dataset yang relevan, peneliti dapat mengidentifikasi faktor-faktor kunci yang berkontribusi terhadap stres siswa dan mengembangkan intervensi yang tepat.

Format Referensi: [Student stress and quality of education](https://www.researchgate.net/publication/262748005_Student_stress_and_quality_of_education)

## Business Understanding

### Problem Statements

Pernyataan masalah latar belakang:
- Bagaimana faktor psikologis, fisiologis, sosial, lingkungan, dan akademik memengaruhi tingkat stres siswa?
- Bisakah model pembelajaran mesin secara akurat memprediksi tingkat stres siswa berdasarkan faktor-faktor multidimensional ini?
- Apa saja fitur, faktor atau variabel paling signifikan yang berkontribusi terhadap stres siswa?

### Goals

Tujuan dari pernyataan masalah:
- Menganalisis dan mengidentifikasi faktor-faktor signifikan yang memengaruhi stres siswa.
- Mengembangkan model pembelajaran mesin yang mampu memprediksi tingkat stres siswa dengan akurasi tinggi.
- Mendukung strategi pencegahan dan intervensi melalui wawasan praktis yang diperoleh dari model.

**Rubrik/Kriteria Tambahan (Opsional)**: 
### Solution statements
Solusi yang pertama adalah dengan menggunakan algoritma Random Forest. Dengan metrik evaluasinya adalah Akurasi, Precision, Recall, dan F1-Score untuk menilai performa model yang dibuat.
Solusi yang kedua adalah dengan menggunakan algoritma XG

## Data Understanding
Dataset ini berisi data mengenai faktor-faktor yang memengaruhi stres siswa.
Sumber Dataset: [Student Stress Factors: A Comprehensive Analysis](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis).

### Fitur-fitur tersebut dipilih secara ilmiah dengan mempertimbangkan 5 faktor utama, yaitu:
Faktor Psikologis:
- anxiety_level: Merupakan tingkat kecemasan yang dirasakan siswa. Tingkat kecemasan yang tinggi biasanya menjadi indikator stres yang signifikan.
- self_esteem: Merupakan tingkat kepercayaan diri siswa terhadap kemampuan mereka. Kepercayaan diri rendah sering dikaitkan dengan stres akademik atau sosial.
- mental_health_history: Merupakan riwayat kesehatan mental siswa, seperti pernah mengalami gangguan mental atau sedang menjalani terapi.
- depression: Merupakan indikator apakah siswa menunjukkan gejala depresi, yang merupakan faktor utama dalam kondisi stres kronis.
Faktor Fisiologis:
- headache: Frekuensi sakit kepala yang dialami siswa, yang sering menjadi tanda stres fisik atau mental.
- blood_pressure: Tingkat tekanan darah siswa, di mana tekanan darah tinggi bisa menjadi indikasi stres.
- sleep_quality: Kualitas tidur siswa, seperti durasi tidur atau gangguan tidur. Kurang tidur sangat memengaruhi kemampuan mengelola stres.
- breathing_problem: Masalah pernapasan yang mungkin dialami siswa, yang dapat memburuk karena stres atau kecemasan.
Faktor Lingkungan:
- noise_level : Tingkat kebisingan di sekitar tempat tinggal siswa, yang dapat mengganggu konsentrasi dan tidur.
- living_conditions : Kondisi tempat tinggal, seperti ukuran, kenyamanan, dan fasilitas yang tersedia.
- safety : Persepsi siswa terhadap keamanan di lingkungan mereka. Rasa tidak aman dapat memicu stres berkelanjutan.
- basic_needs : Akses siswa terhadap kebutuhan dasar seperti makanan, air, dan fasilitas kesehatan.
Faktor Akademik:
- academic_performance: Prestasi akademik siswa, seperti nilai atau peringkat kelas. Kinerja buruk dapat meningkatkan tekanan emosional.
- study_load: Beban belajar siswa, termasuk jumlah tugas, jadwal ujian, dan jam belajar.
- teacher_student_relationship: Hubungan siswa dengan guru, yang dapat memengaruhi kenyamanan belajar.
- future_career_concerns: Kekhawatiran siswa terhadap masa depan karir mereka, terutama dalam membuat keputusan besar.
Faktor Sosial:
- social_support: Tingkat dukungan yang diterima siswa dari keluarga, teman, atau komunitas. Dukungan sosial rendah dapat meningkatkan kerentanan terhadap stres.
- peer_pressure: Tekanan yang dirasakan siswa dari teman sebaya, misalnya untuk mengikuti norma kelompok.
- extracurricular_activities: Partisipasi siswa dalam kegiatan ekstrakurikuler, yang dapat membantu atau justru menambah beban.
- bullying: Pengalaman siswa terhadap perundungan, baik secara fisik, verbal, maupun online.

**Rubrik/Kriteria Tambahan (Opsional)**:
Visualisasi heatmap untuk melihat korelasi antar fitur:
![image](https://github.com/user-attachments/assets/c4e018ed-2f0e-4057-b7f4-865ab29517a8)

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

Penanganan Missing Values, bertujuan untuk meningkatkan kualitas data karena jika terdapat nilai yang hilang dapat menyebabkan error dalam perhitungan atau algoritma machine learning.
Data Cleaning Missing Values
df.isnull().sum()
![image](https://github.com/user-attachments/assets/ff1117af-5cc2-4e6e-86d1-b75c7fdbb2e2)
Data sudah tidak ada missing values

Penanganan Imbalanced Data, bertujuan untuk memeriksa apakah terdapat target (stress_level) yang tidak seimbang. Dikarenakan ketidakseimbangan data dapat membuat model lebih condong memprediksi kelas mayoritas karena jumlahnya dominan. Dengan distribusi yang lebih seimbang, model dapat memahami pola kelas minoritas lebih baik dan menghasilkan prediksi yang lebih akurat pada semua kelas.
Deteksi imbalanced data fitur stress_level:
df['stress_level'].value_counts()
![image](https://github.com/user-attachments/assets/b41792ec-9cea-4c5c-9ef6-da1a56d74193)
Dataset terbilang balanced


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

