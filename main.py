import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import pyaudio
import wave


# Ses kaydı fonksiyonu
def record_audio(file_path, record_duration=600):  # 600 saniye (10 dakika)
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    seconds = record_duration

    p = pyaudio.PyAudio()

    print("Ses kaydediliyor...")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Ses kaydedildi ve dosyaya yazıldı:", file_path)


# Ses dosyasını yükleme ve özellikleri çıkarma
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=None)  # Örnekleme oranı için sr parametresi kullanılmıyor
    mfccs = librosa.feature.mfcc(y=audio, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


# Örnek veri dosyalarının yolları
working_machine_file = "çalışan_makine_sesi.wav"
not_working_machine_file = "çalışmayan_makine_sesi.wav"

# Örnek veri özellikleri
working_machine_features = extract_features(working_machine_file)
not_working_machine_features = extract_features(not_working_machine_file)

# Etiketleme
X = np.vstack((working_machine_features, not_working_machine_features))
y = np.array(['Çalışıyor', 'Çalışmıyor'])

# Modeli oluşturma
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Veriyi yeniden şekillendirme
X = X.reshape(X.shape[0], X.shape[1], 1)

# Modeli eğitme
model.fit(X, y, epochs=10, batch_size=8)

# Ses kaydetme
recorded_file = "kaydedilen_makine_sesi.wav"
record_audio(recorded_file)

# Test verisi için tahmin yapma
test_features = extract_features(recorded_file)
test_features = test_features.reshape(1, test_features.shape[0], 1)
prediction = model.predict(test_features)

# Tahmin sonucunu gösterme
if prediction[0][0] > 0.5:
    print("Makine çalışıyor.")
else:
    print("Makine çalışmıyor.")
