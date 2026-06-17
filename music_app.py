import streamlit as st
st.set_page_config(page_title="Sound Sense", page_icon=":musical_note:")

st.title("Sound Sense")
st.header("Instrument Classification System")

import pandas as pd
import music_feature

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r"C:\course\sem 4\ml Project\complete app folder\Music Dataset new.csv")
X = data.drop("Instrument", axis=1)
y = data["Instrument"]


scaler = StandardScaler()
X = scaler.fit_transform(X)


svm = SVC(C=10, gamma=0.001, kernel='rbf', class_weight='balanced')
knn = KNeighborsClassifier(n_neighbors=3)
rf = RandomForestClassifier(n_estimators=300)


svm.fit(X, y)
knn.fit(X, y)
rf.fit(X, y)

uploaded_file = st.file_uploader("Upload a .wav audio file of a musical instrument",type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    f = music_feature.make_feature("temp.wav")
    f = f.reshape(1, -1)
    f = scaler.transform(f)

    pred_svm = svm.predict(f)[0]
    pred_knn = knn.predict(f)[0]
    pred_rf  = rf.predict(f)[0]

    st.subheader("Model Predictions")
    st.write(f"SVM: {pred_svm}")
    st.write(f"KNN: {pred_knn}")
    st.write(f"Random Forest: {pred_rf}")

    votes=[pred_svm, pred_knn, pred_rf]

    prediction = max(set(votes), key=votes.count)
    if prediction == "pia":
        prediction = "Piano"
    elif prediction == "voi":
        prediction = "Voice"
    elif prediction == "cel":
        prediction = "Cello"
    elif prediction == "cla":
        prediction = "Clarinet"
    elif prediction == "flu":
        prediction = "Flute"
    elif prediction == "gac":
        prediction = "Acoustic Guitar"
    elif prediction == "gel":
        prediction = "Electric Guitar"
    elif prediction == "org":
        prediction = "Organ"
    elif prediction == "sax":
        prediction = "Saxophone"
    elif prediction == "tru":
        prediction = "Trumpet"
    elif prediction == "vio":
        prediction = "Violin"
    st.subheader("Final Prediction")
    st.success(prediction)
