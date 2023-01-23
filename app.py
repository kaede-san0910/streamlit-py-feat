import streamlit as st
from PIL import Image
from feat import Detector


# 感情認識モデルのセットアップ
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "svm"
emotion_model = "resmasknet"
detector = Detector(
    face_model=face_model,
    landmark_model=landmark_model,
    au_model=au_model,
    emotion_model=emotion_model
)

# タイトルの描画
st.title("py-feat demo app")

# 画像を撮影
img_buf = st.camera_input("カメラで撮影")

# 画像を撮影後
if img_buf:

    # 画像ファイルを一時保存
    Image.open(img_buf).save("_temp.jpg")

    # 感情推定
    result = detector.detect_image("_temp.jpg").to_dict()
    result = {
            "anger": result["anger"],
            "disgust": result["disgust"],
            "fear": result["fear"],
            "happiness": result["happiness"],
            "sadness": result["sadness"],
            "superise": result["surprise"],
            "neutral": result["neutral"]
        }
    st.write(result)
