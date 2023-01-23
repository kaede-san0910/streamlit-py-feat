import streamlit as st
from PIL import Image
from feat import Detector
import gc
# import matplotlib.pyplot as plt


# 感情認識モデルのセットアップ
face_model = "img2pose"
landmark_model = "mobilefacenet"
au_model = "svm"
emotion_model = "resmasknet"
detector = Detector(
    face_model=face_model,
    landmark_model=landmark_model,
    au_model=au_model,
    emotion_model=emotion_model
)

target_cols = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# タイトルの描画
st.title("py-feat demo app")

# 画像を撮影
img_buf = st.camera_input("")

# 画像を撮影後
if img_buf:

    # 画像ファイルを一時保存
    Image.open(img_buf).resize((352, 198)).save("_temp.jpg")

    # メモリ解放
    del img_buf
    gc.collect()

    # 感情推定
    try:
        result = detector.detect_image("_temp.jpg")[target_cols]
        result.index = ["score"]

        # 結果の表示
        st.write("Show result")
        # fig, ax = plt.subplots()
        # result.plot.barh(ax=ax, stacked=True, figsize=(12, 2))
        # ax.legend(result.columns, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=8)
        # ax.set_title("Facial Expression Analysis")
        # ax.set_xlim(0, 1)
        # ax.axes.yaxis.set_visible(False)
        # st.pyplot(fig)
        st.write(result)

        # メモリ解放
        del result
        gc.collect()

    except Exception as e:
        st.write("Face Recognition Error")
