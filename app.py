import gc

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rmn import RMN


# 感情推定モデルのセットアップ
model = RMN()

# タイトルの描画
st.title("感情分析デモ")

# 画像を撮影
# img_buf = st.camera_input("カメラで撮影")
img_buf = st.file_uploader("画像をアップロードしてください")

# 画像を撮影後
if img_buf:

    # 感情推定
    try:
        img = np.uint8(Image.open(img_buf).convert("RGB"))

        # メモリ解放
        del img_buf
        gc.collect()

        results = model.detect_emotion_for_single_frame(img)
        img = model.draw(img, results)
        st.image(img)

        # 顔が検出された時
        if results:
            probs = results[0]['proba_list']
            probs = pd.json_normalize({list(d.keys())[0]: list(d.values())[0] for d in probs})
            probs.index = ["score"]

            # 結果の表示
            st.write("分析結果")
            fig, ax = plt.subplots()
            probs.plot.barh(ax=ax, stacked=True, figsize=(12, 2))
            ax.legend(probs.columns, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=8)
            ax.set_title("Facial Expression Analysis")
            ax.set_xlim(0, 1)
            ax.axes.yaxis.set_visible(False)
            st.pyplot(fig)

            # メモリ解放
            del results, probs, fig, ax
            gc.collect()

        # 顔が検出されなかった時
        else:
            st.write("判定できませんでした")

    except Exception as e:
        print(e)
        st.write("エラーが発生しました")
