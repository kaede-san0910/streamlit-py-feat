import streamlit as st
from PIL import Image
import numpy as np
import gc

from rmn import RMN
m = RMN()

# タイトルの描画
st.title("py-feat demo app")

# 画像を撮影
img_buf = st.camera_input("")


# 画像を撮影後
if img_buf:

    # numpy BGR
    img = np.uint8(Image.open(img_buf))[:, :, ::-1]

    # メモリ解放
    del img_buf
    gc.collect()

    # 感情推定
    try:
        results = m.detect_emotion_for_single_frame(img)
        st.write(results)

    except Exception as e:
        print(e)

        # # 結果の表示
        # st.write("Show result")
        # fig, ax = plt.subplots()
        # result.plot.barh(ax=ax, stacked=True, figsize=(12, 2))
        # ax.legend(result.columns, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=8)
        # ax.set_title("Facial Expression Analysis")
        # ax.set_xlim(0, 1)
        # ax.axes.yaxis.set_visible(False)
        # st.pyplot(fig)
        # st.write(result)

        # # メモリ解放
        # del result, fig, ax
        # gc.collect()