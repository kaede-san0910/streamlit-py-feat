import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pandas as pd

import av
from rmn import RMN
m = RMN()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


result_queue = queue.Queue()


def webcam_callback(frame: av.VideoFrame):
    """webcamのコールバック処理"""
    image = frame.to_ndarray(format="bgr24")

    # 感情分析
    result = m.detect_emotion_for_single_frame(image)
    annotated_image = m.draw(image, result)

    if result:
        # convert dataframe
        probs = result[0]['proba_list']
        probs = pd.json_normalize({list(d.keys())[0]: list(d.values())[0] for d in probs})
        probs.index = ["score"]
    else:
        probs = None

    # キュー経由で描画処理
    result_queue.put(probs)
    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


def write_results():
    """結果を描画する"""
    labels = st.empty()
    while True:
        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            result = None
        labels.write(result)


ctx = webrtc_streamer(
    key="streamlit-webrtc-py-feat",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=webcam_callback,
    media_stream_constraints={"video": True},
    async_processing=True
)
if ctx.state.playing:
    write_results()
