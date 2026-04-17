import asyncio
import re
from picamera2 import Picamera2
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import openai
import base64
import azure.cognitiveservices.speech as speechsdk
import sys
import traceback
picam2 = Picamera2()
picam2.start()
def excepthook(exc_type, exc_value, exc_traceback):
    print("excepthook:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = excepthook

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

# ----------- OpenAI 新版Client初始化 -----------
OPENAI_API_KEY = os.getenv("Azure_OPENAI_API_KEY")  # 从环境变量获取API Key
BASE_URL =  os.getenv("Azure_OPENAI_BASE_URL", "https://api.openai.com/v1")  # 可选：自定义Base URL，默认为官方URL
OPENAI_CHAT_MODEL = os.getenv("Azure_OPENAI_CHAT_MODEL", "gpt-4.1")

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

# ----------- Azure Speech 配置 -----------
SPEECH_KEY = os.getenv("SPEECH_KEY")  # 从环境变量获取Azure语音服务Key
SPEECH_REGION = os.getenv("SPEECH_REGION", "japaneast")  # 可选：自定义区域，默认为"japaneast"
chat_history = []
SYSTEM_PROMPT_TEXT = (
        "你是智能助手。每次都要用简洁、直接的话帮用户解决问题。"
        "请根据上下文和用户上传的图像，结合实际情况分析、指导。"
        "除非用户要求详细说明，否则默认精炼回答。如用户说“详细说明”再补充更多细节。"
    )

previous_response_id = None
# 播放队列
tts_play_queue = asyncio.Queue()
main_event_loop = asyncio.new_event_loop()
def loop_runner(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=loop_runner, args=(main_event_loop,), daemon=True).start()

# ----------- Mediapipe初始化 -----------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture("libcamerasrc ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_queue = queue.Queue(maxsize=2)  # 小容量，保证让input线程总能拿到"新"帧
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "zh-CN"
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)  # None表示直接播放

def to_ssml(text, lang, voice_name):
    return f'''
        <speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="{lang}">
            <voice name="{voice_name}">
                <lang xml:lang="{lang}">
                    <prosody rate="17%">{text.replace('*', ' ').replace('#', ' ')}</prosody>
                </lang>
            </voice>
        </speak>
    '''

async def tts_audio_enqueue(sentence, lang, voice_name):
    await tts_play_queue.put(sentence)
    '''
    ssml_text = to_ssml(sentence, lang, voice_name)
    
    t=speech_synthesizer.speak_ssml_async(ssml_text)
    await tts_play_queue.put(t)
    '''

async def tts_player():
    """
    后台任务：串行播放队列tts。用Azure SDK自己的playback，不会和队列其它冲突。
    """
    while True:
        sentence = await tts_play_queue.get()
        ssml_text = to_ssml(sentence, lang="zh-CN", voice_name="zh-CN-XiaoxiaoMultilingualNeural")
        #print(f"\n【朗读】{sentence}\n")
        result = speech_synthesizer.speak_ssml_async(ssml_text).get()
        # 或者保存成文件再播放
        tts_play_queue.task_done()

asyncio.run_coroutine_threadsafe(tts_player(), main_event_loop)

def split_sentences(text):
    # 支持中英文分句，标点后切割
    sents = re.split(r'(?<=[。！？!?\.])', text)
    return [s.strip() for s in sents if s.strip()]

def ask_openai(image_np, question):
    import base64
    import cv2
    lang = "zh-CN"
    voice_name = "zh-CN-XiaoxiaoNeural"  # 可替换为其他Azure TTS支持的中文声音
    _, buf = cv2.imencode('.jpg', image_np)
    b64str = base64.b64encode(buf).decode('utf-8')
    input_payload=[
        {"role": "system", "content": SYSTEM_PROMPT_TEXT},
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": question },
                { "type": "input_image", "image_url": f"data:image/jpeg;base64,{b64str}" }
            ],
        }
    ]
    
    try:
        print("\n=== OpenAI 回答(流式) ===")
        global previous_response_id
        stream = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            input=input_payload,
            previous_response_id=previous_response_id,
            stream=True,
        )
        answer = ""
        final_response = None
        first_chunk = True
        buffer = ''
        sent_buffer = []
        current_count = 1  # 初始朗读1句
        full_reply = ""
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
                content = event.delta
                answer += content
                buffer += content
                if content:
                    sentences = split_sentences(buffer)
                    # 只保留最后残句
                    if sentences:
                        buffer = sentences[-1] if buffer and not re.match(r'.*[。！？!?\.]$', buffer) else ''
                        for sent in sentences[:-1] if buffer else sentences:
                            if sent.strip():
                                sent_buffer.append(sent.strip())
                        # 可按你的倍增/一次一条读
                        to_read = ' '.join(sent_buffer)
                        if to_read:
                            asyncio.run_coroutine_threadsafe(
                                tts_audio_enqueue(to_read, lang, voice_name), main_event_loop
                            )
                            sent_buffer = []
            elif event.type == "response.completed":
                final_response = event.response
                print()  # 换行
                break

        # 收尾
        if buffer.strip():
            asyncio.run_coroutine_threadsafe(
                tts_audio_enqueue(buffer.strip(), lang, voice_name), main_event_loop
            )

        if final_response is not None:
            previous_response_id = final_response.id
        print("=" * 30)
    except Exception as e:
        print(f"OpenAI出错: {e}")

# ----------- 语音识别线程 -----------
def speech_thread():
    
    print("请用麦克风提问，对着麦克风说话……（按 Ctrl+C 停止）")
    while True:
        print("\n【请提问......】")
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            question = result.text.strip()
            print(f"识别到语音: {question}")
            if not question:
                continue
            if not frame_queue.empty():
                # 拿最新帧
                image_np = frame_queue.get()
                chat_history.append({"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": None}},  # 先临时为None
                ]})
                # 保证最多10轮（每轮1条user和对应的assistant，合计20条，有图像就只1user+1assistant）
                chat_history[:] = chat_history[-10:]
                ask_openai(image_np, question)
                print("等待朗读完毕……")
                asyncio.run_coroutine_threadsafe(tts_play_queue.join(), main_event_loop).result()
                print("朗读结束，可以新一轮提问。")
            else:
                print("未获得画面，请再试。")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("未识别到语音。")
        elif result.reason == speechsdk.ResultReason.Canceled:
            print("语音识别被取消或出错。")
            cancellation_details = result.cancellation_details
            print(f"Cancellation details: {cancellation_details.reason}")

t = threading.Thread(target=speech_thread, daemon=True)
t.start()

# ----------- 主循环 -----------
while True:
    #success, frame = cap.read()
    frame = picam2.capture_array()
    '''
    if not success:
        print("cap failed.")
        break
    '''
    #frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    # 绘制关键点和骨架
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for idx, landmark in enumerate(hand):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                color = (0,255,0) if idx in [4,8,12,16,20] else (255,0,0)
                cv2.circle(frame, (x, y), 6, color, -1)
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)]
            for start, end in HAND_CONNECTIONS:
                x_start = int(hand[start].x * w)
                y_start = int(hand[start].y * h)
                x_end = int(hand[end].x * w)
                y_end = int(hand[end].y * h)
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (0,200,200), 2)
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Two Hands Landmarks", frame_bgr)

    # 队列只放一帧，防止积压
    if not frame_queue.full():
        frame_queue.put(frame.copy())

    # ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

#cap.release()
picam2.close()
cv2.destroyAllWindows()