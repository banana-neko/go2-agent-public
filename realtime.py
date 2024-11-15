import asyncio
import websockets
import pyaudio
import numpy as np
import base64
import json
import wave
import io
import os
from pynput import keyboard
import sys
from go2_tools_test import tools, tool_dict

shift_pressed = False

def on_press(key):
    global shift_pressed
    if key == keyboard.Key.shift:
        shift_pressed = True
        print("🎙️ Recording...")

def on_release(key):
    global shift_pressed
    if key == keyboard.Key.shift:
        shift_pressed = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)


def tool_handler(tool_name, args=None):
    tool_dict[tool_name]()


def base64_to_pcm16(base64_audio):
    audio_data = base64.b64decode(base64_audio)
    return audio_data


async def send_audio(websocket, stream, CHUNK):
    def read_audio_block():
        try:
            return stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print(f"[Error] {e}")
            return None
    
    # バッファに音声が保存されているか？
    audio_saved = False

    while True:
        if shift_pressed:
            audio_data = await asyncio.get_event_loop().run_in_executor(None, read_audio_block)
            if audio_data is None:
                    continue  # 読み取りに失敗した場合はスキップ
            audio_saved = True
            # PCM16データをBase64にエンコード
            base64_audio = base64.b64encode(audio_data).decode("utf-8")

            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }

            await websocket.send(json.dumps(audio_event))
        else:
            if audio_saved:
                await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await websocket.send(json.dumps({"type": "response.create"}))
                audio_saved = False

        await asyncio.sleep(0)


async def receive_audio(websocket, output_stream):
    loop = asyncio.get_event_loop()

    while True:
        response = await websocket.recv()
        response_data = json.loads(response)
       # print(response_data)

        # サーバーからの応答をリアルタイム（ストリーム）で表示
        if "type" in response_data and response_data["type"] == "response.function_call_arguments.done":
            func_name = response_data["name"]
            args = response_data["arguments"]
            call_id = response_data["call_id"]
            tool_handler(func_name)
            
            func_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "正常に動作しました。"
                }
            }
            await websocket.send(json.dumps(func_event))
            await websocket.send(json.dumps({"type": "response.create"}))
            print(f"<FunctionCalling> name: {func_name}, args: {args}", end="")

        elif "type" in response_data and response_data["type"] == "response.created":
            print("assistant: ", end="", flush=True)

        elif "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
            print(response_data["delta"], end = "", flush = True)

        # サーバからの応答が完了したことを取得
        elif "type" in response_data and response_data["type"] == "response.done":
            print()

        elif "type" in response_data and response_data["type"] == "error":
            print(f"error:\n{response_data['error']}")
            sys.exit()

        # サーバーからの応答に音声データが含まれているか確認
        if "delta" in response_data:
            if response_data["type"] == "response.audio.delta":
                base64_audio_response = response_data["delta"]
                if base64_audio_response:
                    pcm16_audio = base64_to_pcm16(base64_audio_response)
                    #音声データがある場合は、出力ストリームから再生
                    await loop.run_in_executor(None, output_stream.write, pcm16_audio)


API_KEY = os.environ.get('OPENAI_API_KEY')

# header info
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
HEADERS = {
    "Authorization": "Bearer "+ API_KEY, 
    "OpenAI-Beta": "realtime=v1"
}

async def stream_audio_and_receive_response():
    async with websockets.connect(WS_URL, extra_headers=HEADERS) as websocket:
        print("[INFO] WebSocket connection established.")

        init_request = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": "ユーザーからの入力に対し適切な動作を選択して呼び出してください。例えば、「慰めて」や「なにかして」と言われたら適切な動作を選択して呼び出してください。また、大阪弁で軽快に喋ってください。",
                "voice": "alloy", #"alloy", "echo", "shimmer"
                "turn_detection": None,
                "tools": tools,
                "tool_choice": "auto"
            }
        }
        await websocket.send(json.dumps(init_request))
        print("[INFO] Initial request sent to the server.")
        
        # PyAudioの設定
        CHUNK = 2048          # マイクからの入力データのチャンクサイズ
        FORMAT = pyaudio.paInt16  # PCM16形式
        CHANNELS = 1          # モノラル
        RATE = 24000          # サンプリングレート（24kHz）

        # PyAudioインスタンス
        p = pyaudio.PyAudio()

        # マイクストリームの初期化
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        # サーバーからの応答音声を再生するためのストリームを初期化
        output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

        print("[INFO] Microphone input activated. Starting audio playback from server...")
        print()

        try:
            # 音声送信タスクと音声受信タスクを非同期で並行実行
            send_task = asyncio.create_task(send_audio(websocket, stream, CHUNK))
            receive_task = asyncio.create_task(receive_audio(websocket, output_stream))

            # タスクが終了するまで待機
            await asyncio.gather(send_task, receive_task)

        except KeyboardInterrupt:
            # キーボードの割り込みで終了
            print("[Exit]")
        finally:
            # ストリームを閉じる
            if stream.is_active():
                stream.stop_stream()
            stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()


if __name__ == "__main__":
    listener.start()
    asyncio.get_event_loop().run_until_complete(stream_audio_and_receive_response())
    listener.join()
