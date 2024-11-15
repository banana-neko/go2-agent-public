import asyncio
import websockets
import pyaudio
import numpy as np
import base64
import json
import wave
import io
import os

API_KEY = os.environ.get('OPENAI_API_KEY')
#わからない人は、上の行をコメントアウトして、下記のように直接API KEYを書き下してもよい
#API_KEY = "sk-xxxxx"

# WebSocket URLとヘッダー情報
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
HEADERS = {
    "Authorization": "Bearer "+ API_KEY, 
    "OpenAI-Beta": "realtime=v1"
}

# PCM16形式に変換する関数
def base64_to_pcm16(base64_audio):
    audio_data = base64.b64decode(base64_audio)
    return audio_data

# 音声を送信する非同期関数
async def send_audio(websocket, stream, CHUNK):
    def read_audio_block():
        """同期的に音声データを読み取る関数"""
        try:
            return stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print(f"音声読み取りエラー: {e}")
            return None

    print("マイクから音声を取得して送信中...")
    while True:
        # マイクから音声を取得
        audio_data = await asyncio.get_event_loop().run_in_executor(None, read_audio_block)
        if audio_data is None:
                continue  # 読み取りに失敗した場合はスキップ
        
        # PCM16データをBase64にエンコード
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        audio_event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }

        # WebSocketで音声データを送信
        await websocket.send(json.dumps(audio_event))

        await asyncio.sleep(0)

# サーバーから音声を受信して再生する非同期関数
async def receive_audio(websocket, output_stream):
    print("assistant: ", end = "", flush = True)
    loop = asyncio.get_event_loop()
    while True:
        # サーバーからの応答を受信
        response = await websocket.recv()
        response_data = json.loads(response)

        # サーバーからの応答をリアルタイム（ストリーム）で表示
        if "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
            print(response_data["delta"], end = "", flush = True)
        # サーバからの応答が完了したことを取得
        elif "type" in response_data and response_data["type"] == "response.audio_transcript.done":
            print("\nassistant: ", end = "", flush = True)

        # サーバーからの応答に音声データが含まれているか確認
        if "delta" in response_data:
            if response_data["type"] == "response.audio.delta":
                base64_audio_response = response_data["delta"]
                if base64_audio_response:
                    pcm16_audio = base64_to_pcm16(base64_audio_response)
                    #音声データがある場合は、出力ストリームから再生
                    await loop.run_in_executor(None, output_stream.write, pcm16_audio)

# マイクからの音声を取得し、WebSocketで送信しながらサーバーからの音声応答を再生する非同期関数
async def stream_audio_and_receive_response():
    # WebSocketに接続
    async with websockets.connect(WS_URL, extra_headers=HEADERS) as websocket:
        print("WebSocketに接続しました。")

        # 初期リクエスト (モダリティ設定)
        init_request = {
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": "ユーザーをサポートしてください。",
                "voice": "echo" #"alloy", "echo", "shimmer"
            }
        }
        await websocket.send(json.dumps(init_request))
        print("初期リクエストを送信しました。")
        
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

        print("マイク入力およびサーバーからの音声再生を開始...")

        try:
            # 音声送信タスクと音声受信タスクを非同期で並行実行
            send_task = asyncio.create_task(send_audio(websocket, stream, CHUNK))
            receive_task = asyncio.create_task(receive_audio(websocket, output_stream))

            # タスクが終了するまで待機
            await asyncio.gather(send_task, receive_task)

        except KeyboardInterrupt:
            # キーボードの割り込みで終了
            print("終了します...")
        finally:
            # ストリームを閉じる
            if stream.is_active():
                stream.stop_stream()
            stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(stream_audio_and_receive_response())
