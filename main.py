from concurrent.futures import ThreadPoolExecutor
import os
import time
import base64
from faster_whisper import WhisperModel
from flask import request, Flask
import uuid

executor = ThreadPoolExecutor(max_workers=10)
model_size = "small.en"
device = os.environ.get("DEVICE","cpu")
# Run on GPU with FP16
# model = WhisperModel(model_size, device="cpu", compute_type="float32")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device=device, compute_type="float32")

# segments, info = model.transcribe("audio.mp3", beam_size=5)
# start_time = time.time()
# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
# print("--- %s seconds ---" % (time.time() - start_time))


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.post('/transcribe')
    def transcribe():
        def _transcribe(audio, userId):
            file_name = f"{userId}-{str(uuid.uuid4())}.wav"
            f = open(f"{file_name}", "wb")
            f.write(base64.b64decode(audio))
            f.close()
            segments, info = model.transcribe(
                f"{file_name}", beam_size=5)
            segment_list = list(segments)
            print(segment_list)
            return segment_list
        
        job = executor.submit(_transcribe,request.json['audio'],request.json['userId'])
        result = job.result()
        result = result[0].text if len(result) > 0 else ""
        return result

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", debug=True)
