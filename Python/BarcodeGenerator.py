from fastapi import FastAPI
from fastapi.responses import FileResponse
import qrcode

app = FastAPI()

@app.get("/generate/{text}")
def generate_qr(text: str):
    file = f"{text}.png"
    qrcode.make(text).save(file)
    return FileResponse(file, media_type="image/png")

