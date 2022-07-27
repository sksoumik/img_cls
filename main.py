from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
    <head>
    <title>Image Classification</title>
    </head>
    <body>
    <h1>Image Classification</h1>

    </body>
    </html>
    """
    return content




