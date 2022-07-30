from cgitb import html
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
from ui_utils import get_html_table, head_html, body_html

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def main():
    content = head_html + body_html
    return content
