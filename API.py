from fastapi import FastAPI
from starlette.responses import FileResponse
import uvicorn

app = FastAPI()

@app.get("/")
def serve_home():
    return FileResponse('main.html')

@app.get("/model")
def serve_model():
    return FileResponse('model.onnx')

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host= "0.0.0.0")