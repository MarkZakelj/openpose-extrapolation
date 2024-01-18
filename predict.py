import torch
from fastapi import FastAPI
from model import LitAutoEncoder
from schema import InferenceRequest, Keypoint

best_model_path = 'lightning_logs/version_5/checkpoints/epoch=9-step=3050.ckpt'

model = LitAutoEncoder.load_from_checkpoint(best_model_path)
model.eval()


app = FastAPI()

@app.get("/")
def health():
    return "OK"

@app.post("/predict")
def predict(input: InferenceRequest):
    ps = []
    for keypoint in input.keypoints:
        if keypoint.visible:
            ps.extend([keypoint.x, keypoint.y])
        else:
            ps.extend([-1.0, -1.0])
    ps.append(input.height / input.width)
    x = torch.tensor([ps])
    with torch.no_grad():
        y_hat = model(x)
    y_hat = y_hat[0][:-1].reshape(-1, 2).tolist()
    keypoints = []
    for row in y_hat:
        points = row[:-1].reshape(-1, 2).tolist()
        translated_points = []
        for point in points:
            translated_points.append(Keypoint(x=point[0], y=point[1], visible=True))
        keypoints.append(translated_points)
    return {"keypoints": keypoints}
