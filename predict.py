import torch
from fastapi import FastAPI
from model import LitAutoEncoder
from schema import InferenceRequest, Keypoint, InferenceResult

best_model_path = 'models/skeleton-extrapolation.ckpt'

model = LitAutoEncoder.load_from_checkpoint(best_model_path)
model.eval()
model.to('cpu')


app = FastAPI()

@app.get("/")
def health():
    return "OK"

@app.post("/predict")
def predict(input: InferenceRequest) -> InferenceResult:
    x = []
    for body in input.keypoints:
        ps = []
        for keypoint in body:
            if keypoint.visible:
                ps.extend([keypoint.x, keypoint.y])
            else:
                ps.extend([-1.0, -1.0])
        ps.append(input.height / input.width)
        x.append(ps)
    x = torch.tensor(x).to('cpu')
    with torch.no_grad():
        y_hat = model(x)
    keypoints = []
    for row in y_hat:
        points = row[:-1].reshape(-1, 2).tolist()
        translated_points = []
        for point in points:
            translated_points.append(Keypoint(x=point[0], y=point[1], visible=True))
        keypoints.append(translated_points)
    return {"keypoints": keypoints}
