import torch
from fastapi import FastAPI
from model import LitAutoEncoder
from schema import InferenceRequest, Keypoint, InferenceResult
import xgboost as xgb

model_name = 'skeleton-extrapolation.ckpt'
best_model_path = f'models/{model_name}'

model = LitAutoEncoder.load_from_checkpoint(best_model_path)
model.eval()
model.to('cpu')

# read xgboost models
xgboost_models = {}
for mod in range(36):
    model = xgb.load_model(f'models/xgboost/v4/model_{mod}.json')
    xgboost_models[mod] = model


app = FastAPI()

@app.get("/")
def health():
    return "OK"

@app.post("/predict")
def predict(input: InferenceRequest) -> InferenceResult:
    x = []
    neck_points = []
    for body in input.keypoints:
        ps = []
        neckx = body[1].x
        necky = body[1].y
        neck_points.append([neckx, necky])
        for keypoint in body:
            if keypoint.visible:
                ps.extend([keypoint.x - neckx, keypoint.y - necky])
            else:
                ps.extend([-10.0, -10.0])
        ps.append(input.height / input.width)
        x.append(ps)
    x = torch.tensor(x).to('cpu')
    with torch.no_grad():
        y_hat = model(x)
    keypoints = []
    for i, row in enumerate(y_hat):
        points = row[:-1].reshape(-1, 2).tolist()
        translated_points = []
        neckx, necky = neck_points[i]
        for point in points:
            translated_points.append(Keypoint(x=point[0] + neckx, y=point[1] + necky, visible=True))
        keypoints.append(translated_points)
    return {"keypoints": keypoints}

@app.post('/predict-xgboost')
def predict(input: InferenceRequest) -> InferenceResult:
    keypoints = []
    for body in input.keypoints:
        x = []
        missing = []
        neckx, necky = body[1]
        for i, keypoint in enumerate(body):
            if keypoint.visible:
                x.extend([keypoint.x - neckx, keypoint.y - necky])
            else:
                x.extend([-10.0, -10.0])
                missing.append((2 * i, 2 * i + 1))
        inp = xgb.DMatrix([x])
        for (px, py) in missing:
            x[px // 2][0] = xgboost_models[px].predict(inp)[0]
            x[px // 2][1] = xgboost_models[py].predict(inp)[0]
        keypoints.append([Keypoint(x=point[0] + neckx, y=point[1] + necky, visible=True) for point in x])
    return {"keypoints": keypoints}
        
        
        

