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
    neck_points = []
    print('KEYPOINTS:')
    for body in input.keypoints:
        print('body')
        ps = []
        if not body[1].visible:
            return {"keypoints": input.keypoints}
        neckx = body[1].x
        necky = body[1].y
        print(f'neck: {neckx} {necky}')
        neck_points.append([neckx, necky])
        for keypoint in body:
            print(f'x: {keypoint.x} y: {keypoint.y} visible: {keypoint.visible}')
            if keypoint.visible:
                ps.extend([keypoint.x - neckx, keypoint.y - necky])
            else:
                ps.extend([-10.0, -10.0])
        ps.append(input.height / input.width)
        x.append(ps)
    x = torch.tensor(x).to('cpu')
    print('x: ', x)
    with torch.no_grad():
        y_hat = model(x)
    print('yhat', y_hat)
    keypoints = []
    for i, row in enumerate(y_hat):
        points = row[:-1].reshape(-1, 2).tolist()
        translated_points = []
        neckx, necky = neck_points[i]
        for point in points:
            translated_points.append(Keypoint(x=point[0] + neckx, y=point[1] + necky, visible=True))
        keypoints.append(translated_points)
    return {"keypoints": keypoints}
