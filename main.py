import base64
import time

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from sqlalchemy import create_engine, text

# uvicorn main:app --reload
app = FastAPI()
engine = create_engine("postgresql://myuser:mypassword@localhost:5432/mydatabase")

origins = [
    "http://localhost:5173",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                # List of allowed origins
    allow_credentials=True,               # Allow cookies and authentication
    allow_methods=["*"],                  # Allow all HTTP methods
    allow_headers=["*"],                  # Allow all headers
)

# Load model
class GarmentClassifier(torch.nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate and load the trained model
PATH = 'model_20241208_214737_4'
model = GarmentClassifier()
model.load_state_dict(torch.load(PATH))
model.eval()

# Define classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define prediction function
def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# Define API input model
class PredictionRequest(BaseModel):
    file: UploadFile

@app.post("/predict")
async def predict_endpoint(file: UploadFile):
    start_time = time.time()
    input_data, prediction, error_message = None, None, None

    if not file.content_type.startswith("image/"):
        log_level = "ERROR"
        error_message = "file 파라미터는 이미지여야 합니다."
    else:
        image_bytes = await file.read()
        file_string = base64.b64encode(image_bytes).decode("utf-8")
        input_data = file_string

        try:
            prediction = predict(image_bytes)
            log_level = "INFO"
        except Exception as e:
            log_level = "ERROR"
            error_message = str(e)
    execution_time = time.time() - start_time

    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(
                    text(
                        """
                        INSERT INTO logs (log_level, timestamp, http_method, endpoint, input_data, prediction, execution_time, error_message)
                        VALUES (:log_level, NOW(), :method, :endpoint, :input_data, :prediction, :execution_time, :error_message)
                        """
                    ),
                    {
                        "log_level": log_level,
                        "method": "POST",
                        "endpoint": "/predict",
                        "input_data": input_data,
                        "prediction": prediction,
                        "execution_time": execution_time,
                        "error_message": error_message,
                    },
                )
    except Exception as e:
        print(str(e))

    return {"class": prediction}