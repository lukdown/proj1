from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite2.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)


app = FastAPI()

from PIL import Image
import numpy as np
import io
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()
    
    # content -> jpg 파일..  http 통신에서는 파일이 character type 왔다갔다함.
    # 1. text -> binary
    # 2. binary -> PIL Image
    # STEP 3: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
    
    # image = mp.Image.create_from_file(IMAGE_FILENAMES[0])
    
    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result": result}