
IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']





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


# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILENAMES[0])

# STEP 4: Classify the input image.
classification_result = classifier.classify(image)
# print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it.
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")
