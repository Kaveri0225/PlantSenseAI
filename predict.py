from transformers import pipeline

MODEL_NAME = "neghani/plantvillage-mobilenetv2"

def load_model():
    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        revision="v1.0"
    )
    return classifier

def predict_plant(image):
    classifier = load_model()
    predictions = classifier(image, top_k=3)
    return predictions