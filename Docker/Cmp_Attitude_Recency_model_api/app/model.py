import joblib

def load_model():
    model = joblib.load("/app/model/model.pkl")
    return model



def predict(model, data):
    # Assuming the model supports the predict method
    return model.predict(data)
