import re  # for regex operations
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from typing import Optional

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run some code on application startup and shutdown.
    1. Code before 'yield' runs on startup.
    2. Code after 'yield' runs on shutdown.
    """
    print("Hello")
    yield
    print("Goodbye")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


# item_id is a 'path parameter' because it is part of the URL path.
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


# @app.get() specifies that this function will handle GET requests to the root URL ("/").
# http://localhost:8000/docs    displays the automatic interactive API documentation (provided by Swagger UI).
# http://localhost:8000/redoc   displays the alternative API documentation (provided by ReDoc).


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


# Any parameter in FastAPI that is not a path parameter will be considered a 'query parameter':
#       Example request: http://localhost:8000/query_items?item_id=3
@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


# NEVER implement databases like this!
database = {"username": [], "password": []}


# http://localhost:8000/login/?username=Olivia&password=123
@app.post("/login/")
def login(username: str, password: str):
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.get("/text_model/")
def contains_email(data: str):
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None,
    }
    return response


@app.post("/text_model_advanced/")
def contains_specific_domain_email(input: dict):
    email = input.get("email", "")
    domain_match = input.get("domain_match", "")
    regex = rf"\b[A-Za-z0-9._%+-]+@{re.escape(domain_match)}\.[A-Z|a-z]{{2,}}\b"
    response = {
        "input": email,
        "domain": domain_match,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, email) is not None,
    }
    return response


# BaseModel is used to define the expected structure of the request body.
class TextModelAdvancedRequest(BaseModel):
    email: str
    domain_match: str


@app.post("/text_model_advanced_base_model/")
def contains_specific_domain_email(input: TextModelAdvancedRequest):
    email = input.email
    domain_match = input.domain_match
    regex = rf"\b[A-Za-z0-9._%+-]+@{re.escape(domain_match)}\.[A-Z|a-z]{{2,}}\b"
    response = {
        "input": email,
        "domain": domain_match,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, email) is not None,
    }
    return response


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = 28, w: int = 28):  # async is needed for UploadFile
    """
    Resizes an uploaded image to the specified height and width.

    Args:
        data (UploadFile): The uploaded image file.
        h (int): Height to resize the image to. Default is 28.
        w (int): Width to resize the image to. Default is 28.
    """
    with open("image.jpg", "wb") as image_file:
        content = await data.read()
        image_file.write(content)

    # Resize
    im = cv2.imread("image.jpg")
    res = cv2.resize(im, (h, w))
    cv2.imwrite("resized_image.jpg", res)

    # response = {
    #     "input": data,
    #     "message": HTTPStatus.OK.phrase,
    #     "status-code": HTTPStatus.OK,
    # }
    # return response

    return FileResponse("resized_image.jpg")
