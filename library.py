import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint as pp
import typing_extensions as typing
# import enum

class PhishingDiagonosis(typing.TypedDict):
    Answers: list[int]
    Prediction: str

# lvalue converted to final rvalue
# class Choice(enum.Enum): # only for response schemas, useless
#     Yes = 1
#     No = 0

load_dotenv()

# https://www.sciencedirect.com/science/article/pii/S0167404822003297
# referred to Literature review list at section 2
# https://www.trellix.com/blogs/research/understanding-phishing-psychology-effective-strategies-and-tips/
def getAttributes():
    max_temp = 1.0
    f = open("system_ins.txt", "r")
    sys_ins = f.read()
    f.close()
    # Once you have fully answered these questions place them in a JSON format with keys being the question number and values being binary, specifically yes being 1 and no being 0.
    return {"max_temp" : max_temp, 
            "sys_ins": sys_ins}

def parseDataset(folder_path, file):
    filepath = os.path.join(os.getcwd(), f"{folder_path}\\{file}") 
    df = pd.read_csv(filepath, skiprows=1, header=None)
    df = df.drop(df.columns[0], axis=1)

    features = df.iloc[:, 0].values  # Assuming the first column is the text feature
    labels = df.iloc[:, 1].values  # Assuming the second column is the text label
    return features, labels

label_encoded = {"Safe Email" : 0, "Phishing Email" : 1}

def writeToFile(text):
    f = open("output.txt", "a")
    f.write(text)
    f.close()

def getModelResponse(sys_ins, prompt):
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash', system_instruction = sys_ins)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1, # only accepting one response
            temperature=0.5,
            response_mime_type="application/json", 
            response_schema=list[PhishingDiagonosis], # JSON schema
        ),
    )
    return response.text