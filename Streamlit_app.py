import streamlit as st
import pandas as pd
import pickle
import pickle

model_path = r"C:\Users\sansk\Documents\Project 2\heart_disease_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")


