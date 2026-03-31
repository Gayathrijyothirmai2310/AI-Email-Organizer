import pandas as pd
import os
import pickle
from tkinter import *
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_FILE = "email_model.pkl"
VEC_FILE = "vectorizer.pkl"

# Train the model
def train_model():
    data = pd.read_csv("database.csv")
    data['text'] = data['text'].str.lower()

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['category']

    model = MultinomialNB()
    model.fit(X, y)

    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(vectorizer, open(VEC_FILE, "wb"))

# Train if model not exists
if not os.path.exists(MODEL_FILE):
    train_model()

# Load model
model = pickle.load(open(MODEL_FILE, "rb"))
vectorizer = pickle.load(open(VEC_FILE, "rb"))

# Create folders
folders = ["Work", "Personal", "Promotions", "Spam"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to organize email
def organize_email():
    email_text = entry.get()

    if email_text == "":
        messagebox.showerror("Error", "Please enter email text")
        return

    email_vector = vectorizer.transform([email_text])
    category = model.predict(email_vector)[0]

    file_count = len(os.listdir(category)) + 1
    filename = f"{category}/email_{file_count}.txt"

    with open(filename, "w") as f:
        f.write(email_text)

    result_label.config(text=f"Email Category: {category}")
    entry.delete(0, END)

# GUI
root = Tk()
root.title("AI Email Organizer")
root.geometry("500x300")

title = Label(root, text="AI Email Organizer", font=("Arial", 18))
title.pack(pady=15)

label = Label(root, text="Enter Email Text")
label.pack()

entry = Entry(root, width=50)
entry.pack(pady=10)

button = Button(root, text="Organize Email", command=organize_email)
button.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()