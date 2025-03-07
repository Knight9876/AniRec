import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# reading the ".csv" file
anime_df = pd.read_csv("anime_with_synopsis.csv")
anime_df.head()

# returns the number of missing values in the dataset
anime_df.isnull().sum()

# dropna will drop all missing values from your original dataset
anime_df.dropna(inplace=True)

# returns the number of duplicated values in the dataset
anime_df.duplicated().sum()

# replaces any unknown values by np.nan
anime_df["Score"] = anime_df["Score"].map(lambda x: 0 if x == "Unknown" else float(x))

# replacing all null values by its median
anime_df["Score"].fillna(anime_df["Score"].median(), inplace=True)

# converting score to float
anime_df["Score"] = anime_df["Score"].astype(float)

# Top 10 Anime_df Based on Score
anime_df.sort_values(by="Score", ascending=False).head(10)

# convert the Genres and sypnopsis which is a string to a list
anime_df["Genres"] = anime_df["Genres"].apply(lambda x: x.split())
anime_df["sypnopsis"] = anime_df["sypnopsis"].apply(lambda x: x.split())

# remove space between two words
anime_df["Genres"] = anime_df["Genres"].apply(lambda x: [i.replace(" ", "") for i in x])
anime_df["sypnopsis"] = anime_df["sypnopsis"].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

# creating a new feature called "features"
anime_df["features"] = anime_df["Genres"] + anime_df["sypnopsis"]

# creating a new dataframe
new_anime_df = anime_df[["Name", "features"]]

# convert list to string
new_anime_df["features"] = new_anime_df["features"].apply(lambda x: " ".join(x))

# PorterStemmer is a stemming algorithm used to reduce words to their root or base form.

# creating an instance of the PorterStemmer
ps = PorterStemmer()


# stemming the text
def stem(text):
    stemmed_words = []

    for i in text.split():
        stemmed_words.append(ps.stem(i))

    return " ".join(stemmed_words)


new_anime_df["features"] = new_anime_df["features"].apply(stem)

# convert to lowercase
new_anime_df["features"] = new_anime_df["features"].apply(lambda x: x.lower())

# Countvectorizer is a method to convert text to numerical data

# creating an instance of CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")

# storing it in an array
vectors = cv.fit_transform(new_anime_df["features"]).toarray()

# using cosine similarity between these vectors to find their similarity.
similarity = cosine_similarity(vectors)


# recommendation function
def recommend(anime):
    try:
        anime_list = new_anime_df[new_anime_df["Name"] == anime].index[0]
        distances = similarity[anime_list]
        animes_list = sorted(
            list(enumerate(distances)), reverse=True, key=lambda x: x[1]
        )[1:100]
        recommendations = [new_anime_df.iloc[i[0]].Name for i in animes_list]
        return recommendations
    except IndexError:
        return []


# Display recommendations function
def display_recommendations():
    anime = anime_entry.get()
    recommendations = recommend(anime)
    if recommendations:
        update_recommendations(recommendations)
    else:
        messagebox.showwarning(
            "Anime Not Found", "Anime not found in database. Please try again."
        )

# Function to update recommendations
def update_recommendations(recommendations):
    # Clear the previous recommendations
    for widget in scrollable_container.winfo_children():
        widget.destroy()

    # If no recommendations are available, show the placeholder # if not require, remove the if part
    if not recommendations:
        placeholder_label = tk.Label(
            scrollable_container,
            text="Anime Recommendations Here!!!",
            font=(font_family, 14),
            bg="white",
            fg="white",
            anchor="center",
        )
        placeholder_label.pack(pady=20)  # Centered placeholder
    else :
        # Display each recommendation
        max_columns = 3  # Number of columns to allow
        for index, recommendation in enumerate(recommendations):
            label = tk.Label(
                scrollable_container,
                text=recommendation,
                font=(font_family, 14),
                bg="white",
                fg="white",
                anchor="w",  # Left align the text
                wraplength=610  # Adjust this value based on your window size
            )
            # Calculate row and column
            row = index // max_columns
            column = index % max_columns
            
            # Use grid to place labels in columns
            label.grid(row=row, column=column, padx=(30, 30), pady=5, sticky='w')


# Create the Tkinter window
root = tk.Tk()
root.title("AniRec: Anime Recommendation System")
root.state("zoomed")

# Styling
root.configure(bg="white")  # Darker background color

# Setting font style
font_family = "Arial"
font_size_header = 32
font_size_body = 18

# Header Frame
header_frame = tk.Frame(root, bg="white", padx=20, pady=20)
header_frame.pack(fill="x")

# Header label
header_label = tk.Label(
    header_frame, 
    text="Anime Recommendation System", 
    bg="#282C34", 
    fg="white", 
    font=(font_family, font_size_header, "bold")
)
header_label.pack()

# Input Frame
input_frame = tk.Frame(root, bg="white")
input_frame.pack(pady=20)

header_label_1 = tk.Label(
    input_frame,
    text="Enter the name of an anime:",
    font=(font_family, font_size_body, "bold"),
    bg="white",
    fg="white",
)
header_label_1.pack()

anime_entry = tk.Entry(input_frame, width=40, font=(font_family, 20))
anime_entry.pack(pady=10)

# Button with hover effect
def on_enter(e):
    recommend_button.config(bg="#557A95")

def on_leave(e):
    recommend_button.config(bg="#465A65")

recommend_button = tk.Button(
    input_frame,
    text="Get Recommendations",
    command=lambda: display_recommendations(),
    bg="#465A65",
    fg="white",
    font=(font_family, 16, "bold"),
    padx=10,
    pady=5,
    relief="flat"
)
recommend_button.pack(pady=10)

# Bind "Enter" key (Return) to call display_recommendations
root.bind('<Return>', lambda event: display_recommendations())

# Optional: Keep the hover effect
recommend_button.bind("<Enter>", on_enter)
recommend_button.bind("<Leave>", on_leave)

# Scrollable Results Frame
scrollable_frame = tk.Frame(root, bg="white")

# Create a canvas and scrollbars
canvas = tk.Canvas(scrollable_frame, bg="white")
vertical_scrollbar = tk.Scrollbar(scrollable_frame, orient="vertical", command=canvas.yview)
scrollable_container = tk.Frame(canvas, bg="white")

# Configure the canvas
scrollable_container.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_container, anchor="nw")
canvas.configure(yscrollcommand=vertical_scrollbar.set)

# Pack the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
vertical_scrollbar.pack(side="right", fill="y")

# Pack the scrollable frame
scrollable_frame.pack(expand=True, fill="both")

update_recommendations([])  # Call with an empty list to show the placeholder

# Run the Tkinter event loop
root.mainloop()