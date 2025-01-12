# dataset
import os

import pandas as pd

import utils
from utils import find_directory


def get_data_files(data_path=find_directory("data")):
    data_files = []
    for root, dir, files in os.walk(data_path):
        for file in files:
            data_files.append(root + "/" + file)
    print(">>> files: ", data_files)
    return data_files


def display_data(df):
    return df.head(3)


def prepare_data():
    try:
        # if utils.if_file_exists("telugu.txt"):
        #     return
        # Load CSV files
        file_paths = get_data_files()
        books_data = pd.read_csv(file_paths[0], nrows=1000)
        news_data = pd.read_csv(file_paths[2])

        # Collect text data
        telugu_data = []
        telugu_data.extend(books_data.get("text", []).dropna().values)  # Handle missing columns or NaN
        # telugu_data.extend(news_data.get("body", []).dropna().values)  # Handle missing columns or NaN

        print(f"Total data lines collected: {len(telugu_data)}")

        # Write the cleaned data to a text file
        with open("telugu.txt", "w", encoding="utf-8") as file:
            for line in telugu_data:
                try:
                    # Validate the line and ensure it's a string
                    assert isinstance(line, str), f"Expected str, got {type(line)}"
                    file.writelines(line.strip() + "\n")  # Write only cleaned strings
                except AssertionError as ae:
                    print(f"Skipping invalid line: {line} ({ae})")
                except Exception as inner_error:
                    print(f"Unexpected error for line: {line} ({inner_error})")

    except FileNotFoundError as fnfe:
        print(f"File not found: {fnfe}")
    except pd.errors.EmptyDataError as ede:
        print(f"Empty CSV file: {ede}")
    except Exception as error:
        print(f"Error occurred: {error}")


def is_telugu_text(text):
    # Ensure the text is a UTF-8 string
    if not isinstance(text, str):
        try:
            text = text.decode('utf-8')  # Decode if it's bytes
        except (UnicodeDecodeError, AttributeError):
            return False  # Invalid UTF-8 or non-string input

    # Define Telugu character pattern using Unicode range
    telugu_pattern = re.compile(r'[\u0C00-\u0C7F]')

    # Check the first 100 characters for efficiency
    text_sample = text[:100]  # Slice first 100 characters
    telugu_chars = len(telugu_pattern.findall(text_sample))  # Count Telugu characters

    # Calculate the percentage of Telugu characters
    return telugu_chars / (len(text_sample) + 1) > 0.6  # 60% threshold


import re


def get_clean_data():
    try:
        # Read the text file
        with open("telugu.txt", "r", encoding='utf-8') as file:
            framed_text_data = file.readlines()

        # Combine all lines into a single text string
        text = ''.join(framed_text_data)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove citations [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)

        # Remove digits
        text = re.sub(r'\d+', '', text)

        # Remove English characters (a-z, A-Z)
        text = re.sub(r'[a-zA-Z]', '', text)

        # Remove special characters but keep Telugu ones
        text = re.sub(r'[^\u0C00-\u0C7F\s\.,]', '', text)

        # Additional cleanups
        # text = re.sub(r'[^\u0C00-\u0C7F@#$%]', '', text)
        text = re.sub(r'[\r\n\xa0]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Write cleaned text to a new text file
        with open("cleaned_text.txt", "w", encoding="utf-8") as output_file:
            output_file.write(text.strip())

        print("Cleaning and saving completed successfully!")
        return text.strip()

    except Exception as error:
        print(f"An error occurred: {error}")
        return ""


if __name__ == "__main__":
    input_files_path = '/data'
    data_files = get_data_files(input_files_path)
