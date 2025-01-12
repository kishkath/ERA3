import os
import sys

def get_unique_chars(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return vocab_size

def if_file_exists(file_name):
    for directory in sys.path[1:]:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            return file_path
    directory = sys.path[0]
    file_path = os.path.join(directory, file_name)
    if os.path.isfile(file_path):
        return file_path
    return None


def find_directory(directory_name):
    for directory in sys.path[1:]:
        if os.path.isdir(directory):
            subdirectories = os.listdir(directory)
            if directory_name in subdirectories:
                return os.path.join(directory, directory_name)
    directory = sys.path[0]
    if os.path.isdir(directory):
        subdirectories = os.listdir(directory)
        if directory_name in subdirectories:
            return os.path.join(directory, directory_name)
    return None
