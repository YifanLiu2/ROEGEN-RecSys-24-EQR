import os
# save a list of all the data files in the data directory
data_dir = "data/wikivoyage_data"
data_files = os.listdir(data_dir)
# remove the .txt extension
data_files = [file[:-4] for file in data_files]
# save the list to a file with utf-8 encoding
with open("data_files.txt", "w", encoding='utf-8') as f:
    for file in data_files:
        f.write(file + "\n")
