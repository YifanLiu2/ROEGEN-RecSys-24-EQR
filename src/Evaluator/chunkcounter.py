import pickle

#file_path = 'embeddings/text-embedding-3-small/section/Accra_chunks.pkl'


import glob

# Path to the directory with pattern for specific file extension
pattern = 'embeddings/text-embedding-3-small/section/*_chunks.pkl'

# List to store files with the specific extension
filtered_files = glob.glob(pattern)

# Print the filtered files
# print(filtered_files)

number_of_chunks = []
chunk_lengths = []
for file_path in filtered_files:
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        number_of_chunks.append(len(data)) # list of strings
        # avg words per chunk:
        for chunk in data:
            chunk_lengths.append(len(chunk.split())) 
        # chunk_lengths.append(len(chunk.split()) for chunk in data)

    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except pickle.UnpicklingError:
        print("Error in unpickling the object.")
    except Exception as e:
        print(f"An error occurred: {e}")

# print(number_of_chunks)
# print(chunk_lengths)

import matplotlib.pyplot as plt

# Create histogram
plt.hist(number_of_chunks, bins=range(min(number_of_chunks), max(number_of_chunks) + 1), edgecolor='black')

# Add labels and title
plt.xlabel('Number of Chunks per city')
plt.ylabel('Frequency')
plt.title('Histogram')

# Show plot
plt.show()
plt.hist(chunk_lengths, bins=range(min(chunk_lengths), 25), edgecolor='black')

# Add labels and title
plt.xlabel('Length of Chunks in Dataset')
plt.ylabel('Frequency')
plt.title('Histogram')

# Show plot
plt.show()

