import os, re
from tqdm import tqdm

class DataProcessor:
    def __init__(self, raw_dest_dir: str, new_dest_dir: str = "data/clean_destination_air_canada_xml"):
        # check if raw file dir exist
        if not os.path.exists(raw_dest_dir):
            raise ValueError(f"Invalid directory path: {raw_dest_dir}")
        self.raw_dest_dir = raw_dest_dir

        # create new file dir
        os.makedirs(new_dest_dir, exist_ok=True)
        self.new_dest_dir = new_dest_dir
    
    def clean_text(self, text: str):
        text = re.sub(r'\{\{[^\}]+\}\}', '', text)  # remove markup
        text = re.sub(r'\[\[[^\|\]]+\|?[^\]]*\]\]', '', text)  # remove all wiki-style links
        text = re.sub(r'\[?\[([^]]+)\]\]?', r'\1', text)  # clean remaining brackets around URLs
        text = re.sub(r'==+([^=]+)==+', r'\1', text)  # remove header equal signs
        text = re.sub(r'<\/?[^>]+>', '', text)  # remove HTML tags
        text = re.sub(r'\{\{([^}]+)\}\}', '', text)  # remove curly braces
        text = re.sub(r'\'{2,}', '', text)  # remove multiple quotes
        text = re.sub(r'\n{3,}', '\n\n', text)  # reduce multiple newlines to exactly two
        text = re.sub(r'http[s]?://[^\s]+', '', text)  # remove URLs
        text = re.sub(r'[^\S\r\n]+', ' ', text)  # replace multiple spaces
        text = re.sub(r'\[\s+', '[', text)  # clean up space after opening bracket
        text = re.sub(r'\s+\]', ']', text)  # clean up space before closing bracket

        return text.strip()

    def process_files(self):
        # only keep txt file
        files = [f for f in os.listdir(self.raw_dest_dir) if f.endswith('.txt')]
        for filename in tqdm(files, desc="Cleaning Text Files"):
            if filename.endswith('.txt'):
                # process the raw file
                raw_file_path = os.path.join(self.raw_dest_dir, filename)
                with open(raw_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    cleaned_content = self.clean_text(content)

                # save to new file path
                new_file_path = os.path.join(self.new_dest_dir, filename)
                with open(new_file_path, 'w', encoding='utf-8') as file:
                    file.write(cleaned_content)

if __name__ == "__main__":
    dp = DataProcessor(raw_dest_dir="data/destination_air_canada_xml")
    dp.process_files()