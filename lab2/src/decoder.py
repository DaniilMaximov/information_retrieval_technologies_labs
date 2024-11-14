import bz2
import os
import re


class BZ2Decoder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def decode_file(self, file_name):
        file_path = os.path.join(self.input_path, file_name)
        try:
            with open(file_path, 'rb') as f:
                compressed_data = f.read()

            decompressed_data = bz2.decompress(compressed_data)
            decoded_content = decompressed_data.decode('cp1251')

            # Remove unnecessary metadata
            cleaned_content = re.sub(
                r'<collection-description>.*?</collection-description>',
                '',
                decoded_content,
                flags=re.DOTALL
            )

            output_file_name = file_name.replace('.bz2', '.xml')
            output_file_path = os.path.join(self.output_path, output_file_name)

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_content)

            print(f"File {output_file_name} successfully decoded and cleaned.")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    def decode_files(self):
        files = [f for f in os.listdir(self.input_path) if f.endswith('.bz2')]
        for file_name in files[:3]:  # Limit to first 3 files
            self.decode_file(file_name)
