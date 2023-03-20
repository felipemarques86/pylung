import os
import re

root_dir = "weights/"
pattern = re.compile(r'\\')

unique_paths = set()

for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = pattern.sub('$', os.path.relpath(file_path, root_dir))
        file_name, file_extension = os.path.splitext(file)
        unique_paths.add(relative_path.replace(file_extension, ''))
