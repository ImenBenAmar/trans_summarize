import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "transcribtion_summarizer"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/transcriber/components/utils.py",
    f"src/{project_name}/transcriber/components/data_preptocessing.py",
    f"src/{project_name}/transcriber/components/whisper_model.py",
    f"src/{project_name}/transcriber/components/test.py",
    f"src/{project_name}/summarizer/components/utils.py",
    f"src/{project_name}/summarizer/components/data_preptocessing.py",
    f"src/{project_name}/summarizer/components/bartcnn_model.py",
    f"src/{project_name}/summarizer/components/test.py",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

   
    if not filepath.suffix:  
        continue

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
