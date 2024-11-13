import os
from pathlib import Path

project_name = "books_recommender"

list_of_files = [

    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/stage_00_data_ingestion.py",  
    f"{project_name}/components/stage_01_data_validation.py",
    f"{project_name}/components/stage_02_data_transformation.py",
    f"{project_name}/components/stage_03_model_trainer.py",

    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    
    f"{project_name}/constant/__init__.py",

    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",

    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception_handler.py",

    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/log.py",

    f"{project_name}/pipline/__init__.py",
    f"{project_name}/pipline/training_pipeline.py",

    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/util.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")

 