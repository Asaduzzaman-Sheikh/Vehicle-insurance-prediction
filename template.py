import os
from pathlib import Path

source_dir = 'src'

# Create all the directories, subdirectories, and files for the project
list_of_directories = [
    f'{source_dir}/__init__.py',
    f'{source_dir}/components/__init__.py',
    f'{source_dir}/components/data_ingestion.py',
    f'{source_dir}/components/data_validation.py',
    f'{source_dir}/components/data_transformation.py',
    f'{source_dir}/components/model_trainer.py',
    f'{source_dir}/components/model_evaluation.py',
    f'{source_dir}/components/model_pusher.py',
    f'{source_dir}/configuration/__init__.py',
    f'{source_dir}/configuration/mongo_db_connection.py',
    f'{source_dir}/configuration/aws_connection.py',
    f'{source_dir}/constants/__init__.py',
    f'{source_dir}/cloud_storage/__init__.py',
    f'{source_dir}/cloud_storage/aws_storage.py',
    f'{source_dir}/data_access/__init__.py',
    f'{source_dir}/data_access/project_data.py',
    f'{source_dir}/entity/__init__.py',
    f'{source_dir}/entity/config_entity.py',
    f'{source_dir}/entity/artifact_entity.py',
    f'{source_dir}/entity/estimator.py',
    f'{source_dir}/entity/s3_estimator.py',
    f'{source_dir}/exceptions/__init__.py',
    f'{source_dir}/logger/__init__.py',
    f'{source_dir}/pipeline/__init__.py',
    f'{source_dir}/pipeline/training_pipeline.py',
    f'{source_dir}/pipeline/prediction_pipeline.py',
    f'{source_dir}/utils/__init__.py',
    f'{source_dir}/utils/main_utils.py',
    
    'app.py',
    'config/model.yaml',
    'config/schema.yaml',
    'Dockerfile',
    '.dockerignore',
    'demo.py',
    'pyproject.toml',
    'requirements.txt',
    'setup.py'
]

for file_path in list_of_directories:
    # Create the directory if it doesn't exist
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != '':
        os.makedirs( file_dir, exist_ok = True )
        
    # Create the file if it doesn't exist
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w') as f:
            pass
    else:
        print(f"{file_path} already exists and is not empty. Skipping file creation.")