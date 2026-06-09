import os
from dotenv import load_dotenv
load_dotenv('.env')
print(os.getenv("MLFLOW_TRACKING_URI"))