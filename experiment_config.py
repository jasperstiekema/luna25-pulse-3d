#experiment_config.py
from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        # self.WORKDIR = Path("/home/constantin/Work/LUNA25/luna25-pulse-3d-v3")
        self.WORKDIR = Path("/opt/app/")
       
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("../dataset/luna25_nodule_blocks")
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("../dataset/luna25_csv")
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "LUNA25-pulse"
        self.MODE = "3D" # 2D or 3D

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 6
        self.SIZE_MM = 50
        # self.SIZE_MM = 70

        self.SIZE_PX = 64
        # self.SIZE_PX = 64
        # self.SIZE_PX = 96

        self.BATCH_SIZE = 16
        self.ROTATION = ((-10, 10), (-10, 10), (-10, 10))
        self.TRANSLATION = True
        self.EPOCHS = 100
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        # self.LEARNING_RATE = 1e-5
        # self.LEARNING_RATE = 1e-6
        # self.LEARNING_RATE = 1e-7
        # self.WEIGHT_DECAY = 5e-4
        # self.WEIGHT_DECAY = 5e-4
        self.WEIGHT_DECAY = 1e-5

        self.HARD_MINING = False   # full-batch training again
        self.HARD_MINING_RATIO = 0.6


config = Configuration()
