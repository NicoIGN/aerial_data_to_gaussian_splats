########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="splat/PCRS_diag_no_prune"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=10000

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

MAX_ITER=6000

# Pas de split
STOP_SPLIT_AT=0

########################################
# GAUSSIAN SPLATTING
########################################

# Pas de densification utile dans ce test
DENSIFY_GRAD_THRESH=1000000000

########################################
# CLEANING / PRUNING
########################################

# Très permissif : on évite de tuer les GS trop vite
CULL_ALPHA_THRESH=0.0000001

CULL_SCREEN_SIZE=0.15

CULL_SCALE_THRESH=0.5

########################################
# DENSIFICATION CONTROL
########################################

# Pas de refine périodique
REFINE_EVERY=0

# Pas de reset alpha pendant ce test court
RESET_ALPHA_EVERY=999999

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=False
USE_SCALE_REGULARIZATION=False
MAX_GAUSS_RATIO=1000.0

########################################
# COLLIDER
########################################

# Très important pour ce test
ENABLE_COLLIDER=False

########################################
# TRAINING STABILITY
########################################

MIXED_PRECISION=False
USE_GRAD_SCALER=False

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
