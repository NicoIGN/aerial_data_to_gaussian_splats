########################################
# PERFORMANCE PROFILE - FAST ONLY
########################################

TRAINING_PROFILE="gpu/fast"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=0.25
MAX_RES=256
NUM_DOWNSCALES=2
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=200
TRAIN_RAYS_PER_BATCH=128

NUM_NERF_SAMPLES_PER_RAY=16
NUM_PROPOSAL_SAMPLES_PER_RAY="32 16"

########################################
# GAUSSIAN SPLATTING
########################################

DENSIFY_GRAD_THRESH=0.0008
CULL_ALPHA_THRESH=0.08
CULL_SCREEN_SIZE=0.35
SPLIT_SCREEN_SIZE=0.03

########################################
# EXPORT FAST
########################################

EXPORT_NUM_POINTS=150000
EXPORT_DOWNSAMPLE=3
