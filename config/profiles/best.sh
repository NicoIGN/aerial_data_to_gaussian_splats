########################################
# PERFORMANCE PROFILE - QUALITY++ REAL
########################################

TRAINING_PROFILE="gpu/quality_pp_real"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=1536

NUM_DOWNSCALES=0
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=18000

# ⚡ clé : ton 400 est beaucoup trop lent
REFINE_EVERY=180

TRAIN_RAYS_PER_BATCH=768

NUM_NERF_SAMPLES_PER_RAY=80
NUM_PROPOSAL_SAMPLES_PER_RAY="160 80"

########################################
# GAUSSIAN SPLATTING (DETAIL DRIVEN)
########################################

# 🔥 énorme correction
DENSIFY_GRAD_THRESH=0.0011

# garde le détail vivant
CULL_ALPHA_THRESH=0.08

# évite blobs sans tuer micro-structure
CULL_SCREEN_SIZE=0.24

# split adapté à 1536px
SPLIT_SCREEN_SIZE=0.016

# laisse le modèle apprendre avant de figer
STOP_SPLIT_AT=15000

########################################
# STABILITY CONTROL
########################################

RESET_ALPHA_EVERY=45
CULL_SCALE_THRESH=0.45
MAX_GAUSS_RATIO=4.0

########################################
# QUALITY
########################################

SSIM_LAMBDA=0.28
USE_SCALE_REGULARIZATION=False

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=2500000
EXPORT_DOWNSAMPLE=1
