########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="gpu/highres-balanced"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# Plus haute résolution que ton profil balanced, sans aller full 1.0
CAMERA_RES_SCALE_FACTOR=1
MAX_RES=5000

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

# Un peu plus long pour converger en plus haute résolution
MAX_ITER=9000

# Stop split avant la fin pour éviter l’explosion tardive
STOP_SPLIT_AT=7000

# Batch un peu plus haut que 512 si VRAM ok
TRAIN_RAYS_PER_BATCH=768

# Signal de densification correct sans trop gonfler
NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# GAUSSIAN SPLATTING (CONTROLLED GROWTH)
########################################

# Légèrement moins agressif que 0.00045 (=> moins de nouveaux splats)
DENSIFY_GRAD_THRESH=0.00055

# Nettoyage un peu strict
CULL_ALPHA_THRESH=0.12

# Contrôle écran : limite les gros splats et les splits super fins
CULL_SCREEN_SIZE=0.22
SPLIT_SCREEN_SIZE=0.03

# Raffinement modéré (pas trop fréquent)
REFINE_EVERY=220

# Stabilisation
RESET_ALPHA_EVERY=35
CULL_SCALE_THRESH=0.5

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=4.5
SSIM_LAMBDA=0.22

########################################
# TRAINING STABILITY
########################################

MIXED_PRECISION=True
USE_GRAD_SCALER=True

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
