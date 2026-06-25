########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="splat/PCRS_lidar_geometry"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1
MAX_RES=4096
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

MAX_ITER=10000

# Split autorisé jusqu'aux 2/3 de l'entraînement
# → plus de temps pour combler les zones floues
STOP_SPLIT_AT=6000

########################################
# LEARNING RATES
########################################

POSITION_LR_INIT=0.000024
POSITION_LR_FINAL=0.0000008
POSITION_LR_DELAY_MULT=0.01
POSITION_LR_MAX_STEPS=25000    # ← corrigé : doit correspondre à MAX_ITER
                                #   sinon le LR tombe à FINAL trop tôt

FEATURE_LR=0.002
OPACITY_LR=0.02

SCALING_LR=0.002
ROTATION_LR=0.001

########################################
# GAUSSIAN SPLATTING — DENSIFICATION MODÉRÉE
########################################

# balanced=0.0002 → ici ×1.5 (au lieu de ×3 avant)
# Plus de splits déclenchés sur les zones sous-représentées
DENSIFY_GRAD_THRESH=0.0003

# Taille max pour déclencher un split
# Un peu plus permissif pour couvrir les surfaces larges
DENSIFY_SIZE_THRESH=0.02

########################################
# CLEANING / PRUNING
########################################

CULL_ALPHA_THRESH=0.08
CULL_SCREEN_SIZE=0.15
CULL_SCALE_THRESH=0.15

########################################
# DENSIFICATION CONTROL
########################################

REFINE_EVERY=300
RESET_ALPHA_EVERY=60

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=false
USE_SCALE_REGULARIZATION=true
SCALE_REG_WEIGHT=0.05

MAX_GAUSS_RATIO=6.0

SSIM_LAMBDA=0.2

########################################
# STABILISATION
########################################

MIXED_PRECISION=false
USE_GRAD_SCALER=false
ENABLE_COLLIDER=false

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
