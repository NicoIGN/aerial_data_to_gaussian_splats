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

CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=1024
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

MAX_ITER=6000

# Pas de split après l'init LiDAR — on ne veut pas de nouveaux splats
STOP_SPLIT_AT=0

########################################
# LEARNING RATES
# Principe : position figée, couleur libre, forme contrainte
########################################

# 🔒 Position quasi-figée : les splats restent sur les points LiDAR
# (valeur typique balanced ~0.00016, on divise par ~20)
POSITION_LR_INIT=0.000008
POSITION_LR_FINAL=0.0000008
POSITION_LR_DELAY_MULT=0.01
POSITION_LR_MAX_STEPS=6000

# 🎨 Couleur/opacité : libre pour s'adapter aux images
FEATURE_LR=0.005
OPACITY_LR=0.05

# 📐 Forme : très contrainte (évite les antennes)
# (valeur typique ~0.005, on divise par 5)
SCALING_LR=0.001
ROTATION_LR=0.0005

########################################
# GAUSSIAN SPLATTING
########################################

# Pas de densification (on part du LiDAR dense)
DENSIFY_GRAD_THRESH=999999999

########################################
# CLEANING / PRUNING
########################################

# Pruning très permissif : on ne veut pas perdre les points LiDAR
CULL_ALPHA_THRESH=0.005
CULL_SCREEN_SIZE=0.15

# 🔑 CLÉ ANTI-ANTENNES : supprime les splats dont
# scale_max > CULL_SCALE_THRESH × scene_extent
# Mettre bas pour tuer les filaments
CULL_SCALE_THRESH=0.1

########################################
# DENSIFICATION CONTROL
########################################

# Refine peu fréquent (juste pour le pruning alpha, pas de split)
REFINE_EVERY=500
RESET_ALPHA_EVERY=999999      # pas de reset (stabilité)

########################################
# QUALITY / REGULARIZATION — CLÉ DU PROFIL
########################################

USE_BILATERAL_GRID=false
USE_SCALE_REGULARIZATION=true  # 🔑 pénalise les splats allongés

# 🔑 ANTI-ANTENNES : ratio max entre la plus grande
# et la plus petite dimension d'un splat
# balanced=10, ici on met 3 → splats quasi-sphériques/disques
MAX_GAUSS_RATIO=3.0

# Régularisation scale : poids de la pénalité sur l'allongement
# (si supporté par ton build nerfstudio)
SCALE_REG_WEIGHT=0.1

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
