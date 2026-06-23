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

# Densification autorisée jusqu'à mi-parcours seulement
# → les splats s'adaptent en début d'entraînement,
#   puis la géométrie se fige pour les dernières itérations
STOP_SPLIT_AT=3000

########################################
# LEARNING RATES
# Principe : petits déplacements autorisés en début,
#            puis décroissance vers quasi-zéro
########################################

# 🔒 Position : déplacements limités
# balanced=0.00016 → ici ×0.15 = léger mais non nul
# La décroissance exponentielle vers FINAL fige
# progressivement la géométrie en fin d'entraînement
POSITION_LR_INIT=0.000024
POSITION_LR_FINAL=0.0000008    # ~×30 plus bas que INIT → quasi-figé à la fin
POSITION_LR_DELAY_MULT=0.01
POSITION_LR_MAX_STEPS=6000

# 🎨 Couleur/opacité : libre
FEATURE_LR=0.002
OPACITY_LR=0.02

# 📐 Forme : contrainte mais pas figée
# Les splats peuvent s'aplatir sur la surface (disque tangent)
# mais pas se transformer en aiguilles
SCALING_LR=0.002
ROTATION_LR=0.001

########################################
# GAUSSIAN SPLATTING — DENSIFICATION LÉGÈRE
########################################

# balanced=0.0002 → ici ×3 = densification peu déclenchée
# Seuls les splats avec un gradient fort (zones floues) splittent
DENSIFY_GRAD_THRESH=0.0006

# Taille max pour déclencher un split (évite d'exploser les gros splats)
DENSIFY_SIZE_THRESH=0.01

########################################
# CLEANING / PRUNING — ACTIF MAIS MODÉRÉ
########################################

# Pruning actif : supprime les splats fantômes sans tuer les LiDAR utiles
CULL_ALPHA_THRESH=0.05

# Screen-space : supprime les splats qui couvrent trop d'écran
# (signe d'un splat mal placé ou trop grand)
CULL_SCREEN_SIZE=0.15

# Scale absolue : supprime les splats trop grands
# (antennes longues, splats qui ont explosé)
CULL_SCALE_THRESH=0.15

########################################
# DENSIFICATION CONTROL
########################################

# Refine fréquent en début pour laisser la géométrie s'adapter,
# puis le STOP_SPLIT_AT coupe la densification à mi-parcours
REFINE_EVERY=300

# Reset alpha périodique : force le réseau à re-justifier
# chaque splat → évite la saturation d'opacité
RESET_ALPHA_EVERY=60

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=false

# 🔑 Régularisation scale active : pénalise mathématiquement
# les splats allongés à chaque step → frein continu aux antennes
USE_SCALE_REGULARIZATION=true
SCALE_REG_WEIGHT=0.05          # modéré : assez pour freiner, pas assez pour bloquer

# 🔑 Ratio max forme : disque aplati OK (3:1), aiguille interdit
# Un peu plus permissif que la version figée pour laisser
# les splats couvrir les surfaces obliques
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
