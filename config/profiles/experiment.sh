########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="splat/experiment"

DEVICE="gpu"
MODEL="splatfacto-big"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# Stable full-res training
CAMERA_RES_SCALE_FACTOR=1.0

# Garde une résolution raisonnable
MAX_RES=1280

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

# Stable long training
MAX_ITER=8000

# IMPORTANT :
# évite la densification tardive explosive
STOP_SPLIT_AT=6000

# Stable gradients
# TRAIN_RAYS_PER_BATCH=1024

# Bon compromis qualité/stabilité
# NUM_NERF_SAMPLES_PER_RAY=48
# NUM_PROPOSAL_SAMPLES_PER_RAY="128 128"

########################################
# GAUSSIAN SPLATTING
########################################

# Densification plus conservative
DENSIFY_GRAD_THRESH=0.0008

########################################
# CLEANING
########################################

# Nettoyage alpha un peu plus agressif
CULL_ALPHA_THRESH=0.1

# Évite gros splats écran
CULL_SCREEN_SIZE=0.15
# SPLIT_SCREEN_SIZE=0.05

########################################
# DENSIFICATION CONTROL
########################################

# Beaucoup plus stable à long terme
REFINE_EVERY=200

# PARAMÈTRE CRITIQUE
# évite saturation alpha / écran blanc
RESET_ALPHA_EVERY=30

# Supprime davantage de gros splats instables
# CULL_SCALE_THRESH=0.5

########################################
# QUALITY / REGULARIZATION
########################################

# Améliore la stabilité visuelle en corrigeant les variations de couleur locales
USE_BILATERAL_GRID=False

# Désactive la régularisation des échelles des gaussiennes (plus de liberté mais moins de contraintes)
USE_SCALE_REGULARIZATION=False

# Limite la taille des covariances pour éviter des splats trop étalés
MAX_GAUSS_RATIO=5.0

# Équilibre entre fidélité visuelle et préservation de la structure de l’image
# SSIM_LAMBDA=0.2

########################################
# TRAINING STABILITY (GPU OPTIMIZATION DISABLED)
########################################

#Désactive la précision mixte (FP16), entraînement plus lent mais plus stable numériquement
MIXED_PRECISION=False

#Désactive le scaling des gradients utilisé avec la précision mixte
USE_GRAD_SCALER=False

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
