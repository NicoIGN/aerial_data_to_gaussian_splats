############################
# PIPELINE CONTROL (SKIP FLAGS)
############################

SKIP_CONDA_UPDATE=true
SKIP_FRAME_EXTRACTION=false
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_EXPORT=false

#NO_PROXY=true


############################
# CONDA
############################

CONDA_ENV_FILE="environment/conda_colab.yml"
CONDA_ENV_NAME="gsplat"


############################
# PROXY
############################

if [ "$NO_PROXY" != true ]; then
  export HTTP_PROXY="http://proxy.ign.fr:3128"
  export HTTPS_PROXY="http://proxy.ign.fr:3128"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"

  echo "🌐 Proxy enabled"
else
  echo "🚫 Proxy disabled (NO_PROXY=true)"
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset http_proxy
  unset https_proxy
fi


############################
# PIPELINE EXECUTION ENV
############################
if [ -z "${ROOT_DIR+x}" ]; then
    ROOT_DIR="runs/default"
fi

export SCENE_NAME="scene3d"



INPUT_DIR=""
FRAME_DIR=""
ORI_DIR=""
OUTPUT_DIR=""
EXPORT_DIR=""


############################
# DEVICE MODE
############################

# If DEVICE is already defined in environment, keep it
if [ -z "${DEVICE+x}" ]; then
  DEVICE="cpu"   # cpu | gpu
fi

export DEVICE

# GPU REQUIREMENTS (important)
# - NVIDIA GPU + CUDA drivers
# - PyTorch compiled with CUDA
# - required for:
#   - splatfacto / splatfacto-w
#   - instant-ngp
#   - zip-nerf
#   - pynerf
#   - feature-splatting
#   - tetra-nerf (partially GPU)

# CPU MODE LIMITATIONS
# - no CUDA kernels
# - slower dataloading + training
# - recommended models:
#   - nerfacto (BEST CPU CHOICE)
#   - nerf
#   - kplanes (slow)
#   - tensorf (slow but works)


############################
# MODEL CONFIG
############################
EXPERIMENT_NAME="model3d"

if [ -z "${MODEL+x}" ]; then
  MODEL="nerfacto"
fi



# Nerfstudio backend implementation
if [ -z "${MODEL_IMPLEMENTATION+x}" ]; then
    MODEL_IMPLEMENTATION="torch"
# torch  -> CPU / safe fallback
# tcnn   -> GPU ONLY (tiny-cuda-nn required)
fi

############################
# VIDEO PIPELINE
############################

VIDEO_NAME="${VIDEO_NAME:-video.mov}"

if [[ -z "${FPS+x}" && -z "${NUM_FRAMES+x}" ]]; then
    FPS=10
fi

############################
# COLMAP / PREPROCESS
############################

if [ -z "${SFMT_TOOL+x}" ]; then
    SFMT_TOOL="hloc" # colmap | hloc | any
    # → hloc = meilleur pour scènes difficiles / moins d’artefacts
fi

if [ -z "${MATCHING_METHOD+x}" ]; then
    MATCHING_METHOD="exhaustive"
    # sequential | vocab_tree | exhaustive
    # → exhaustive = plus précis mais lent
    # → vocab_tree = bon compromis
    # → sequential = vidéo uniquement
fi

if [ -z "${NUM_DOWNSCALES+x}" ]; then
    NUM_DOWNSCALES=0
    # 0 | 1 | 2 | 3
    # → plus haut = moins de détails mais plus stable
fi

if [ -z "${SKIP_IMAGE_PROCESSING+x}" ]; then
    SKIP_IMAGE_PROCESSING=true
    # true | false
    # → évite resize/copie images si déjà préparées
fi

if [ -z "${CAMERA_TYPE+x}" ]; then
    CAMERA_TYPE="perspective"
    # perspective | pinhole | fisheye | equirectangular
    # → mauvais choix = déformations COLMAP
fi

############################
# FEATURE / MATCHING
############################

if [ -z "${FEATURE_TYPE+x}" ]; then
    FEATURE_TYPE="any"
    # sift | superpoint | superpoint_aachen | disk | r2d2 | any
    # → sift = COLMAP pur (robuste mais limité sur scènes peu texturées)
    # → superpoint / disk = meilleurs pour scènes difficiles (indoor, faible texture)
    # → disk souvent plus performant sur surfaces pauvres (table, objets)
    # ⚠️ superpoint/disk nécessitent HLOC (pas compatibles COLMAP pur)
fi

if [ -z "${MATCHER_TYPE+x}" ]; then
    MATCHER_TYPE="any"
    # NN | NN-mutual | superglue | superglue-fast | lightglue | disk+lightglue | any
    # → NN / NN-mutual = matching classique (COLMAP pur)
    # → superglue = très précis mais plus lent (HLOC)
    # → lightglue = plus rapide et souvent plus robuste (recommandé)
    # → disk+lightglue = combo très performant pour scènes peu texturées
    # ⚠️ superglue / lightglue nécessitent HLOC (pas compatibles COLMAP pur)
fi

############################
# CAMERA / STRUCTURE
############################

if [ -z "${USE_SFM_DEPTH+x}" ]; then
    USE_SFM_DEPTH=false
    # true | false
    # → depth SfM utile pour densification mais plus lourd
fi

if [ -z "${REFINE_INTRINSICS+x}" ]; then
    REFINE_INTRINSICS=true
    # true | false
    # → améliore calibration caméra
fi

if [ -z "${USE_SINGLE_CAMERA_MODE+x}" ]; then
    USE_SINGLE_CAMERA_MODE=true
    # true | false
    # → true si une seule caméra
fi

############################
# CROPPING / FILTERING
############################

if [ -z "${PERCENT_RADIUS_CROP+x}" ]; then
    PERCENT_RADIUS_CROP=0.95
    # 0.0 → 1.0
    # → supprime les bords parasites
fi

if [ -z "${CROP_FACTOR+x}" ]; then
    CROP_FACTOR="0.02 0.05 0.02 0.02"
    # top bottom left right (0-1)
    # → crop manuel zones instables
fi

############################
# IMAGE PREPROCESSING
############################

if [ -z "${CAMERA_RES_SCALE_FACTOR+x}" ]; then
    CAMERA_RES_SCALE_FACTOR=0.5
    #CAMERA_RES_SCALE_FACTOR=0.75  # 0.5 = FAST MODE (~4x speedup)
fi

############################
# TRAINING PARAMETERS (NERF CORE)
############################

if [ -z "${STEPS_PER_SAVE+x}" ]; then
  STEPS_PER_SAVE=5000
fi

if [ -z "${STEPS_PER_EVAL_ALL_IMAGES+x}" ]; then
  STEPS_PER_EVAL_ALL_IMAGES=2000
fi

if [ -z "${REFINE_EVERY+x}" ]; then
  REFINE_EVERY=100
fi

# Nombre total d’itérations d’entraînement
# → 1 itération = optimisation sur un batch de rayons
# ↑ augmente la qualité mais augmente le temps de calcul
if [ -z "${MAX_ITER+x}" ]; then
  MAX_ITER=3000
fi

if [ -z "${MAX_JOBS+x}" ]; then
  MAX_JOBS=2
fi


# Nombre de rayons (pixels simulés) traités par batch
# → contrôle la stabilité et la mémoire utilisée
# ↑ plus grand = plus stable mais plus lent
if [ -z "${TRAIN_RAYS_PER_BATCH+x}" ]; then
    # TRAIN_RAYS_PER_BATCH=1024
    # TRAIN_RAYS_PER_BATCH=512
    # TRAIN_RAYS_PER_BATCH=256
    unset TRAIN_RAYS_PER_BATCH
fi


############################
# SAMPLING (RECONSTRUCTION 3D)
############################

# Nombre de points échantillonnés par rayon caméra
# → chaque rayon est "découpé" en 3D pour estimer couleur + densité
# ↑ plus élevé = détails plus fins mais calcul plus lourd
if [ -z "${NUM_NERF_SAMPLES_PER_RAY+x}" ]; then
    # NUM_NERF_SAMPLES_PER_RAY=32
    unset NUM_NERF_SAMPLES_PER_RAY
fi

# Échantillonnage en 2 étapes (proposal network)
# 1er nombre : exploration grossière (zones importantes)
# 2e nombre : raffinement des zones sélectionnées
# → améliore qualité et efficacité du rendu
if [ -z "${NUM_PROPOSAL_SAMPLES_PER_RAY+x}" ]; then
    # NUM_PROPOSAL_SAMPLES_PER_RAY="160 64"
    # NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"
    unset NUM_PROPOSAL_SAMPLES_PER_RAY
fi


############################
# GAUSSIAN SPLATTING (DENSIFICATION & PRUNING)
############################

# Seuil de gradient pour la densification des gaussiennes
# → contrôle quand de nouvelles gaussiennes sont ajoutées
# ↑ plus bas = plus de détails, mais plus de bruit et mémoire
if [ -z "${DENSIFY_GRAD_THRESH+x}" ]; then
 #  DENSIFY_GRAD_THRESH=0.0004
    unset DENSIFY_GRAD_THRESH
fi

# Seuil alpha pour supprimer les gaussiennes faibles
# → enlève les éléments peu visibles / inutiles
# ↑ plus haut = scène plus propre mais perte de détails fins
if [ -z "${CULL_ALPHA_THRESH+x}" ]; then
  # CULL_ALPHA_THRESH=0.05
    unset CULL_ALPHA_THRESH
fi

# Taille écran pour culling (élimination des petites contributions)
# → supprime les splats trop petits à l’écran
# ↑ plus grand = plus agressif, moins de détails éloignés
if [ -z "${CULL_SCREEN_SIZE+x}" ]; then
  # CULL_SCREEN_SIZE=0.3
    unset CULL_SCREEN_SIZE
fi

# Taille écran pour split (division des gaussiennes)
# → contrôle quand une gaussienne est divisée en plusieurs
# ↑ plus bas = plus de précision locale, mais plus de splats
if [ -z "${SPLIT_SCREEN_SIZE+x}" ]; then
  # SPLIT_SCREEN_SIZE=0.02
    unset SPLIT_SCREEN_SIZE
fi

# Ratio max entre les axes d’un gaussien (anisotropie)
# → contrôle à quel point une gaussienne peut être allongée
# ↓ plus bas = formes plus compactes, meilleure stabilité géométrique
# ↑ plus haut = plus flexible mais risque de “spaghettis” et artefacts
if [ -z "${MAX_GAUSS_RATIO+x}" ]; then
  # MAX_GAUSS_RATIO=5
    unset MAX_GAUSS_RATIO
fi

# Stoppe la croissance des splats après une phase stable
# → empêche explosion tardive (> 4M)
#if [ -z "${STOP_SPLIT_AT+x}" ]; then
#  STOP_SPLIT_AT=8000
#fi

# Réinitialisation périodique des alphas
# → évite accumulation de splats semi-actifs parasites
if [ -z "${RESET_ALPHA_EVERY+x}" ]; then
  # RESET_ALPHA_EVERY=30
    unset RESET_ALPHA_EVERY
fi

# Régularisation des scales (évite blobs et étirements)
# → rend les gaussiennes plus homogènes et stables
if [ -z "${USE_SCALE_REGULARIZATION+x}" ]; then
  # USE_SCALE_REGULARIZATION=true
    unset USE_SCALE_REGULARIZATION
fi

if [ -z "${USE_BILATERAL_GRID+x}" ]; then
  # USE_BILATERAL_GRID=True
    unset USE_BILATERAL_GRID
fi

# Limite la taille relative des gaussiennes
# → évite structures trop dominantes instables
if [ -z "${CULL_SCALE_THRESH+x}" ]; then
 # CULL_SCALE_THRESH=0.5
    unset CULL_SCALE_THRESH
fi

# SSIM loss (stabilité perceptuelle + réduction bruit)
# → améliore cohérence visuelle globale
if [ -z "${SSIM_LAMBDA+x}" ]; then
  # SSIM_LAMBDA=0.2
    unset SSIM_LAMBDA
fi

############################
# IMAGE / DATA RESOLUTION
############################

# Résolution maximale utilisée pendant l’entraînement
# → les images peuvent être downscalées automatiquement
# ↑ plus élevé = plus de détails mais plus lent et plus gourmand
if [ -z "${MAX_RES+x}" ]; then
    # MAX_RES=1024
    # MAX_RES=512
    unset MAX_RES
fi

############################
# NERF ADVANCED SETTINGS
############################

CAMERA_MODE="off"


############################
# TRAINING MODE (VISUALISATION)
############################

# Contrôle l’interface de visualisation pendant le training
# options possibles :
#   viewer               → interface web interactive (par défaut)
#   tensorboard          → logs only (RECOMMANDÉ pour pipeline automatisé)
#   comet                → tracking expérimental
#   wandb                → tracking cloud
#   viewer+tensorboard   → hybride (debug + logs)
#   viewer+wandb         → hybride
#   viewer+comet        → hybride
#   viewer_beta         → version expérimentale viewer
#
# ⚠️ IMPORTANT :
# - "viewer" pour voir le resultat dans Nerfstudio
# - "tensorboard" pour mode headless réel (batch / scripts)
if [ -z "${TRAIN_VIS_MODE+x}" ]; then
    TRAIN_VIS_MODE="tensorboard"
    # TRAIN_VIS_MODE="viewer"
fi

############################
# EXPORT CONFIG
############################
# normales

NORMAL_METHOD="open3d"
REMOVE_OUTLIERS=True
