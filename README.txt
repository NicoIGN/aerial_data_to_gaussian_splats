Convertir l'aero en données colmap:
python python/xml_laz_to_colmap.py --xml $PCRS_HOME/aero/23FD1805A_adjust.XML --laz $PCRS_HOME/roofer/lidar_subset.laz --images $PCRS_HOME/images --out $PCRS_HOME/ori --image-factor 4 --axis-convention rot_cw_90,flip_yz


Visualiser le résultat de COLMAP
python python/photogrammetry_inspector.py --colmap-dir $PCRS_HOME/ori 
