# Rapport d'avancement de Mimo

Date de mise à jour : 15 octobre 2026 (Version grand public)

## Vision

Mimo a pour objectif d'observer fidèlement les mouvements d'un signeur à
partir d'une vidéo, avant tout retargeting vers un avatar. La priorité est la
qualité mesurable des landmarks du buste, des bras et des mains, puis leur
export dans un BVH inspectable dans Blender et vers des standards d'animation 3D (VRM Animation, Mixamo).

Le projet privilégie une chaîne hors ligne, reproductible, minimale et ouverte au grand public, centrée sur MediaPipe sans dépendances lourdes (PyTorch, MMPose, Anipose retirés).

## Architecture actuelle

```text
Vidéo (monoculaire)
  -> extraction MediaPipe (Holistic, Hybrid ou Pose+Hands)
  -> normalisation des frames
  -> filtrage et lissage temporel (OneEuro)
  -> conversion landmarks -> squelette BVH / VRMA
  -> retargeting et bake d'animation (.glb Mixamo, .vrm)
  -> rapport JSON de reconstruction
```

Les commandes publiques sont centralisées dans `mimo/main.py` :

```powershell
mimo bvh video.mp4 sortie.bvh --preset sign
mimo animate video.mp4 --model clara.glb -o sortie.glb --preset sign
mimo vrma video.mp4 sortie.vrma
mimo server
```

## Ce qui est bien fait

- Le pipeline BVH principal est indépendant et focalisé sur MediaPipe.
- Trois variantes MediaPipe sont disponibles :
  - `holistic` pour une extraction simple et rapide ;
  - `hybrid` pour le corps Holistic et les 21 points des mains dédiées Hand Landmarker ;
  - `pose_hands` pour `PoseLandmarker` et `HandLandmarker` avec timestamps partagés.
- Les mains utilisent les 21 landmarks nécessaires aux doigts.
- Le BVH supporte le haut du corps ou le corps complet.
- Export binaire officiel `.vrma` conforme `VRMC_vrm_animation` (glTF 2.0 / OpenXR Humanoid) avec 15 os par main.
- Retargeting BVH vers Mixamo avec recalage de la pose de repos (`align_rest`).
- Outils de bake d'animation directement dans les modèles `.glb` et `.vrm`.
- Suppression des dépendances expérimentales lourdes pour une installation simple et grand public (`pip install -e .`).
- Suite de tests unitaires (109 tests) à 100% de réussite.

## Ce qui était fragile et a été assaini

- Retrait de `backend/mixamo/` (l'ancienne chaîne directe vidéo->JSON Mixamo redondante avec `mimo bvh` + `mimo retarget`).
- Retrait du pipeline expérimental `parallel` (ViTPose/YOLOv8) et de ses dépendances volumineuses (PyTorch, MMPose, MMCV).
- Retrait de la triangulation multi-vues `anipose` et `multiview.py` (nécessitant étalonnage complexe et aniposelib).
- Simplification de `mimo animate` en mode monoculaire direct.
- Refonte de `mimo server` pour servir les fichiers BVH générés par le pipeline moderne.

## Analyse de Recherche : Retargeting VRM et Interpénétrations

### 1. Synthèse de l'Implémentation VRM 1.0 / VRMA
Le projet Mimo a validé une chaîne complète de conversion de la gestuelle signée (LSF/ASL) :
- **Solveur IK Anatomique Découplé** : Projection des 33 repères corporels et 21 points 3D par main en quaternions unitaires relatifs conformes au standard **VRM 1.0 / OpenXR Humanoid**.
- **Résolution Fine des Mains (15 os / main)** : Calcul de l'opposition tridimensionnelle du pouce (articulation trapézo-métacarpienne en selle) et décomposition flexion/abduction des 4 doigts longs.
- **Export Binaire Universel `.vrma`** : Serializer direct conforme à l'extension officielle `VRMC_vrm_animation` (Khronos glTF 2.0).

### 2. Le Phénomène d'Interpénétration des Doigts (Collisions)
L'observation expérimentale des animations générées met en évidence des croisements ou interpénétrations de surface lors de configurations serrées (dactylologie : lettres "A", "E", "S", "M", "O").

Ce phénomène est intrinsèque à la vision monoculaire filaire sans volume de chair.
Mimo applique déjà un **bornage biomécanique strict** (clipping) interdisant l'hyperextension vers l'arrière et limitant l'abduction latérale.

## Perspectives futures

1. Comparer quantitativement `holistic` et `hybrid` sur des gestes de dactylologie.
2. Implémenter une couche de relaxation géométrique post-solveur (PBD) pour les collisions de doigts.
3. Développer un visualiseur web Three.js / @pixiv/three-vrm intégré.
