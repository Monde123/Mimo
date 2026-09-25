# Rapport d'avancement de Mimo

Date de mise à jour : 22 septembre 2026

## Vision

Mimo a pour objectif d'observer fidèlement les mouvements d'un signeur à
partir d'une vidéo, avant tout retargeting vers un avatar. La priorité est la
qualité mesurable des landmarks du buste, des bras et des mains, puis leur
export dans un BVH inspectable dans Blender.

Le projet privilégie une chaîne hors ligne, reproductible et comparable. Le
serveur live et le retargeting Mixamo restent secondaires.

## Architecture actuelle

```text
Vidéo
  -> extraction (Holistic, Hybrid ou Parallel)
  -> normalisation des frames
  -> filtrage et lissage temporel
  -> conversion landmarks -> squelette BVH
  -> rapport JSON de reconstruction
```

Les commandes publiques sont centralisées dans `mimo/main.py` :

```powershell
mimo bvh video.mp4 sortie.bvh --preset sign
mimo mixamo video.mp4 sortie.json
mimo server
```

## Ce qui est bien fait

- Le pipeline BVH principal est indépendant de Mixamo.
- Deux variantes MediaPipe sont disponibles :
  - `holistic` pour une extraction simple ;
  - `hybrid` pour le corps Holistic et les mains dédiées MediaPipe.
- Une troisième variante MediaPipe dédiée, `pose_hands`, utilise
  `PoseLandmarker` et `HandLandmarker` avec des timestamps partagés.
- Les mains utilisent les 21 landmarks nécessaires aux doigts.
- Le BVH supporte le haut du corps ou le corps complet.
- Le bassin, les jambes, les chevilles, les talons et les extrémités des pieds
  sont représentés lorsque `--body full` est utilisé et que les points sont
  visibles.
- La tête possède une hiérarchie BVH corrigée avec une extrémité verticale.
- Les sorties sont accompagnées d'un rapport `*.report.json`.
- Les poids lourds YOLOv8 et ViTPose sont locaux et ignorés par Git.
- Le checkpoint `vitpose-s-coco_25.pth` est associé à
  `ViTPose_coco_25.py`, donc à une sortie de 25 keypoints.
- Les tests unitaires couvrent les mains, le BVH, la qualité et la découverte
  de la configuration parallèle.

## Ce qui était mal fait ou reste fragile

- Les anciennes commandes documentées utilisaient directement des modules
  internes ; elles sont remplacées progressivement par `mimo`.
- Le pipeline ViTPose dépend de PyTorch, MMPose, MMCV et Ultralytics, qui ne
  sont pas des dépendances de base.
- L'inférence parallèle n'a pas encore été validée sur une vidéo réelle dans
  l'environnement courant.
- ViTPose produit des coordonnées 2D ; la profondeur BVH est donc estimée ou
  neutralisée, contrairement aux landmarks monde de MediaPipe.
- La fusion du corps ViTPose et des mains MediaPipe doit être comparée
  quantitativement à la pipeline Hybrid.
- Les tests historiques Mixamo présentent encore des divergences et ne
  doivent pas être confondus avec la validation BVH.
- Le serveur Flask est conservé comme fonctionnalité optionnelle et ne fait
  pas partie du flux de mesure principal.

## État de la méthode parallèle

Structure attendue :

```text
backend/models/parallel/
  yolov8l.pt
  vitpose-s-coco_25.pth
backend/vitPose/
  ViTPose_common.py
  ViTPose_coco_25.py
```

Commande :

```powershell
mimo bvh input.mp4 output_parallel.bvh --preset parallel
```

La configuration 25 points doit rester cohérente avec le checkpoint. Toute
modification de l'ordre des keypoints exige une mise à jour du mapping dans
`backend/parallel_extractor.py`.

## Analyse de Recherche : Retargeting VRM, Verrous Géométriques et Interpénétrations

### 1. Synthèse de l'Implémentation VRM 1.0 / VRMA
Le projet Mimo a validé une chaîne complète de conversion de la gestuelle signée (LSF/ASL) :
- **Solveur IK Anatomique Découplé** : Projection des 33 repères corporels et 21 points 3D par main en quaternions unitaires relatifs conformes au standard **VRM 1.0 / OpenXR Humanoid**.
- **Résolution Fine des Mains (15 os / main)** : Calcul de l'opposition tridimensionnelle du pouce (articulation trapézo-métacarpienne en selle) et décomposition flexion/abduction des 4 doigts longs.
- **Export Binaire Universel `.vrma`** : Serializer direct conforme à l'extension officielle `VRMC_vrm_animation` (Khronos glTF 2.0).

### 2. Étude Critique : Le Phénomène d'Interpénétration des Doigts (Collisions)
L'observation expérimentale des animations générées met en évidence des croisements ou interpénétrations de surface lors de configurations serrées (dactylologie : lettres "A", "E", "S", "M", "O").

**Nature du problème : Spécifique à un modèle ou universel ?**
Ce phénomène est **universel à l'ensemble de la modélisation 3D en animation temps réel**, et indépendant du modèle 3D employé :
1. **Discordance Squelette vs Volume (Mesh)** : Le tracking visuel (MediaPipe) et le solveur IK opèrent exclusivement sur des squelettes filaires sans volume (arbres de segments cinématiques). À l'inverse, l'avatar 3D habillant ce squelette possède une géométrie surfacique dotée d'une épaisseur de chair (environ 1.5 à 2 cm par phalange).
2. **Occultation Monoculaire en Poing Fermé** : Lorsque la main se referme, la paume masque les phalanges distales. L'inférence monoculaire 2D perd la certitude sur la coordonnée de profondeur ($Z$).
3. **Absence de Détection de Contact Rigide** : Un solveur cinématique direct applique les rotations calculées sans tester la pénétration des volumes géométriques contigus. Si la morphologie digitale de l'avatar diffère de celle du locuteur filmé, les maillages s'interpénètrent.

**Ce qui est déjà résolu dans Mimo :**
- Bornage biomécanique strict (clipping) interdisant l'hyperextension vers l'arrière ($[0, \pi/2]$ rad) et limitation de l'abduction latérale ($[-0.4, 0.4]$ rad), éliminant les torsions non-humaines.

### 3. Perspectives de Recherche & Intégration d'Anipose
Pour surmonter ces limites intrinsèques à la vision monoculaire, deux axes de recherche sont engagés :
- **Intégration d'Anipose (Triangulation Multi-Vues Calibrée)** :
  - Déploiement d'un banc de capture à caméras synchronisées (face + vue latérale à 45° ou 90°).
  - Élimination des pertes de profondeur par croisement des matrices de projection $P = K[R|t]$.
  - Optimisation spatio-temporelle sous contrainte de segments rigides (invariance temporelle des longueurs de phalanges).
- **Couche de Relaxation Géométrique (PBD / Volumes Englobants)** :
  - Modélisation de capsules englobantes (sphero-cylindres) sur chaque segment digital.
  - Correction post-solveur par projection de contraintes de non-interpénétration (PBD - *Position Based Dynamics*) garantissant un contact tangentiel parfait lors des fermetures de poing.

---

## Perspectives prioritaires

1. Installer et valider les dépendances parallèles dans un environnement
   dédié.
2. Produire les trois sorties sur la même vidéo :
   `holistic`, `hybrid` et `parallel`.
3. Comparer la couverture, les frames perdues, la stabilité des poignets et
   l'erreur de reconstruction BVH.
4. Vérifier les trois BVH dans Blender, en particulier la tête, les poignets,
   les doigts et les pieds.
5. Ajouter des métriques de comparaison et des overlays vidéo.
6. Améliorer la profondeur et l'orientation des mains avant tout retargeting.
7. Réparer ou requalifier les tests Mixamo historiques après stabilisation du
   format BVH.

## Conclusion

Mimo dispose maintenant d'une base fonctionnelle pour mesurer MediaPipe
directement en BVH, d'une méthode parallèle YOLOv8/ViTPose pour comparer le
corps, et d'une interface CLI unique. La prochaine validation importante n'est
pas d'ajouter davantage de modèles, mais de mesurer objectivement quelle
pipeline restitue le mieux les bras, le buste et les mains d'un signeur.
