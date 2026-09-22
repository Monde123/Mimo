# Mimo — guide d'utilisation

Mimo extrait les landmarks MediaPipe d'une vidéo de langue des signes et les
exporte d'abord en BVH afin d'observer la qualité de la détection. Le
retargeting Mixamo est optionnel et isolé dans `backend/mixamo/`.

Les commandes publiques sont accessibles via l'outil `mimo` après installation
du projet :

```powershell
python -m pip install -e .
mimo --help
```

## 1. Créer l'environnement Python

Le projet est validé avec **Python 3.13**. Python 3.10 à 3.13 est accepté ;
Python 3.13 est recommandé pour l'environnement actuellement utilisé.

Windows PowerShell :

```powershell
git clone https://github.com/Monde123/Mimo.git
cd Mimo
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le pipeline facultatif `parallel` installe ses dépendances séparément :

```powershell
python -m pip install -r requirements-parallel.txt
```

Il combine YOLOv8 et ViTPose via MMPose. Le checkpoint ViTPose doit être
accompagné de la configuration MMPose correspondante (`--vitpose-config`) ;
un checkpoint seul ne suffit pas. Les mains sont produites par le Hand
Landmarker MediaPipe afin de conserver les 21 points nécessaires aux doigts.

Linux/macOS :

```bash
git clone https://github.com/Monde123/Mimo.git
cd Mimo
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Vérifier l'interpréteur actif :

```bash
python --version
python -c "import cv2, mediapipe, numpy, scipy; print('Mimo runtime: OK')"
```

## 2. Modèles MediaPipe

Les bundles `.task` sont chargés depuis `backend/models/` :

```text
backend/models/hand_landmarker.task
backend/models/holistic_landmarker.task
backend/models/pose_landmarker_full.task
```

## 3. Export BVH — pipeline principal

Des préréglages raccourcissent les commandes courantes :

```powershell
# Langue des signes : corps Holistic + mains dédiées, haut du corps
mimo bvh input.mp4 signer.bvh --preset sign

# Corps complet, sans activer les mains dédiées
mimo bvh input.mp4 corps_complet.bvh --preset full
```

Le pipeline principal ne dépend pas de Mixamo. Il exporte le haut du corps et
les mains, sans jambes ni pieds :

```powershell
python -m backend.process_video input.mp4 output_holistic.bvh `
  --body upper --hands on --pipeline holistic
```

Le pipeline hybride utilise Holistic pour le corps et le Hand Landmarker dédié
pour les 21 landmarks de chaque main :

```powershell
python -m backend.process_video input.mp4 output_hybrid.bvh `
  --body upper --hands on --pipeline hybrid
```

Pipeline corps optionnel :

```powershell
python -m backend.process_video input.mp4 output_parallel.bvh `
  --pipeline parallel --yolo-model yolov8n.pt `
  --vitpose-config vitpose_config.py --vitpose-checkpoint vitpose.pth
```

Le préréglage `parallel` recherche automatiquement les modèles dans le dossier
local `backend/models/parallel/` :

```text
backend/models/parallel/yolov8l.pt
backend/models/parallel/vitpose-s-coco_25.pth
backend/vitPose/ViTPose_common.py
backend/vitPose/ViTPose_coco_25.py
```

La configuration et le fichier commun doivent rester dans le même dossier,
car `ViTPose_coco_25.py` utilise un import relatif. Il faut encore fournir
`--vitpose-config` si la configuration n'est pas placée à cet emplacement :

```powershell
python -m backend.process_video input.mp4 parallel.bvh `
  --preset parallel --vitpose-config chemin\vers\vitpose_config.py
```

Pour utiliser un autre dossier local, définir `MIMO_MODEL_DIR` :

```powershell
$env:MIMO_MODEL_DIR = "D:\mes-modeles-mimo"
```

Chaque BVH est accompagné d'un rapport `*.report.json` contenant les frames
retenues, la couverture des détections, les points maintenus/interpolés, le
repère utilisé et les avertissements de reconstruction.

Options utiles :

```text
--bvh-mode positions     positions brutes, utile pour diagnostiquer MediaPipe
--bvh-mode rotations      rotations BVH calculées à partir des landmarks
--source auto|world|normalized
--inspect                 affiche le format des frames avant/après lissage
--no-recenter             conserve la position absolue
```

## 4. Comparer les deux pipelines

Exporter les deux fichiers avec la même vidéo et les mêmes options, puis
comparer les rapports JSON :

```powershell
python -m backend.process_video input.mp4 holistic.bvh `
  --body upper --hands on --pipeline holistic
python -m backend.process_video input.mp4 hybrid.bvh `
  --body upper --hands on --pipeline hybrid
```

Holistic utilise ses propres mains. Hybrid conserve le corps Holistic mais
remplace ses mains par le modèle Hand Landmarker dédié. Cette comparaison doit
être faite avant tout retargeting.

## 5. Chaîne Mixamo optionnelle

La chaîne Mixamo est isolée dans `backend/mixamo/` et n'est pas importée par
le pipeline BVH. Son outil précis produit encore un JSON Mixamo :

```powershell
python -m backend.mixamo.process_precise input.mp4 output.json
```

Les solvers, le schéma JSON, la calibration de rig, l'export et les outils
GLB associés sont également dans ce dossier. Cette chaîne n'est pas nécessaire
pour inspecter les sorties MediaPipe en BVH.

## 6. Tests

```bash
python -m pytest
```

Les tests BVH sont indépendants d'une vidéo réelle. Les tests Mixamo restent
regroupés avec les modules de `backend/mixamo/`.

## 7. Serveur HTTP

Le serveur Flask est optionnel et hors du flux d'analyse BVH :

```bash
python -m backend.server
```
