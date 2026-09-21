# Mimo — guide d'utilisation hors ligne

## Objectif

Mimo transforme une vidéo 2D d'une personne signante en un clip JSON de rotations locales compatible avec les noms de bones Mixamo. Le pipeline ne nécessite pas Unity et ne traite pas la vidéo en temps réel.

```text
vidéo → validation → MediaPipe Holistic → interpolation → lissage → rotations → JSON
```

## 1. Installation

```bash
git clone https://github.com/Monde123/Mimo.git
cd Mimo
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Modèles MediaPipe

Télécharger les bundles `.task` compatibles avec MediaPipe Tasks et les placer dans `backend/models/` :

```text
backend/models/holistic_landmarker.task
```

Le pipeline hors ligne utilise Holistic pour obtenir simultanément le corps et les deux mains. Le modèle Hand Landmarker séparé peut être ajouté plus tard pour une passe spécialisée des mains.

## 3. Conversion d'une vidéo

```bash
python -m backend.process_video videos/signe.mp4 exports/signe.json
```

Forcer une cadence de sortie :

```bash
python -m backend.process_video a.mp4 exports/signe.json --fps 30
```

Le FPS est égal à celui de la vidéo par défaut. Il ne faut pas modifier le FPS uniquement dans le fichier : le pipeline ré-échantillonne réellement les frames produites.

## 4. Format de sortie

```json
{
  "format": "mimo.mixamo.animation",
  "version": 1,
  "skeleton": "mixamo",
  "fps": 30,
  "duration": 2.4,
  "frames": [
    {
      "time": 0.0,
      "bones": {
        "mixamorig:RightHandIndex1": {
          "rotation": [0.0, 0.0, 0.0, 1.0]
        }
      }
    }
  ]
}
```

Les rotations sont des quaternions `[x, y, z, w]`. Les valeurs sont des rotations locales destinées à être appliquées aux bones portant les noms Mixamo `mixamorig:*`. L'application cliente doit vérifier que son modèle utilise exactement ces noms et la même convention d'axes.

Bones de doigts produits : `Thumb1..3`, `Index1..3`, `Middle1..3`, `Ring1..3`, `Pinky1..3`, pour chaque côté. Les bras et avant-bras sont également produits lorsque la pose du corps est disponible.

## 5. Ordre exact du pipeline

1. Lire les métadonnées vidéo et récupérer le FPS source.
2. Décoder les frames dans l'ordre.
3. Exécuter MediaPipe en mode `VIDEO`, avec timestamps monotoniques.
4. Séparer gauche/droite à partir de `handedness`.
5. Refuser une séquence sans pose exploitable.
6. Maintenir/interpoler les courtes absences de landmarks.
7. Appliquer le lissage temporel aux coordonnées, sans moyenner des quaternions.
8. Calculer l'orientation de la paume à partir du poignet, MCP index et MCP auriculaire.
9. Calculer l'orientation de chaque phalange à partir de ses deux landmarks voisins.
10. Calculer les rotations des bras à partir épaule→coude et coude→poignet.
11. Convertir les rotations dans les noms et l'ordre Mixamo.
12. Écrire atomiquement le JSON final.

## 6. Utilisation Python

```python
from pathlib import Path
from backend.process_video import process_video

process_video(Path("videos/signe.mp4"), Path("exports/signe.json"), fps=30)
```

## 7. Pourquoi le serveur n'est pas requis

`backend/server.py` est seulement une enveloppe HTTP optionnelle. Il est utile si une application distante doit envoyer une vidéo à une machine de calcul. Pour une génération locale ou batch, utilisez `backend.process_video` et ne lancez pas Flask.

## 8. Limites et validation obligatoire

La version actuelle est une base de conversion, pas une garantie biomécanique universelle :

- une caméra monoculaire ne mesure pas parfaitement la profondeur ;
- l'orientation autour de l'axe d'un doigt est partiellement ambiguë ;
- les rotations dépendent de l'orientation T-pose et des axes du rig Mixamo ;
- les occlusions peuvent produire une pose maintenue temporairement ;
- les noms de bones varient parfois selon l'export du modèle.

Avant l'intégration mobile, valider au minimum :

1. une vidéo T-pose/pose neutre ;
2. flexion séparée de chaque doigt ;
3. rotation de la paume ;
4. gestes rapides avec occlusion ;
5. main gauche et main droite ;
6. comparaison vidéo source / avatar ;
7. application des mêmes quaternions au modèle cible.

## 9. Inspirations intégrées

- `DigiHuman` fournit la structure MediaPipe et le traitement frame par frame.
- `Mimic` inspire l'organisation extraction → smoothing → solve → export, les budgets de frames manquantes et les rotations parent-locales ; voir notamment son `rotation_solver.py`.
- `Kalidokit` inspire le calcul de la paume avec trois landmarks et les angles de chaque phalange (`HandSolver`), ainsi que les limites gauche/droite.

Ces dépôts peuvent évoluer ; les recherches de code utilisées pour l'analyse peuvent être incomplètes. Voir [DigiHuman](https://github.com/Danial-Kord/DigiHuman/tree/main/Backend), [Mimic](https://github.com/Abadli-Badro/Mimic/tree/2a22b77e7a1ee60b7ae7d5aa315b7b9f86a13897/mimic) et [Kalidokit](https://github.com/yeemachine/kalidokit).
