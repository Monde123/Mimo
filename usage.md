# Mimo — guide d'utilisation

Mimo extrait les landmarks MediaPipe d'une vidéo de langue des signes et les
exporte d'abord en BVH afin d'observer la qualité de la détection, puis rejoue ce
BVH sur un modèle Mixamo (`mimo retarget`).

Les commandes publiques sont accessibles via l'outil `mimo` après installation
du projet :

```powershell
python -m pip install -e .
mimo --help
mimo <commande> --help
```

| Commande | Rôle | Section |
|---|---|---|
| `mimo animate` | vidéo → BVH → `.glb` Mixamo animé, en une commande | 6 |
| `mimo bvh` | vidéo → BVH (+ rapport `.report.json`) | 3 |
| `mimo retarget` | BVH → clip JSON Mixamo, ou `.glb` animé | 6 |
| `mimo bake-mixamo` | clip JSON → `.glb` Mixamo animé | 6 |
| `mimo vrma` | vidéo → clip VRM Animation | 5 |
| `mimo bake-vrm`, `mimo inspect-vrm` | injecter / inspecter un clip VRMA | 5 |
| `mimo server` | API HTTP Flask | 8 |

Chaîne recommandée : `mimo bvh` → vérifier le BVH dans Blender → `mimo retarget`,
ou les deux étapes d'un coup avec `mimo animate`.

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

# Pose Landmarker + Hand Landmarker dédiés
mimo bvh input.mp4 precise.bvh --preset precise
```

Le pipeline principal exporte le haut du corps et les mains :

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

La pipeline `pose_hands` utilise `pose_landmarker_full.task` pour le corps et
`hand_landmarker.task` pour les mains, dans une seule lecture vidéo avec des
timestamps partagés :

```powershell
mimo bvh input.mp4 output_pose_hands.bvh `
  --pipeline pose_hands --body upper --hands on
```

Chaque BVH est accompagné d'un rapport `*.report.json` contenant les frames
retenues, la couverture des détections, les points maintenus/interpolés, le
repère utilisé et les avertissements de reconstruction.

Options utiles :

```text
--bvh-mode positions     positions brutes, utile pour diagnostiquer MediaPipe
--bvh-mode rotations     rotations BVH calculées à partir des landmarks
--source auto|world|normalized
--inspect                affiche le format des frames avant/après lissage
--no-recenter            conserve la position absolue
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

## 5. Export VRMA pour Avatars VRM / Three.js (Recommandé)

Le pipeline VRMA convertit les repères de buste et les 21 points 3D de chaque main vers le standard officiel **VRM 1.0 / OpenXR Humanoid** avec l'extension `VRMC_vrm_animation`. C'est le format universel pour animer en temps réel un avatar 3D en langue des signes :

```powershell
# 1. Extraire et exporter directement au format binaire standard .vrma (glTF 2.0 / VRMC_vrm_animation) :
mimo vrma input.mp4 animation.vrma --pipeline hybrid --fps 30

# 2. Exporter au format JSON structuré pour inspection ou intégration web personnalisée :
mimo vrma input.mp4 animation.vrma.json --pipeline hybrid --fps 30

# 3. Injecter (baker) directement une animation dans un modèle 3D VRM autonome :
mimo bake-vrm avatar.vrm animation.vrma.json avatar_anime.vrm

# 4. Inspecter en mode console ASCII les angles et la motricité des doigts :
mimo inspect-vrm animation.vrma.json
```

Le fichier `.vrma` binaire généré est directement chargeable par `@pixiv/three-vrm-animation`, Blender (avec le plugin VRM) et Unity. Il contient les canaux d'animation pour les 55 os humanoïdes et en particulier les **15 os de chaque main** (`ThumbMetacarpal`, `IndexProximal`, etc.).

## 6. Retargeting BVH → Mixamo

Le BVH produit par `mimo bvh` se rejoue sur un modèle Mixamo glTF
(os `mixamorig:*`). Il faut l'extra `glb` : `python -m pip install -e .[glb]`.

```powershell
# 1. Vidéo -> BVH (intermédiaire inspectable dans Blender)
mimo bvh a.mp4 out\a_sign.bvh --preset sign

# 2. BVH -> clip Mixamo pour le modèle cible
mimo retarget out\a_sign.bvh clara.glb -o out\a_sign.mixamo.json

# Sans recalage de la pose de repos (formule brute, pour comparer)
mimo retarget out\a_sign.bvh clara.glb -o out\a_sign.brut.json --no-align-rest

# Ou directement le modèle animé, autonome (lisible par three.js, Blender, un viewer glTF)
mimo retarget out\a_sign.bvh clara.glb -o out\clara_a_sign.glb --name Signe_A

# Intégrer un clip JSON déjà produit
mimo bake-mixamo out\a_sign.mixamo.json clara.glb -o out\clara_a_sign.glb
```

Avec une sortie `.glb`, l'animation (pistes `rotation`, interpolation `LINEAR`) est
ajoutée au modèle. Le maillage, la peau et les textures ne sont pas modifiés.
Les animations déjà présentes dans le modèle sont retirées, sauf avec `--keep-animations`
(seule une animation du même nom est alors remplacée). Les deux chemins
(`retarget -o x.glb`, ou `retarget -o x.json` puis `bake-mixamo`) produisent le même
fichier. Le bake refuse un clip dont un quaternion est nul ou non fini, ou dont les temps
ne sont pas strictement croissants.

Pour regarder le résultat dans Cursor / VS Code : extension **GLTF/GLB Viewer**
(`OHZIInteractiveStudio.ohzi-vscode-glb-viewer`), puis ouvrir le `.glb` et lancer
l'animation dans le panneau *Animations*.

La commande affiche un résumé : frames, fps, os animés, os introuvables dans le modèle,
plus grande correction de repos. Elle s'arrête avec un message clair (code 2) si le BVH
est absent ou mal formé, si le modèle n'est pas un `.glb` / `.gltf` / `.vrm` valide, ou
si aucun os `mixamorig:*` n'y est trouvé (un modèle VRM n'a pas ces noms).

Le JSON contient une piste de quaternions locaux glTF par os (`tracks`, `[x, y, z, w]`,
une clé par frame avec `time` en secondes), `fps`, `frameCount`, et dans `meta` les os
introuvables (`bonesSkipped`) et l'écart de repos corrigé par os (`restCorrectionDeg`).
Dans three.js, chaque piste devient un `QuaternionKeyframeTrack` sur
`<nom de l'os>.quaternion`.

- La T-pose du modèle n'est pas celle du BVH : sur Clara, le bras est à 7° et le pouce à
  plus de 20°. Chaque os est donc recalé sur la direction du BVH, la torsion étant fixée
  par une direction secondaire (ligne des hanches, des épaules, index → auriculaire).
- Résultat : tous les os pointent comme dans le BVH. Les directions secondaires gardent
  seulement l'écart de forme entre les deux squelettes (1,8° et 4,8° pour la paume de Clara).
- `align_rest=False` revient à la formule UniVRM d'origine (`L·W⁻¹·N·W`), sans ce recalage.
- La translation du bassin n'est pas transférée.
- La tête du BVH n'a pas de rotation propre : elle suit le haut du buste.

### Tout en une commande : `mimo animate`

`mimo animate` enchaîne l'extraction et le retargeting en mode monoculaire. Le résultat est identique à `mimo bvh` suivi de `mimo retarget -o x.glb`. Le BVH
intermédiaire est gardé à côté de la sortie (`x.bvh`, ou le chemin donné par `--bvh`).

```powershell
mimo animate a.mp4 --model clara.glb -o out\clara_a.glb --preset sign
```

Les arguments sont vérifiés avant l'extraction :
- sortie requise en `.glb` ;
- vidéo existante ;
- modèle lisible avec squelette Mixamo (`mixamorig:*`).

## 7. Tests

```bash
python -m pip install -e .[dev]
python -m pytest
```

Les tests du retargeting rejouent des poses aléatoires sur `clara.glb` (racine du dépôt)
et sont ignorés si ce fichier est absent.

## 8. Serveur HTTP

Le serveur Flask est optionnel et hors du flux d'analyse standard :

```bash
pip install -e .[server]
mimo server                      # http://127.0.0.1:5000, debug désactivé
mimo server --host 0.0.0.0 --port 8000
```

`POST /process-video` accepte un fichier vidéo (.mp4, .mov, etc.), effectue l'extraction via le pipeline hybride et met à disposition le BVH généré via `GET /download/<clipId>`.
