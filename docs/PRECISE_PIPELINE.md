# Précision corps + mains

La version précise utilise deux modèles MediaPipe dans **une seule lecture vidéo** :

- `pose_landmarker_full.task` pour le corps en world landmarks ;
- `hand_landmarker.task` pour les 21 landmarks de chaque main.

Commande :

```bash
python -m backend.process_precise input.mp4 output.json --rig configs/mixamo_default.json
```

L'ordre est :

```text
vidéo
→ Pose + Hand Landmarker synchronisés par frame
→ score de qualité
→ suppression des bords non fiables et arrêt après perte prolongée
→ interpolation bornée
→ filtrage One-Euro
→ solveur corps world→local
→ solveur mains Kalidokit-style
→ calibration Mixamo
→ validation JSON
→ export atomique
```

## Pourquoi ne pas additionner aveuglément deux passes

`extract_pose()` et `extract_hands()` existants décodent chacun la vidéo. Les exécuter séparément puis concaténer les listes peut désaligner les frames lorsque la pose est absente. Mimo utilise donc `extract_pose_and_hands()` : un seul décodage, des timestamps communs et une frame de sortie même si un seul composant est détecté.

## Calibration

Le solveur du corps fournit des rotations locales après calcul des rotations monde. La calibration doit ensuite préciser les axes réels du rig Mixamo, notamment `hand.bendAxis` et `hand.thumbBendAxis`. Les quaternions sont en `[x,y,z,w]`.

## Limites

Les orientations axiales non observables par une caméra monoculaire restent ambiguës. Le calcul avant-bras utilise le vecteur poignet→index comme référence de roll ; il doit être validé avec une vidéo de pronation/supination et le modèle cible.
