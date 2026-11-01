# Fall Detection : projet complet

## Contenu
- `colab/FallDetection_Training.ipynb` : entrainement dans Google Colab
- `webcam.py` : detection en temps reel (webcam ou video), alarme sonore + journal `falls_log.csv`
- `requirements.txt`

## 1. Entrainer (Colab)
1. Mets les donnees sur Drive : `Memoir/Fall/Keypoints_CSV` et `Memoir/No_Fall/Keypoints_CSV`.
2. Colab -> Fichier -> Importer le notebook -> choisis `FallDetection_Training.ipynb`.
3. Runtime -> Change runtime type -> **T4 GPU**.
4. Execute toutes les cellules dans l'ordre. Le chargement prend quelques minutes la 1ere fois (cache ensuite).
5. A la fin tu obtiens `best_model_v2.pth` et `config.json` (dans Drive/Memoir et telecharges).

## 2. Tester (VS Code)
```bash
pip install -r requirements.txt
```
Mets `best_model_v2.pth` et `config.json` a cote de `webcam.py`, puis :
```bash
python webcam.py                      # webcam
python webcam.py --video test.mp4     # une video (ideal pour tester sans te jeter par terre)
python webcam.py --threshold 0.7      # changer le seuil
```
Place la camera pour qu'on te voie en entier (tete aux pieds), a 2-3 m. Q pour quitter.

## 3. Ajouter les donnees VR plus tard
Exporte les keypoints d'Unity dans le meme format CSV (`Frame,Keypoint,X,Y,Confidence`, 17 points COCO),
puis dans la cellule 1 du notebook renseigne `EXTRA_DIRS` et relance l'entrainement.

## Remarques
- Le modele se base sur 4 secondes de mouvement (120 frames a 30 fps) : `webcam.py` re-echantillonne le flux pour
  que la vitesse corresponde, meme si ta webcam tourne a un autre nombre d'images/seconde.
- Les resultats de validation sont plus honnetes que les 97 % precedents (split par sessions, fenetres de 120 frames,
  axe Y corrige automatiquement). Un chiffre plus bas est normal.
- Les SMS ont ete retires.
- Si Colab dit que le dossier est introuvable : dans Drive, clic droit sur le dossier partage -> Organiser -> Ajouter un raccourci -> Mon Drive.
