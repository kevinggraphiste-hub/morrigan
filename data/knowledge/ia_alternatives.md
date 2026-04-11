# Architectures IA Alternatives aux Transformers

## Liquid Neural Networks (LNN)

Les Liquid Neural Networks sont une classe de reseaux inspires du vers C. elegans et de ses 302 neurones. Developpes au MIT par Ramin Hasani et son equipe, ils utilisent des equations differentielles pour modeliser des neurones a constantes de temps continues (Liquid Time-Constant, LTC).

Leurs avantages : peu de parametres (quelques milliers suffisent), interpretabilite, stabilite bornee, et capacite d'adaptation apres entrainement. Ils tournent efficacement sur CPU, sans necessiter de GPU.

## CfC (Closed-form Continuous-time)

Le modele CfC est une version des LNN avec solution sous forme fermee. Publie en 2022 dans Nature Machine Intelligence, il est de 1 a 5 ordres de grandeur plus rapide a l'entrainement et a l'inference que les LTC originaux, tout en preservant leurs proprietes d'expressivite.

## Mamba et State Space Models

Mamba est une architecture basee sur les State Space Models selectifs, publiee en 2023 par Albert Gu et Tri Dao. Elle offre une inference en temps lineaire (vs quadratique pour les transformers) et une memoire constante, ce qui la rend 4 a 5 fois plus rapide qu'un transformer equivalent.

## RWKV (Receptance Weighted Key Value)

RWKV est une architecture hybride RNN-Transformer, entierement open source sous licence Apache 2.0. Elle combine l'efficacite d'inference des RNN (memoire constante, temps lineaire) avec la qualite d'entrainement des transformers. RWKV-7 "Goose", publie en 2025, atteint des performances state-of-the-art a 3 milliards de parametres.

## KAN (Kolmogorov-Arnold Networks)

Les KAN, proposes par Liu et Tegmark en 2024, remplacent les fonctions d'activation fixes des perceptrons par des fonctions apprenables sur les aretes. Ils se montrent tres performants pour la regression symbolique et la decouverte scientifique, avec des reseaux beaucoup plus petits et interpretables que les MLP classiques.

## IA Neuro-Symbolique

L'IA neuro-symbolique combine les reseaux de neurones (apprentissage par exemples) avec des moteurs symboliques (raisonnement logique, regles). Des systemes comme AlphaGeometry de DeepMind montrent que cette hybridation permet de resoudre des problemes complexes qu'aucune approche seule ne maitrise. IBM developpe NeuroVSA, une architecture neuro-vectorielle symbolique pour le raisonnement cognitif.

## Pourquoi les LLMs sont inefficients

Un LLM stocke tout dans ses poids : connaissances factuelles, capacite linguistique, raisonnement logique, memoire contextuelle. Les techniques de quantization et de pruning montrent que la majorite de ces poids sont redondants. L'information utile tient dans une fraction de l'espace, mais l'entrainement massif reste le seul chemin connu pour y acceder.

L'approche modulaire propose de decomposer l'intelligence en fonctions specialisees, chacune implementee par le composant le plus efficient : reseau neuronal pour l'intuition, moteur symbolique pour la logique, memoire vectorielle pour les connaissances, et petit modele de langage pour la generation.
