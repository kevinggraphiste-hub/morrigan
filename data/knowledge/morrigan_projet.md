# Projet Morrigan

## Vision

Morrigan est une architecture IA modulaire post-LLM concue par Kevin (Scarlet Wolf). Son objectif est d'atteindre un niveau de performance conversationnelle comparable aux LLMs actuels, mais avec une architecture fondamentalement differente : modulaire, efficiente, et capable de tourner sur du materiel modeste.

## Philosophie

L'efficience prime sur la puissance brute. Le faucon pelerin calcule des trajectoires balistiques avec un cerveau de quelques grammes. Le cerveau humain fait tourner l'intelligence a 20 watts. Morrigan s'inspire de cette efficience architecturale plutot que du paradigme "toujours plus gros".

## Modules

Morrigan est composee de six modules nommes d'apres la mythologie celtique.

An Dagda est l'orchestrateur central. Il analyse chaque input utilisateur, determine quels modules activer, et assemble les reponses.

Brigid est le reseau neuronal liquide (LNN). Elle gere l'intuition, le pattern recognition, la classification d'intention, et les taches creatives. Son architecture est basee sur les Closed-form Continuous-time networks (CfC).

Ogham est le moteur symbolique. Il applique des regles logiques, structure les reponses, et effectue le raisonnement par deduction. Implemente avec pyDatalog.

Danann est la memoire vectorielle. Elle stocke les connaissances a long terme via Supabase pgvector et les recupere par recherche semantique. Utilise sentence-transformers pour les embeddings locaux.

Scathach est le module langage. Elle genere le texte en langage naturel a partir des donnees des autres modules. Phase 1 : templates Jinja2. Phase 2 : backend RWKV.

Cauldron est la memoire de travail. Elle gere le contexte conversationnel, l'historique des echanges, et la condensation des sessions.

## Stack Technique

Python 3.11+, PyTorch pour Brigid, ncps pour les Liquid Neural Networks, sentence-transformers pour les embeddings, Supabase pgvector pour la persistence, pyDatalog pour le raisonnement symbolique, Jinja2 pour les templates, et python-telegram-bot pour l'interface utilisateur.
