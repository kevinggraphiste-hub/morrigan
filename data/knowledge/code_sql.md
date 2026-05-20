# SQL — requêtes et schéma

## SELECT et projection

La forme canonique : `SELECT col1, col2 FROM table WHERE cond ORDER BY col`.
Toujours nommer les colonnes plutôt que `SELECT *` en production — c'est
plus rapide, plus robuste aux changements de schéma, et auto-documenté.

`DISTINCT` déduplique mais coûte cher (tri). Sur PostgreSQL,
`SELECT DISTINCT ON (col)` permet une déduplication par groupe avec ordre.

## JOIN

`INNER JOIN` ne retourne que les lignes ayant un match dans les deux tables.
`LEFT JOIN` garde toutes les lignes de gauche, `NULL` à droite si pas de
match — utile pour détecter les manquants : `WHERE right.id IS NULL`.

Toujours préférer le JOIN explicite à la jointure implicite par virgule
dans le FROM : c'est plus lisible et plus difficile de produire un
produit cartésien par erreur.

## Agrégation et GROUP BY

`GROUP BY` regroupe les lignes ; `WHERE` filtre avant agrégation, `HAVING`
filtre après. Les fonctions d'agrégation usuelles : `COUNT`, `SUM`, `AVG`,
`MIN`, `MAX`, `STRING_AGG`, `ARRAY_AGG`.

Sur PostgreSQL, les fenêtres (`OVER (PARTITION BY ...)`) permettent
d'agréger sans regrouper, idéal pour le rang, le cumul, la moyenne mobile.

## CTE et lisibilité

Les **Common Table Expressions** (`WITH cte AS (…) SELECT … FROM cte`)
nomment les sous-requêtes et permettent l'écriture séquentielle de
requêtes complexes. Sur PostgreSQL 12+, les CTE sont fusionnées par le
planner (sauf si on les marque `MATERIALIZED`).

## Index

Un index accélère les lectures (WHERE, JOIN, ORDER BY) au prix des
écritures et de l'espace disque. Règle : indexer les colonnes de
prédicat fréquentes, les FK, les colonnes triées.

Un index multi-colonnes `(a, b, c)` peut servir une requête qui filtre
sur `a` seul, sur `(a, b)`, ou sur `(a, b, c)` — mais pas sur `b` seul.

## Contraintes

`PRIMARY KEY` identifie unique + non-null. `FOREIGN KEY` référence une
clé d'une autre table — protège l'intégrité. `UNIQUE`, `CHECK`, `NOT NULL`
complètent l'arsenal de contraintes déclaratives. Préférer les
contraintes en base au contrôle applicatif — plus difficile à contourner.

## EXPLAIN

`EXPLAIN ANALYZE` exécute la requête et montre le plan choisi + les temps
réels. C'est l'outil de référence pour comprendre pourquoi une requête
est lente. Chercher les `Seq Scan` sur les grosses tables, les estimations
de cardinalité aberrantes, les tris en mémoire externe.
