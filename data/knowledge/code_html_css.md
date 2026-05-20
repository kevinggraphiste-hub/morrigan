# HTML et CSS — frontend essentiel

## HTML sémantique

Préférer les balises sémantiques aux `<div>` génériques : `<header>`,
`<nav>`, `<main>`, `<article>`, `<section>`, `<aside>`, `<footer>`. Elles
améliorent l'accessibilité (lecteurs d'écran), le SEO, et la maintenance.

`<section>` regroupe du contenu thématique avec un heading explicite.
`<article>` représente une unité auto-suffisante (post, commentaire,
carte). En cas de doute, `<section>` par défaut.

## Accessibilité

Toujours fournir un texte alternatif aux images : `alt="description"`,
ou `alt=""` pour les images purement décoratives. Pour les icônes
interactives, ajouter `aria-label`. Les boutons clickables doivent être
des `<button>`, pas des `<div onclick>`.

Tester au clavier seul : si la navigation `Tab` ne couvre pas tout, la
page n'est pas accessible. Le focus visible (outline) est essentiel —
ne pas le supprimer en CSS sans alternative.

## Images responsives

`srcset` permet au navigateur de choisir la résolution adaptée au
contexte d'affichage. `sizes` complète en précisant les contraintes
de largeur. Pour des images alternatives selon le viewport,
utiliser `<picture>` avec `<source media="...">`.

## CSS layout : flexbox vs grid

**Flexbox** raisonne sur une dimension (rangée ou colonne) — idéal pour
des composants (barres de nav, cartes alignées). Modèle main-axis /
cross-axis avec `justify-content` / `align-items`.

**Grid** raisonne en 2D — idéal pour des layouts de page complets.
`grid-template-columns: repeat(auto-fit, minmax(200px, 1fr))` crée
une grille responsive sans media query.

Les deux ne s'excluent pas : un grid contient souvent des flex.

## Sélecteurs et spécificité

La spécificité hiérarchise inline > id > class/attribute/pseudo-class
> élément. `!important` casse cette logique — à éviter sauf en
override délibéré dans une feuille utilisateur.

Les pseudo-classes (`:hover`, `:focus-visible`, `:has(…)`) permettent
des sélections sans JS. `:has()` est arrivée tard mais débloque des
patterns CSS qui demandaient avant un préprocesseur ou du JavaScript.

## Media queries

`@media (max-width: 600px)` cible les écrans étroits. Penser **mobile-first** :
écrire les styles de base pour mobile, puis enrichir avec `min-width:`.
Les media queries de préférence utilisateur sont utiles aussi :
`prefers-reduced-motion`, `prefers-color-scheme`.

## Unités

`px` pour les bordures fines, `rem` (relative à la racine) pour la
typographie et les espacements globaux (zoom-friendly), `em` pour des
échelles locales à un composant. `%` et `fr` (en grid) pour le fluide.
