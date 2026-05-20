# JavaScript — basiques modernes

## Déclarations de variables

Trois mot-clés pour déclarer une variable, mais une seule recommandation :
utiliser `const` par défaut, `let` quand la réassignation est nécessaire,
et **éviter `var`**. La portée de `var` est la fonction, celle de
`let` et `const` est le bloc — beaucoup plus prévisible.

`const` interdit la **réassignation** mais pas la mutation de l'objet :
`const arr = []; arr.push(1)` reste valide.

## Fonctions fléchées

Les arrow functions `(x) => x * 2` capturent le `this` lexical (pas celui
de l'appelant). Cela les rend dangereuses comme méthodes de classe mais
idéales comme callbacks. Pour un retour d'objet littéral, parenthéser :
`() => ({ a: 1 })`.

## Promises et async/await

Une `Promise` représente une valeur future. `.then()` chaîne les
transformations, `.catch()` gère les erreurs. La syntaxe `async/await`
est du sucre syntaxique sur les promises et permet d'écrire du code
asynchrone qui se lit de manière séquentielle.

`Promise.all([...])` parallélise et échoue à la première erreur.
`Promise.allSettled([...])` attend tout, succès et échecs confondus.

## Modules ESM vs CommonJS

ESM (`import`/`export`) est le standard moderne et le format natif des
modules dans Node 14+ et les navigateurs. CommonJS (`require`/`module.
exports`) reste massivement présent dans l'écosystème Node historique.

Dans `package.json`, `"type": "module"` active ESM pour les fichiers `.js`,
sinon utiliser `.mjs` (ESM) ou `.cjs` (CommonJS) explicitement.

## Closures

Une closure est une fonction qui capture des variables de son scope
englobant. C'est le mécanisme qui permet d'écrire des fabriques de
fonctions, du currying, et des compteurs privés.

## Event loop et microtâches

JavaScript est mono-thread. L'event loop alterne entre tâches (callbacks
de `setTimeout`, I/O…) et microtâches (résolutions de promises). Les
microtâches sont prioritaires — un `await` ou un `.then()` s'exécute
avant le prochain `setTimeout(..., 0)`.
