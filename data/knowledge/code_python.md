# Python — bases du langage

## Boucles et comprehensions

La boucle `for` itère sur tout objet iterable : liste, tuple, dict, générateur,
fichier. `range(start, stop, step)` produit une séquence d'entiers paresseuse.

Les **comprehensions** condensent un map+filter+collect en une expression :

```python
squares = [x * x for x in range(10) if x % 2 == 0]
```

Les **dict comprehensions** suivent la même syntaxe avec deux-points :
`{k: v for k, v in items}`. Pour des données volumineuses, préférer une
expression génératrice `(x*x for x in ...)` qui n'évalue qu'au besoin.

## Gestion d'erreurs

Le bloc `try/except` capture les exceptions. Toujours intercepter le type
le plus spécifique possible — un `except Exception` attrape trop. Le bloc
`finally` s'exécute toujours (utile pour libérer une ressource), et `else`
s'exécute si aucune exception n'a été levée.

`raise ... from e` préserve la chaîne d'exceptions, précieux en debug.

## Décorateurs

Un décorateur est une fonction qui prend une fonction et renvoie une fonction.
Sucre syntaxique : `@deco` au-dessus de `def f()` est équivalent à
`f = deco(f)`. Les décorateurs paramétrés sont des fabriques de décorateurs.
Pour préserver la signature et la doc, utiliser `functools.wraps`.

## Async/await

Un `async def` définit une coroutine. `await` suspend l'exécution jusqu'à
ce qu'un awaitable se résolve. `asyncio.gather(*coros)` lance des coroutines
en parallèle. Différence avec `threading` : asyncio est coopératif
mono-thread, idéal pour I/O massif, pas pour du CPU-bound.

## Classes et `__init__`

`__init__(self, ...)` initialise l'instance après création. Pour une
dataclass immuable : `@dataclass(frozen=True)`. Les méthodes `__repr__`,
`__eq__`, `__hash__` sont générées automatiquement par `@dataclass`.

## Le point d'entrée

Le pattern `if __name__ == "__main__":` permet à un module d'être à la
fois importable et exécutable. C'est le bon endroit pour les CLI scripts.
