# Bash — scripting essentiel

## Variables et expansion

Une variable bash se déclare sans `$`, se lit avec : `name=value` puis
`echo "$name"`. **Toujours guillemeter les variables** pour éviter le
splitting de mots et le globbing intempestif.

L'expansion de paramètre offre des opérations puissantes :
`${var:-default}` (valeur par défaut), `${var%suffix}` (retire un suffixe),
`${var/foo/bar}` (substitution).

## Conditions

Le `if [ ... ]` historique reste utile mais `[[ ... ]]` (le double bracket
bash) est plus moderne et sûr : pas besoin de quoter, support des regex
(`=~`), et opérateurs `&&`/`||` à l'intérieur.

```bash
if [[ -f "$file" && "$file" =~ \.log$ ]]; then
  echo "C'est un fichier .log"
fi
```

## Fonctions et arguments

Une fonction se définit `name() { … }` ou `function name { … }`. Les
arguments sont `$1`, `$2`, etc. `$@` liste tous les arguments en
préservant les espaces (utiliser `"$@"`), `$*` les concatène. `$#`
donne le nombre d'arguments.

Pour des options nommées, `getopts` parse `-x` et `--long` (ce dernier
nécessite quelques contorsions en pur bash).

## Pipes, redirections, substitution

Le pipe `|` envoie stdout d'une commande vers stdin de la suivante.
`>` redirige stdout vers un fichier (écrase), `>>` ajoute, `2>` redirige
stderr, `&>` redirige les deux.

La substitution de commande `$(cmd)` capture la sortie d'une commande
en chaîne. La substitution de processus `<(cmd)` la présente comme un
fichier (utile avec `diff`).

## Trap et nettoyage

`trap 'commande' EXIT` exécute une commande à la sortie du script, même
en cas d'erreur. Pattern courant : `trap 'rm -rf "$tmpdir"' EXIT` pour
nettoyer un répertoire temporaire.

## Vérification syntaxique

`bash -n script.sh` parse le script sans l'exécuter — la base pour
détecter les erreurs de syntaxe sans risquer d'effet de bord. C'est
exactement ce que Morrigan-Code utilise pour son vérifieur Bash.
