# Git et Docker — outillage dev

## Git : workflow

Les opérations de base : `git status`, `git add -p` (sélectif),
`git commit -m`, `git push`. `git log --oneline --graph` donne un
résumé visuel utile au quotidien.

**Branche par feature** : `git checkout -b feat/short-desc`, on travaille,
on pousse, on ouvre une PR. Jamais commiter directement sur `main` d'un
repo établi.

## Rebase vs merge

`git rebase main` rejoue les commits de la branche feature au-dessus
de main — historique linéaire, pas de commit de merge. À utiliser
**localement** avant d'avoir poussé.

`git merge --no-ff feature` crée un commit de merge qui matérialise
l'intégration — historique plus chaotique mais traçable. C'est le
comportement par défaut de GitHub "Create a merge commit".

**Règle d'or** : ne jamais rebaser des commits déjà publiés sur une
branche partagée. Réécrire l'historique commun casse les checkouts
des collègues.

## Stash

`git stash` met de côté les modifications non commitées. `git stash pop`
les restaure. Utile pour basculer rapidement vers une autre branche sans
commit intermédiaire.

`git stash -u` inclut les fichiers non suivis (utile pour les nouveaux
fichiers). `git stash list` montre toutes les piles stashées.

## Conflits

Quand `git merge` ou `git rebase` ne sait pas combiner, il met des
marqueurs `<<<<<<<` / `=======` / `>>>>>>>` dans le fichier. Éditer
à la main, retirer les marqueurs, `git add` puis `git rebase --continue`
ou `git commit`.

## Docker : Dockerfile

Un Dockerfile décrit comment construire une image. Instructions clés :
`FROM` (image de base), `WORKDIR`, `COPY`, `RUN`, `ENV`, `EXPOSE`,
`CMD` (commande par défaut) ou `ENTRYPOINT` (commande fixe).

`COPY` copie depuis le contexte de build. `ADD` fait pareil mais
auto-décompresse les archives et accepte des URLs — préférer `COPY`
sauf besoin spécifique (moins surprenant).

## Multi-stage builds

Plusieurs `FROM` permettent de séparer le build (avec compilateurs)
de l'image finale (juste les binaires). Réduit drastiquement la taille
de l'image livrée.

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /src
COPY . .
RUN go build -o app

FROM gcr.io/distroless/base
COPY --from=builder /src/app /app
ENTRYPOINT ["/app"]
```

## docker-compose

`docker-compose.yml` orchestre plusieurs conteneurs (app, db, cache).
`depends_on` ne garantit pas que le service est prêt — seulement
qu'il est démarré. Pour attendre une connexion DB, utiliser un
script de wait ou `healthcheck` + `depends_on: { condition: service_healthy }`.

`docker compose up -d` lance en arrière-plan, `docker compose logs -f`
suit les logs, `docker compose down -v` arrête et supprime les volumes
(attention, perte de données).
