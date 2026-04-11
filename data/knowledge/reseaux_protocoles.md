# Protocoles Reseau

## TCP (Transmission Control Protocol)

TCP est un protocole de transport fiable oriente connexion. Il garantit la livraison des paquets dans l'ordre et sans erreur. Avant toute transmission, TCP etablit une connexion via un handshake en trois temps (SYN, SYN-ACK, ACK).

TCP est utilise pour les applications qui exigent une fiabilite absolue : navigation web (HTTP/HTTPS), transfert de fichiers (FTP), email (SMTP, IMAP, POP3), et acces distant (SSH). Son inconvenient est la latence induite par le controle d'erreurs et la retransmission des paquets perdus.

## UDP (User Datagram Protocol)

UDP est un protocole de transport rapide mais sans garantie de livraison. Contrairement a TCP, il n'etablit pas de connexion prealable et n'effectue pas de controle de flux. Chaque datagramme est envoye independamment.

UDP est prefere pour les applications temps reel ou la rapidite prime sur la fiabilite : streaming video et audio, jeux en ligne, VoIP, et DNS. La perte occasionnelle d'un paquet est acceptable dans ces contextes.

## Comparaison TCP vs UDP

TCP est fiable, ordonne, oriente connexion, mais plus lent. UDP est rapide, leger, sans connexion, mais peu fiable. Le choix depend des besoins : fiabilite (TCP) ou vitesse (UDP).

## HTTP et HTTPS

HTTP (HyperText Transfer Protocol) est le protocole de base du Web. Il fonctionne au dessus de TCP. HTTPS est la version securisee qui ajoute une couche TLS pour chiffrer les communications.

## DNS (Domain Name System)

DNS traduit les noms de domaine en adresses IP. Il utilise principalement UDP sur le port 53 pour des requetes rapides. Pour les transferts de zone ou les reponses volumineuses, il bascule sur TCP.
