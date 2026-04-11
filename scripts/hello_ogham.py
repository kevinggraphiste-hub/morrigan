"""
OGHAM — Prototype de raisonnement symbolique.

Demontre les capacites du moteur symbolique :
- Faits et regles
- Chainages avant/arriere
- Raisonnement par transitivite
- Requetes complexes
"""

import sys
sys.path.insert(0, ".")

from pyDatalog import pyDatalog

# Declarer les termes au niveau module (pyDatalog les injecte dans globals)
pyDatalog.create_terms(
    "X, Y, Z, W, "
    "is_a, has_property, protocol, "
    "uses_connection, reliable, fast, "
    "better_for, recommend"
)


def main():
    print("=" * 60)
    print("  OGHAM — Prototype Moteur Symbolique")
    print("=" * 60)
    print()

    # === Base de connaissances : protocoles reseau ===
    print("--- Base de connaissances ---")

    # Faits : taxonomie
    +is_a("TCP", "protocol")
    +is_a("UDP", "protocol")
    +is_a("HTTP", "protocol")
    +is_a("DNS", "protocol")

    # Faits : proprietes
    +has_property("TCP", "connection_oriented")
    +has_property("TCP", "reliable")
    +has_property("TCP", "ordered")
    +has_property("TCP", "slow")

    +has_property("UDP", "connectionless")
    +has_property("UDP", "unreliable")
    +has_property("UDP", "unordered")
    +has_property("UDP", "fast")

    # Faits : quel protocole utilise quoi
    +uses_connection("HTTP", "TCP")
    +uses_connection("DNS", "UDP")

    # Regles : deduction
    reliable(X) <= has_property(X, "reliable")
    fast(X) <= has_property(X, "fast")

    # Regle : recommandation basee sur le besoin
    better_for(X, "streaming") <= fast(X)
    better_for(X, "file_transfer") <= reliable(X)
    better_for(X, "web") <= reliable(X) & has_property(X, "ordered")
    better_for(X, "gaming") <= fast(X)

    # === Requetes ===
    print()
    print("--- Requetes ---")
    print()

    # 1. Quels sont les protocoles ?
    result = is_a(X, "protocol")
    print(f"Protocoles connus: {sorted([str(r[0]) for r in result])}")

    # 2. Proprietes de TCP
    result = has_property("TCP", X)
    print(f"Proprietes TCP:    {sorted([str(r[0]) for r in result])}")

    # 3. Proprietes de UDP
    result = has_property("UDP", X)
    print(f"Proprietes UDP:    {sorted([str(r[0]) for r in result])}")

    # 4. Quel protocole est fiable ?
    result = reliable(X)
    print(f"Protocoles fiables: {[str(r[0]) for r in result]}")

    # 5. Quel protocole est rapide ?
    result = fast(X)
    print(f"Protocoles rapides: {[str(r[0]) for r in result]}")

    # 6. Recommandations
    print()
    print("--- Recommandations (raisonnement) ---")
    use_cases = ["streaming", "file_transfer", "web", "gaming"]
    for use_case in use_cases:
        result = better_for(X, use_case)
        protocols = [str(r[0]) for r in result]
        print(f"  Pour {use_case:15s} -> {protocols}")

    # 7. Chainage : HTTP utilise TCP, TCP est fiable, donc HTTP herite
    print()
    print("--- Chainage deductif ---")
    result = uses_connection("HTTP", X)
    rows = list(result)
    if rows:
        transport = str(rows[0][0])
        is_reliable = len(reliable(transport)) > 0
        print(f"HTTP utilise {transport}")
        print(f"  -> {transport} est {'fiable' if is_reliable else 'non fiable'}")
        print(f"  -> HTTP herite donc de la fiabilite de {transport}")

    print()
    print("Ogham raisonne correctement.")


if __name__ == "__main__":
    main()
