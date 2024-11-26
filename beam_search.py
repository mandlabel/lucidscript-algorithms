import heapq

def beam_search(predict_next, start_state, beam_width, max_steps):
    """
    Beam Search algoritmus implementáció.

    :param predict_next: Egy függvény, amely bemeneti állapot alapján visszaadja az összes lehetséges következő állapotot
                         és azok valószínűségeit egy listában [(probability, next_state), ...].
    :param start_state: A kezdőállapot.
    :param beam_width: A sugár szélessége (az állapotok maximális száma egy iterációban).
    :param max_steps: A maximális iterációk száma (állapotok hossza).
    :return: Az optimalizált állapotok listája és az azokhoz tartozó valószínűségek.
    """
    # Beam keresési lista
    beam = [(1.0, start_state)]  # (valószínűség, állapot)

    for step in range(max_steps):
        print(f"\nStep {step + 1}:")
        candidates = []
        for prob, state in beam:
            next_states = predict_next(state)
            for next_prob, next_state in next_states:
                combined_prob = prob * next_prob
                candidates.append((combined_prob, next_state))
        
        # Csak a legjobb állapotokat tartjuk meg a sugár szélessége alapján
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        
        # Logolás a jelenlegi állapotokról
        for prob, state in beam:
            print(f"  Állapot: {state}, Valószínűség: {prob:.4f}")

    return beam

# Példa: Egy egyszerű nyelvi modell, amely szavakat generál
def simple_predict_next(state):
    # Minden állapot (egy szó) egy listát ad vissza [(valószínűség, következő állapot)] formában
    transitions = {
        "START": [("0.4", "the"), ("0.6", "a")],
        "the": [("0.5", "cat"), ("0.5", "dog")],
        "a": [("0.5", "mouse"), ("0.5", "house")],
        "cat": [("1.0", "END")],
        "dog": [("1.0", "END")],
        "mouse": [("1.0", "END")],
        "house": [("1.0", "END")],
    }
    return [(float(p), next_state) for p, next_state in transitions.get(state, [])]

# Beam Search futtatása
beam_width = 2
max_steps = 3
start_state = "START"

results = beam_search(simple_predict_next, start_state, beam_width, max_steps)

# Eredmények kiírása
print("Optimalizált utak:")
for prob, state in results:
    print(f"Ut: {state}, Valószínűség: {prob}")


"""
Miért hasznos a Beam Search?
Hatékonyabb, mint a teljes keresés, mert csak a legígéretesebb állapotokat követi.
Skálázható, különösen olyan problémákhoz, ahol az állapotok száma gyorsan növekszik (például természetes nyelvfeldolgozásban).
Ez a konkrét kód egy nyelvi modell egyszerű példája, de a Beam Search sok más alkalmazási területen is használható (pl. útvonaltervezés, gépi tanulás).
"""
