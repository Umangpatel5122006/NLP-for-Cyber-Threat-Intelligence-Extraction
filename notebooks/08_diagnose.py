
"""
the CRF predicts few entities, but we don’t know why. This script answers three questions at once:

Train vs dev score. A high train score with a low dev score means the model memorizes and doesn’t generalize. Low on both means our clues are too weak.
Predicted vs true entity counts. This shows how cautious the model is.
Relaxed recall and precision, plus 10 real missed entities, so we can see what kind of mistakes these are.


Result:


Train F1 (first 2000 sentences): 0.926
Dev F1: 0.18

True entities: 903   Predicted entities: 516
Relaxed recall:    0.208
Relaxed precision: 0.364

Examples of missed entities:
  [THREAT_ACTOR] 'Lazarus'  in: A cascade of compromise : unveiling Lazarus ’ new campaign securelist.com/unveiling-lazarus-new-campaign/11088
  [MALWARE_TOOL] 'Lazarus malware'  in: A cascade of compromise : unveiling Lazarus ’ new campaign securelist.com/unveiling-lazarus-new-campaign/11088
  [THREAT_ACTOR] 'the threat actor'  in: What ’s remarkable is that these software vulnerabilities were not new , and despite warnings and patches from
  [THREAT_ACTOR] 'threat actor'  in: This recurring breach suggested a persistent and determined threat actor with the likely objective of stealing
  [THREAT_ACTOR] 'The adversary'  in: The adversary demonstrated a high level of sophistication , employing advanced evasion techniques and introduc
  [MALWARE_TOOL] 'SIGNBT malware'  in: The adversary demonstrated a high level of sophistication , employing advanced evasion techniques and introduc
  [MALWARE_TOOL] 'SIGNBT malware'  in: In one instance , while examining the memory of the compromised security software from a victim ’s system , we
  [MALWARE_TOOL] 'shellcode'  in: This shellcode was responsible for launching a Windows executable file directly in memory .
  [INFRASTRUCTURE] 'spoolsv.exe'  in: These include the creation of a file called ualapi.dll in the system folder , which is automatically loadedby
  [INFRASTRUCTURE] 'registry entries'  in: Additionally , in several instances , registry entries were recorded to execute legitimate files for the purpo


"""

from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities

from cti.models.crf import CRFTagger
from cti.models.data import load_bio

LAYER = "entities"
tx, ty = load_bio(LAYER, "train")
dx, dy = load_bio(LAYER, "dev")

model = CRFTagger()
model.fit(tx, ty)
pred = model.predict(dx)

print("Train F1 (first 2000 sentences):", round(f1_score(ty[:2000], model.predict(tx[:2000])), 3))
print("Dev F1:", round(f1_score(dy, pred), 3))


def overlap(a, b):                      # same type and the spans touch
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


n_gold = n_pred = hit_gold = hit_pred = 0
missed = []
for i, (g, p) in enumerate(zip(dy, pred)):
    G, P = get_entities(g), get_entities(p)
    n_gold += len(G)
    n_pred += len(P)
    hit_pred += sum(any(overlap(pe, ge) for ge in G) for pe in P)
    for ge in G:
        if any(overlap(ge, pe) for pe in P):
            hit_gold += 1
        elif len(missed) < 10:
            missed.append((ge[0], " ".join(dx[i][ge[1]:ge[2] + 1]), " ".join(dx[i])))

print(f"\nTrue entities: {n_gold}   Predicted entities: {n_pred}")
print(f"Relaxed recall:    {hit_gold / n_gold:.3f}")
print(f"Relaxed precision: {hit_pred / n_pred:.3f}")

print("\nExamples of missed entities:")
for label, span, sent in missed:
    print(f"  [{label}] {span!r}  in: {sent[:110]}")