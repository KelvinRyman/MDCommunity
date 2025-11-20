import os
import sys
import numpy as np

COMMUNITY_PATH = os.path.join(os.path.dirname(__file__), "MultiDismantler_community")
if COMMUNITY_PATH not in sys.path:
    sys.path.append(COMMUNITY_PATH)

from graph import Graph


def main():
    g = Graph(30)
    feats = getattr(g, "community_features", None)
    if feats is None:
        print("community_features missing")
        return
    print("community_features shape:", feats.shape)
    print("first 5 rows:\n", feats[:5])
    print("min:", feats.min(), "max:", feats.max())
    in_range = (feats >= 0).all() and (feats <= 1).all()
    print("values within [0,1]:", in_range)


if __name__ == "__main__":
    main()
