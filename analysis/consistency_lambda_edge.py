"""Arm F's CV picked the largest lambda tried (1.0). Extend the grid to 3 and 10 on the same
folds to see whether the CV loss keeps falling. PYTHONPATH=analysis python analysis/consistency_lambda_edge.py"""
import numpy as np
import consistency as C

C.LAMBDAS = [1.0, 3.0, 10.0]
lam, cv = C.choose_lambda()
rel = cv / cv.min(0, keepdims=True)
for l, r, m in zip(C.LAMBDAS, rel.mean(1), np.median(cv, 1)):
    print(f'lambda {l:5}: mean relative CV loss {r:.4f}, median CV loss {m:.5f}')
print('share of cells best at each lambda', np.bincount(cv.argmin(0), minlength=len(C.LAMBDAS)) / cv.shape[1])
np.savez('analysis/consistency/lambda_edge.npz', lambdas=C.LAMBDAS, cv=cv)
