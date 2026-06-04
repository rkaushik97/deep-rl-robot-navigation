# Analysis: exp001_ddpg_sparse
Total training episodes: **300**
## Headline
- Success rate, **last 100 eps: 0%**  (overall: 0%)
- Failure breakdown, last 100 eps: wall 84% | obstacle 0% | timeout 16% | tumble 0%

## Success-rate curve (per 100-ep block)
```
ep    1-100 :   1% 
ep  101-200 :   0% 
ep  201-300 :   0% 
```

## Where it's lacking (failure modes, whole run)
- COLL_WALL: 84%
- COLL_OBST: 0%
- TIMEOUT: 16%
- TUMBLE: 0%

## Verdict
- FLAT — success 2% -> 0% (no clear learning).
- Dominant failure (last 100): **COLL_WALL** (84%) -> primary thing to address next.
