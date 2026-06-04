# Analysis: exp002_ddpg_sparse_progress
Total training episodes: **831**
## Headline
- Success rate, **last 100 eps: 31%**  (overall: 25%)
- Failure breakdown, last 100 eps: wall 69% | obstacle 0% | timeout 0% | tumble 0%

## Success-rate curve (per 100-ep block)
```
ep    1-100 :   2% 
ep  101-200 :  23% ####
ep  201-300 :  29% #####
ep  301-400 :  11% ##
ep  401-500 :  33% ######
ep  501-600 :  27% #####
ep  601-700 :  39% #######
ep  701-800 :  33% ######
ep  801-831 :  32% ######
```

## Where it's lacking (failure modes, whole run)
- COLL_WALL: 71%
- COLL_OBST: 0%
- TIMEOUT: 3%
- TUMBLE: 1%

## Verdict
- LEARNING — success rose 10% -> 33% (first vs last fifth).
- Dominant failure (last 100): **COLL_WALL** (69%) -> primary thing to address next.
