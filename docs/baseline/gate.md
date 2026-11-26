# Gate baseline, 2026-11-26

`symbfitrs` on the frozen gate tiers. Apple Silicon, one thread, recorded under a load average of
about 3.5, so compare timings with runs made under similar load.
Tier files are not committed (they embed third-party data); `python -m bench gate build`
recreates them from local runs. References depend on those runs, so check the hashes before
comparing.

```
986be03c8989bdd1dcaafcb0f60c83b565fc84fab239f9ed3eb01529da08a29d  dev.json
da15dd32e3af0c41478a97d56c5d1c16b04c2cf19e16d5e62dd4d1d207996832  quick.json
10a8226aa65f276fed95f28af3d0694ade777861991b14ffa7ce74a173167b4c  rank.json
ed878ce5103d6bfdac280d91cd5c6d20a92802e0d5aa90da74e38ff27ba05cbf  test.json
```

## quick

success (mean over suites; per-suite columns at the main checkpoint)

| solver    | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms synthetic_scale   |
|:----------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|
| symbfitrs | 85.4%  | 87.5%  | 89.6%   | 91.7%   | 100.0%           | 75.0%      | 87.5%                  |

speed and failures

| solver    |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------|----------------:|----------------------:|-----------:|------------------:|-------:|
| symbfitrs |           0.133 |                 0.053 |          0 |                 0 |     48 |

## dev

success (mean over suites; per-suite columns at the main checkpoint)

| solver    | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms fitbenchmarking   | 5 ms sagefit_spb   | 5 ms synthetic_scale   | 5 ms stringer2019_gratings   |
|:----------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|:-------------------|:-----------------------|:-----------------------------|
| symbfitrs | 67.9%  | 85.1%  | 86.1%   | 87.6%   | 100.0%           | 89.5%      | 100.0%                 | 49.2%              | 94.7%                  | 77.3%                        |

speed and failures

| solver    |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------|----------------:|----------------------:|-----------:|------------------:|-------:|
| symbfitrs |           3.566 |                  0.33 |          0 |                 0 |    552 |

## test

success (mean over suites; per-suite columns at the main checkpoint)

| solver    | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms fitbenchmarking   | 5 ms sagefit_spb   | 5 ms synthetic_scale   | 5 ms stringer2019_gratings   |
|:----------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|:-------------------|:-----------------------|:-----------------------------|
| symbfitrs | 71.1%  | 83.3%  | 85.2%   | 86.3%   | 100.0%           | 93.0%      | 92.3%                  | 36.3%              | 100.0%                 | 78.0%                        |

speed and failures

| solver    |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------|----------------:|----------------------:|-----------:|------------------:|-------:|
| symbfitrs |           7.002 |                 0.256 |          1 |                 0 |    521 |

## rank

sagefit_spb: 18 curves

|           |   tau @ 10 ms |   top1 @ 10 ms |   lost @ 10 ms |   tau @ 100 ms |   top1 @ 100 ms |   lost @ 100 ms |
|:----------|--------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| symbfitrs |         0.692 |          0.667 |          0.448 |          0.768 |           0.722 |           0.384 |

stringer2019_gratings: 100 curves

|           |   tau @ 10 ms |   top1 @ 10 ms |   lost @ 10 ms |   tau @ 100 ms |   top1 @ 100 ms |   lost @ 100 ms |
|:----------|--------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| symbfitrs |         0.620 |          0.640 |          0.000 |          0.674 |           0.710 |           0.000 |

