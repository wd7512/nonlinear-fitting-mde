# Gate baseline, 2026-11-27

`symbfitrs` on the frozen gate tiers. Apple Silicon, one thread, recorded under a load average of
about 2.5, so compare timings with runs made under similar load.
Tier files are not committed (they embed third-party data); `python -m bench gate build`
recreates them from local runs. References depend on those runs, so check the hashes before
comparing.

```
8381eb80847d32e4bdf987824b6aa27c6357a1ad32ee4929acaa50ecd03cc518  dev.json
c60da23149fcd5870ce72d45878b763513ddc4251c4ae87577ebfd563c2bbafb  quick.json
70c9ba5b49b843f4c31d73229ecaf02983940ba106068d64a722f976c40e4f56  rank.json
e7646ce9d99f3b01e082cc9d6ccbc0a65201570ec309a1ca0d5bc53b9f10a88b  test.json
```

## quick

success (mean over suites; per-suite columns at the main checkpoint)

| solver    | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms synthetic_scale   |
|:----------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|
| symbfitrs | 85.4%  | 89.6%  | 91.7%   | 91.7%   | 100.0%           | 75.0%      | 93.8%                  |

speed and failures

| solver    |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------|----------------:|----------------------:|-----------:|------------------:|-------:|
| symbfitrs |           0.094 |                 0.036 |          0 |                 0 |     48 |

## dev

success (mean over suites; per-suite columns at the main checkpoint)

| solver    | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms fitbenchmarking   | 5 ms sagefit_spb   | 5 ms synthetic_scale   | 5 ms stringer2019_gratings   |
|:----------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|:-------------------|:-----------------------|:-----------------------------|
| symbfitrs | 72.4%  | 87.0%  | 88.5%   | 89.0%   | 100.0%           | 90.0%      | 100.0%                 | 52.4%              | 95.3%                  | 84.0%                        |

speed and failures

| solver    |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------|----------------:|----------------------:|-----------:|------------------:|-------:|
| symbfitrs |           2.243 |                 0.287 |          1 |                 0 |    555 |

## test

success (mean over suites; per-suite columns at the main checkpoint)

| solver    | 1 ms   | 5 ms   | 10 ms   | 25 ms   | 5 ms nist_strd   | 5 ms mgh   | 5 ms fitbenchmarking   | 5 ms sagefit_spb   | 5 ms synthetic_scale   | 5 ms stringer2019_gratings   |
|:----------|:-------|:-------|:--------|:--------|:-----------------|:-----------|:-----------------------|:-------------------|:-----------------------|:-----------------------------|
| symbfitrs | 75.5%  | 85.6%  | 87.4%   | 87.7%   | 100.0%           | 93.0%      | 96.2%                  | 39.5%              | 100.0%                 | 84.7%                        |

speed and failures

| solver    |   median fit ms |   median ms to target |   failures |   overruns (>10%) |   fits |
|:----------|----------------:|----------------------:|-----------:|------------------:|-------:|
| symbfitrs |           4.542 |                 0.195 |          3 |                 0 |    521 |

## rank

sagefit_spb: 18 curves

|           |   tau @ 10 ms |   top1 @ 10 ms |   lost @ 10 ms |   tau @ 100 ms |   top1 @ 100 ms |   lost @ 100 ms |
|:----------|--------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| symbfitrs |         0.736 |          0.778 |          0.392 |          0.755 |           0.833 |           0.360 |

stringer2019_gratings: 100 curves

|           |   tau @ 10 ms |   top1 @ 10 ms |   lost @ 10 ms |   tau @ 100 ms |   top1 @ 100 ms |   lost @ 100 ms |
|:----------|--------------:|---------------:|---------------:|---------------:|----------------:|----------------:|
| symbfitrs |         0.689 |          0.710 |          0.000 |          0.709 |           0.730 |           0.000 |
