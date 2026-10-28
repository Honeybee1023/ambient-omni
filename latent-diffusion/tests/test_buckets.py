"""Checks for training/buckets.py (run: python -m pytest tests/test_buckets.py, from latent-diffusion/)."""
import itertools, math, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training import buckets as bk

PM, PS = -0.4, 1.0


def test_level_to_sigma_matches_doc():
    # the plan doc: levels .85/.90/.95 -> sigma 1.9/2.4/3.5 (1.89/2.41/3.47) on ImageNet (P_mean -0.4, P_std 1.0)
    got = [round(bk.level_to_sigma(l, PM, PS), 2) for l in (0.85, 0.90, 0.95)]
    assert got == [1.89, 2.41, 3.47]
    assert bk.level_to_sigma(0.0, PM, PS) == 0.0 and bk.level_to_sigma(1.0, PM, PS) == math.inf


def test_step_function_floors():
    spec = {'type': 'per_bucket', 'buckets': {'1': {'phases': [[0, 0.0], [0.5, 0.9]]},
                                              '2': {'phases': [[0, 0.0], [0.3, 0.95]]},
                                              '3': {'phases': [[0, 0.0], [0.1, 1.0]]}}}
    s = bk.BucketSchedule(spec, 3, PM, PS, horizon_nimg=1000)
    assert list(s.floors(0)) == [0, 0, 0, 0]
    f = s.floors(400)    # progress .4: b2 and b3 have jumped, b1 not yet
    assert f[0] == 0 and f[1] == 0 and round(f[2], 2) == 3.47 and f[3] == math.inf
    f = s.floors(500)
    assert round(f[1], 2) == 2.41


def test_fixed_spec_and_validation():
    s = bk.BucketSchedule({'type': 'fixed', 'sigma_min': {'1': 0.2, '2': 0.2, '3': 0.2}}, 3, PM, PS)
    assert list(s.floors(123)) == [0, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        bk.BucketSchedule({'type': 'fixed', 'sigma_min': {'1': 0.2}}, 3, PM, PS)
    with pytest.raises(ValueError):
        bk.BucketSchedule({'type': 'per_bucket', 'buckets': {str(b): {'phases': [[0.1, 0.9]]} for b in (1, 2, 3)}},
                          3, PM, PS, horizon_nimg=10)


def draw(sampler, n):
    out = list(itertools.islice(iter(sampler), n))
    dec = [sampler.decided.popleft() for _ in out]
    assert [d[0] for d in dec] == out
    return np.array(out), np.array([d[1] for d in dec]), np.array([d[2] for d in dec])


def test_sampler_keeps_noise_distribution_and_respects_floors():
    rng = np.random.RandomState(0)
    bucket = rng.choice(4, size=20000, p=[.1, .5, .26, .14]).astype(np.int8)
    s = bk.BucketSchedule({'type': 'fixed', 'sigma_min': {'1': 0.2, '2': 1.0, '3': math.inf}}, 3, PM, PS)
    smp = bk.BucketSampler(len(bucket), bucket, s, seed=1, P_mean=PM, P_std=PS)
    idx, sig, fl = draw(smp, 30000)
    # every sample is used above its floor, and its floor is its bucket's floor
    assert np.all(sig > fl)
    assert np.allclose(fl, s.floors(0)[bucket[idx]])
    assert not np.any(bucket[idx] == 3)                      # floor inf -> never used
    # sigma marginal is EDM2's log-normal
    z = (np.log(sig) - PM) / PS
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    # at low sigma only clean + bucket 1 above .2
    assert set(np.unique(bucket[idx[sig < 0.2]])) == {0}


def test_sampler_schedule_change_and_subset():
    bucket = np.array([0, 1, 2, 3] * 500, dtype=np.int8)
    spec = {'type': 'per_bucket', 'buckets': {str(b): {'phases': [[0, 0.0], [0.5, 1.0]]} for b in (1, 2, 3)}}
    s = bk.BucketSchedule(spec, 3, PM, PS, horizon_nimg=100)
    sub = np.zeros(len(bucket), bool); sub[:400] = True
    smp = bk.BucketSampler(len(bucket), bucket, s, seed=0, P_mean=PM, P_std=PS, subset=sub)
    it = iter(smp)
    first = [next(it) for _ in range(200)]
    assert max(first) < 400 and len(set(bucket[first])) == 4    # before the jump: all buckets, as clean
    smp.cur_nimg = 60                                          # past the jump: bad buckets never used
    later = [next(it) for _ in range(400)][50:]                # skip what was decided before the change
    assert set(bucket[later]) == {0}


def test_resume_position_continues_walk():
    bucket = np.zeros(1000, np.int8)
    s = bk.BucketSchedule({'type': 'fixed', 'sigma_min': {}}, 0, PM, PS)
    a = bk.BucketSampler(1000, bucket, s, seed=3, P_mean=PM, P_std=PS)
    idx, _, _ = draw(a, 2500)
    b = bk.BucketSampler(1000, bucket, s, seed=3, P_mean=PM, P_std=PS, start_pos=1500)
    idx2, _, _ = draw(b, 1000)
    assert list(idx2) == list(idx[1500:2500])      # all clean: walk position == count, same order (epochs 1-2)
