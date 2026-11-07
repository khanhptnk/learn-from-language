import numpy as np
import torch

from gft.algos.common import execute, rollout
from gft.algos.gft_exact import exact_slot_recursion, run_gft_exact
from gft.algos.rl_neural import PolicyGradient, RLConfig
from gft.envs.bandit import audit_instance
from gft.envs.reqworld import ReqWorldConfig, arg_for, noisy_args, outcome_perm, sample_requests
from gft.feedback.logical_form import NEG, POS, TeacherConfig, audit, ideal
from gft.models.set_policy import SetPolicy


def test_sigma_is_a_permutation_and_inverts():
    cfg = ReqWorldConfig(n_slots=5, n_values=7, m=3)
    sigma, inv = outcome_perm(cfg), arg_for(cfg)
    for k in range(cfg.n_slots):
        assert sorted(sigma[k].tolist()) == list(range(7))
        assert torch.equal(sigma[k][inv[k]], torch.arange(7))
    assert not torch.equal(sigma[0], torch.arange(7))  # scrambled
    assert torch.equal(outcome_perm(ReqWorldConfig(n_values=7, scramble=False))[0], torch.arange(7))


def test_correct_args_satisfy_and_feedback_reports_outcomes():
    cfg = ReqWorldConfig(n_slots=6, n_values=9, m=4)
    g = torch.Generator().manual_seed(0)
    req = sample_requests(cfg, 64, g)
    args = arg_for(cfg)[req.slots, req.values]
    ep = execute(cfg, req, req.slots.clone(), args, torch.full((64,), 4))
    assert ep.sat.all()
    wrong = (args + 1) % cfg.n_values
    ep = execute(cfg, req, req.slots.clone(), wrong, torch.full((64,), 4))
    assert not ep.sat.any()
    msg = audit(req, ep.sat, ep.actual, cfg.n_values, TeacherConfig(1.0, 1.0, True), g)
    assert (msg.pol == NEG).all()
    assert torch.equal(msg.values, outcome_perm(cfg)[req.slots, wrong])


def test_ideal_feedback_reports_targets_as_satisfied():
    cfg = ReqWorldConfig(n_values=9, m=4)
    g = torch.Generator().manual_seed(1)
    req = sample_requests(cfg, 200, g)
    f = ideal(req, cfg.n_values, TeacherConfig(0.5, 0.9, True), g)
    assert (f.pol[f.mask] == POS).all()
    assert torch.equal(f.values[f.mask], req.values[f.mask])
    assert 0.4 < f.mask.float().mean() < 0.6


def test_noisy_args_competence():
    cfg = ReqWorldConfig(n_values=16, m=4)
    g = torch.Generator().manual_seed(2)
    req = sample_requests(cfg, 4000, g)
    ep = execute(cfg, req, req.slots.clone(), noisy_args(cfg, req, 0.3, g), torch.full((4000,), 4))
    assert abs(ep.sat.float().mean().item() - 0.3) < 0.02


def test_set_policy_interface_and_align_reward():
    cfg = ReqWorldConfig(n_values=8, m=3)
    g = torch.Generator().manual_seed(3)
    pol = SetPolicy(cfg, value_head=True)
    req = sample_requests(cfg, 16, g).repeat_interleave(4)
    ep = rollout(pol, cfg, req, None, g)
    lp, valid = pol.log_probs(req, None, ep.act_slots, ep.act_values, ep.lengths)
    assert lp.shape == (64, 3) and valid.all()
    msg = audit(req, ep.sat, ep.actual, cfg.n_values, TeacherConfig(), g)
    lp_fb, _ = pol.log_probs(req, msg, ep.act_slots, ep.act_values, ep.lengths)
    assert not torch.allclose(lp, lp_fb)
    pg = PolicyGradient(pol, cfg, TeacherConfig(), RLConfig(algo="grpo", reward="align", group_size=4), g)
    r = pg._reward(ep)
    assert r.shape == (64, 3)
    adv = pg._advantage(r)
    assert adv.shape == (64, 3)
    np.testing.assert_allclose(adv.view(16, 4, 3).sum(1), 0.0, atol=1e-4)


def test_exact_slot_recursion_matches_enumerated_gft():
    # One requirement, 3 outcomes, factorized base: compare closed form with the enumerated update.
    R = np.array([[1], [0], [0]])
    p0 = np.array([0.2, 0.5, 0.3])
    for ks, kv in [(0.5, 0.5), (0.3, 0.8), (1.0, 1.0)]:
        inst = audit_instance(R, np.array([ks]), p0)
        if kv != ks:  # audit_instance uses outcome-independent kappa; skip biased case there
            continue
        run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=10)
        np.testing.assert_allclose(run.success, exact_slot_recursion(0.2, ks, kv, 10), atol=1e-12)
