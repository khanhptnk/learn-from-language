"""Pretraining v3 data (docs/minicraft_design.md): a diverse behaviour prior from random courses of primitive skills.

Each world runs its own course: a sequence of options drawn from a small grammar, each option a short scripted motor
skill that knows no recipe and no goal --
  gather   walk to a resource cell of some kind (tree / reeds / clay pit / rock) and use it, 1-3 times
  craft    walk to a station, put a random 1-3 of the things held (select until the cursor shows the kind, then put),
           press craft, and sometimes take back what is left in the slot
  wander   a few random actions
  done     end the episode
with light curiosity biases that know nothing about recipes: resource kinds not held yet, stations visited least,
item kinds not put anywhere yet in the episode, and slot contents not tried at that station yet are preferred.
(Repeating an option that gained something, with probability 0.5 or 0.8, did not raise craft success or depth.) Recipes succeed only by chance, cheap ones often and deep ones rarely. Episodes that make a
held-out item are dropped (and counted). There are no requests and no text: language comes from post-training.
"""

from collections import Counter

import numpy as np

from gft.minicraft.core import (ACTION_NAMES, CLAYPIT, COLLECT, CRAFT, DONE, EV_COLLECT, EV_CRAFT, EV_CRAFT_FAIL,
                                ITEM_NAMES, N_WORLD_ACTIONS, PUT, RECIPES, REEDS, ROCK, SELECT, STATION_INDEX, STATIONS,
                                TAKE, TREE, USE)
from gft.minicraft.expert import _adjacent_dir, _adjacent_mask, _bfs_first_step
from gft.minicraft.goals import MAX_GOALS, Requests
from gft.minicraft.teacher import event_arrays, slot_steps

RESOURCES = (TREE, REEDS, CLAYPIT, ROCK)
RESOURCE_ITEM = np.array([COLLECT[k][0] for k in RESOURCES])
OPTION_P = dict(gather=0.45, craft=0.35, wander=0.15, done=0.05)   # per option draw; done only after MIN_STEPS
MIN_STEPS = 20


class PrimitiveExplorer:
    """act_fn for learn.run_episodes: every world follows its own random course of primitive options (module doc).
    `log` holds each world's options as (name, argument, first step) for coverage and tests."""

    def __init__(self, n, rng, option_p=OPTION_P):
        self.rng = rng
        self.names = list(option_p)
        self.p = np.array([option_p[k] for k in self.names], float)
        self.p /= self.p.sum()
        self.queue = [[] for _ in range(n)]           # per world: pending low-level steps of the current option
        self.option = [None] * n
        self.visits = np.zeros((n, len(STATIONS)))
        self.tried = [set() for _ in range(n)]        # (station, slot kinds) crafted at
        self.put_kinds = [set() for _ in range(n)]    # item kinds put anywhere so far
        self.log = [[] for _ in range(n)]

    def __call__(self, env, t):
        out = np.zeros(env.B, dtype=np.int64)
        for b in np.nonzero(~env.done)[0]:
            out[b] = self._act(env, b, t)
        return out

    # ------------------------------------------------------------ option choice
    def _new_option(self, env, b, t):
        while True:
            name = self.names[int(self.rng.choice(len(self.names), p=self.p))]
            if name == "done" and t < MIN_STEPS:
                continue
            if name == "craft" and env.inv[b].sum() == 0 and env.slot[b].sum() == 0:
                continue                              # nothing to put anywhere
            break
        arg = None
        if name == "gather":
            w = np.where(env.inv[b, RESOURCE_ITEM] > 0, 1.0, 3.0) * [(env.grid[b] == k).any() for k in RESOURCES]
            if w.sum() == 0:
                name = "wander"                       # nothing left to gather
            else:
                arg = (RESOURCES[int(self.rng.choice(len(RESOURCES), p=w / w.sum()))], int(self.rng.integers(1, 4)))
        if name == "craft":
            w = 1.0 / (1.0 + self.visits[b])
            arg = STATIONS[int(self.rng.choice(len(STATIONS), p=w / w.sum()))]
        if name == "wander":
            arg = int(self.rng.integers(3, 11))
        self.option[b] = [name, arg, 0]               # name, argument, progress counter
        self.log[b].append((name, arg, t))

    # ------------------------------------------------------------ primitives
    def _go_to(self, env, b, kind):
        """A move toward a cell next to one of `kind`, a turn to face it, or None once facing it."""
        pos = (int(env.pos[b, 0]), int(env.pos[b, 1]))
        d = _adjacent_dir(env.grid[b], *pos, kind)
        if d >= 0:
            return None if int(env.face[b]) == d else d   # moving into a non-passable cell only turns
        step = _bfs_first_step(env.grid[b], pos, _adjacent_mask(env.grid[b], kind))
        return -1 if step is None or step < 0 else step   # -1: unreachable

    def _act(self, env, b, t):
        if self.queue[b]:
            return self.queue[b].pop(0)
        if self.option[b] is None:
            self._new_option(env, b, t)
        while True:
            a = self._step_option(env, b, t)
            if a is not None:
                return a
            self._new_option(env, b, t)

    def _step_option(self, env, b, t):
        """The next action of world b's current option, or None when the option is finished."""
        name, arg, k = self.option[b]
        if name == "done":
            return DONE
        if name == "wander":
            if k >= arg:
                return None
            self.option[b][2] += 1
            return int(self.rng.integers(N_WORLD_ACTIONS))
        if name == "gather":
            kind, times = arg
            if k >= times or not (env.grid[b] == kind).any():
                return None
            a = self._go_to(env, b, kind)
            if a == -1:
                return None
            if a is None:                             # facing it: use it (one gather done)
                self.option[b][2] += 1
                return USE
            return a
        # craft at station `arg`: walk there, then put / craft / maybe take as one queued burst
        a = self._go_to(env, b, arg)
        if a == -1:
            return None
        if a is not None:
            return a
        if k:                                         # the burst was already queued and played
            return None
        self.option[b][2] = 1
        si = STATION_INDEX[arg]
        self.visits[b, si] += 1
        burst, kinds = self._put_burst(env, b, si)
        self.tried[b].add((int(arg), kinds))
        self.put_kinds[b].update(burst)
        queue = []
        cursor = int(env.cursor[b])
        inv = env.inv[b].copy()
        for j in burst:                               # select until the cursor shows j, then put
            while cursor != j:
                held = np.nonzero(inv > 0)[0]
                nxt = held[held > cursor]
                cursor = int(nxt[0]) if len(nxt) else int(held[0])
                queue.append(SELECT)
            queue.append(PUT)
            inv[j] -= 1
        queue.append(CRAFT)
        if self.rng.random() < 0.5:
            queue.append(TAKE)                         # take back whatever is left (nothing after a success)
        self.queue[b] = queue[1:]
        return queue[0]

    def _put_burst(self, env, b, si, tries=4):
        """Items (one entry per unit) to put at station si: 1-3 units of 1-2 kinds held (kinds not put yet weigh 3x),
        preferring a resulting slot content not yet crafted at this station. Returns (items, kinds in the slot
        afterwards)."""
        held = np.nonzero(env.inv[b] > 0)[0]
        w = np.array([1.0 if j in self.put_kinds[b] else 3.0 for j in held])
        best = None
        for _ in range(tries):
            burst = []
            if len(held):
                kinds = self.rng.choice(held, size=min(len(held), int(self.rng.integers(1, 3))), replace=False,
                                        p=w / w.sum())
                for j in kinds:
                    burst += [int(j)] * int(min(env.inv[b, j], self.rng.integers(1, 3)))
                burst = burst[:3]
            after = env.slot[b, si].copy()
            np.add.at(after, burst, 1)
            kinds = tuple(int(j) for j in np.nonzero(after)[0])
            best = best or (burst, kinds)
            if (STATIONS[si], kinds) not in self.tried[b]:
                return burst, kinds
        return best


def _new_stats():
    return dict(generated=0, dropped=0, steps=0, crafted=Counter(), collected=Counter(), craft_attempts=Counter(),
                craft_ok=Counter(), options=Counter(), actions=Counter(), length=[], events=[])


def _add_rollout(st, ev, acts, valid, held, option_log=None):
    """Account one rollout's episodes in st; returns the indices of the kept ones (no held-out item made)."""
    keep = []
    for e in range(valid.shape[0]):
        st["generated"] += 1
        typ, item, station = ev["type"][:, e], ev["item"][:, e], ev["station"][:, e]
        if set(int(i) for i in item[np.isin(typ, (EV_CRAFT, EV_COLLECT))]) & held:   # made or picked up
            st["dropped"] += 1
            continue
        keep.append(e)
        st["length"].append(int(valid[e].sum()))
        st["actions"].update(ACTION_NAMES[a] for a in acts[e][valid[e]])
        if option_log is not None:
            st["options"].update(o[0] for o in option_log[e])
        st["crafted"].update(ITEM_NAMES[i] for i in item[typ == EV_CRAFT])
        st["collected"].update(ITEM_NAMES[i] for i in item[typ == EV_COLLECT])
        slot = slot_steps(typ, item, station)
        for t in np.nonzero(np.isin(typ, (EV_CRAFT, EV_CRAFT_FAIL)) & (station >= 0))[0]:
            key = (int(station[t]), tuple(np.nonzero(slot[t] > 0)[0].tolist()))
            st["craft_attempts"][key] += 1
            st["craft_ok"][key] += int(typ[t] == EV_CRAFT)
        st["events"].append(int(np.isin(typ, (EV_COLLECT, EV_CRAFT, EV_CRAFT_FAIL)).sum()))
    return np.array(keep, dtype=np.int64)


def _fresh_worlds(rng, n):
    return Requests(np.full((n, MAX_GOALS), -1), rng.integers(0, 2 ** 31, size=n))   # no request: goals all -1


def explore_corpus(cfg, n_episodes, held_items, seed=0, chunk=256):
    """Pretraining v3 corpus: n_episodes primitive-explorer episodes on fresh worlds, keeping only those that never
    make a held-out item. Returns (steps: dict of grid / pf / state / a / b arrays for BC, coverage)."""
    from gft.minicraft.learn import run_episodes, stack_steps
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("grid", "pf", "state", "a", "b")}
    st = _new_stats()
    held = set(int(i) for i in held_items)
    n_kept = 0
    while st["generated"] < n_episodes:
        n = min(chunk, n_episodes - st["generated"])
        ex = PrimitiveExplorer(n, rng)
        env, log, snaps, acts, valid, _, _ = run_episodes(cfg, _fresh_worlds(rng, n), ex)
        keep = _add_rollout(st, event_arrays(log), acts, valid, held, ex.log)
        grid, pf, state, a, b, _ = stack_steps(snaps, acts, valid)
        new_id = np.full(n, -1)
        new_id[keep] = n_kept + np.arange(len(keep))
        n_kept += len(keep)
        sel = new_id[b] >= 0
        for k, v in (("grid", grid), ("pf", pf), ("state", state), ("a", a)):
            out[k].append(v[sel])
        out["b"].append(new_id[b[sel]])
    steps = {k: np.concatenate(v) for k, v in out.items()}
    st["steps"] = len(steps["a"])
    return steps, coverage(st)


def pretrain_explorer(steps, cfg, device, n_steps=10_000, batch=512, lr=1e-3, seed=0, arch=None):
    """Pretraining v3: unconditional BC of a CleanPolicy on an explore_corpus (no prompt: the text pathway stays
    untrained, as in learn.pretrain_base). Returns (model, losses every 100 steps)."""
    import torch
    import torch.nn.functional as F_
    from gft.minicraft.learn import CleanPolicy, obs_tensor, on_device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    grid, pf, state = on_device(device, steps["grid"], steps["pf"], steps["state"])
    a = torch.as_tensor(steps["a"], device=device)
    model = CleanPolicy(cfg.size, **(arch or {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for k in range(n_steps):
        i = torch.as_tensor(rng.integers(len(a), size=batch), device=device)
        x, iv = obs_tensor(grid[i], pf[i], state[i], device)
        loss = F_.cross_entropy(model(x, iv, None)[0], a[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if k % 100 == 0:
            losses.append(float(loss))
    return model, losses


def policy_coverage(cfg, model, n_episodes, held_items, device, seed=0, chunk=256):
    """The same coverage as explore_corpus for a policy's own rollouts (sampled actions, no prompt): does the cloned
    prior keep the explorer's diversity?"""
    import torch
    from gft.minicraft.learn import obs_tensor, run_episodes
    rng = np.random.default_rng(seed)
    st = _new_stats()
    held = set(int(i) for i in held_items)

    def act(env, t):
        x, iv = obs_tensor(env.grid, np.concatenate([env.pos, env.face[:, None]], 1), env.agent_state(), device)
        with torch.no_grad():
            return torch.distributions.Categorical(logits=model(x, iv, None)[0]).sample().cpu().numpy()
    while st["generated"] < n_episodes:
        n = min(chunk, n_episodes - st["generated"])
        env, log, _, acts, valid, _, _ = run_episodes(cfg, _fresh_worlds(rng, n), act, record=False)
        _add_rollout(st, event_arrays(log), acts, valid, held)
        st["steps"] += int(valid.sum())
    return coverage(st)


def coverage(st):
    """Summary of a corpus or rollout: drop rate, episode length, actions and options, how often each recipe was made
    by chance, and how many distinct (station, slot kinds) craft attempts were seen."""
    kept = st["generated"] - st["dropped"]
    n_act = max(sum(st["actions"].values()), 1)
    recipes = {ITEM_NAMES[r.output]: st["crafted"][ITEM_NAMES[r.output]] / max(kept, 1) for r in RECIPES}
    return dict(episodes=kept, generated=st["generated"], drop_rate=st["dropped"] / max(st["generated"], 1),
                steps=st["steps"], mean_length=float(np.mean(st["length"])) if st["length"] else 0.0,
                options=dict(st["options"]), actions={k: v / n_act for k, v in st["actions"].items()},
                crafted_per_episode=recipes,
                collected_per_episode={k: v / max(kept, 1) for k, v in st["collected"].items()},
                distinct_craft_attempts=len(st["craft_attempts"]),
                craft_success=sum(st["craft_ok"].values()) / max(sum(st["craft_attempts"].values()), 1),
                events_per_episode=float(np.mean(st["events"])) if st["events"] else 0.0)
