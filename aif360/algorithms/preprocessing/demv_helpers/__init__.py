import pandas as pd
import numpy as np

def _balance_set(w_exp, w_obs, df: pd.DataFrame, tot_df, round_level=None, debug=False, k=-1):
    if w_obs == 0:
        return df, 0, 0
    disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
    disparity = [disp]
    i = 0
    while disp != 1 and i != k and w_obs != 0:
        if w_exp / w_obs > 1:
            df = pd.concat([df, df.sample()])
        elif w_exp / w_obs < 1:
            df = df.drop(df.sample().index, axis=0)
        w_obs = len(df) / len(tot_df)
        disp = round(
            w_exp / w_obs, round_level) if round_level else w_exp / w_obs
        disparity.append(disp)
        if debug:
            print(w_exp / w_obs)
        i += 1
    if i == k:
        print('Warning: max iterations reached')
    return df, disparity, i


def sample(d: pd.DataFrame, s_vars: list, label: str, round_level: float, debug: bool = False,
            i: int = 0, G=None, cond: bool = True, stop: int = 10000):
    if G is None:
        G = []
    d = d.copy()
    n = len(s_vars)
    disparities = []
    iter = 0
    if i == n:
        for l in np.unique(d[label]):
            g = d[(cond) & (d[label] == l)]
            if len(g) > 0:
                w_exp = (len(d[cond]) / len(d)) * \
                    (len(d[d[label] == l]) / len(d))
                w_obs = len(g) / len(d)
                g_new, disp, k = _balance_set(
                    w_exp, w_obs, g, d, round_level, debug, stop)
                g_new = g_new.astype(g.dtypes.to_dict())
                disparities.append(disp)
                G.append(g_new)
                iter = max(iter, k)
        return G, iter, disparities
    else:
        s = s_vars[i]
        i = i + 1
        G1, k1, disp1 = sample(d, s_vars, label, round_level, debug, i,
                                G.copy(), cond=cond & (d[s] == 0), stop=stop)
        G2, k2, disp2 = sample(d, s_vars, label, round_level, debug, i,
                                G.copy(), cond=cond & (d[s] == 1), stop=stop)
        G += G1
        G += G2
        iter = max([iter, k1, k2])
        new_disps = disp1 + disp2
        disparities.append(new_disps)
        limit = 1
        for s in s_vars:
            limit *= len(np.unique(d[s]))
        if len(G) == limit * len(np.unique(d[label])):
            return pd.DataFrame(G.pop().append([g for g in G]).sample(frac=1, random_state=2)), disparities, iter
        else:
            return G, iter, disparities
