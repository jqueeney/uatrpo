"""Interface to all algorithm files."""
from uatrpo.algs.trpo import TRPO
from uatrpo.algs.uatrpo import UATRPO

gen_algs = ['getrpo','geuatrpo']

def init_alg(sim_seed,env,actor,critic,runner,ac_kwargs,alg_name,
    idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
    """Initializes algorithm."""

    if alg_name in ['trpo','getrpo']:
        alg = TRPO(sim_seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    elif alg_name in ['uatrpo','geuatrpo']:
        alg = UATRPO(sim_seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    else:
        raise ValueError('invalid alg_name')

    return alg