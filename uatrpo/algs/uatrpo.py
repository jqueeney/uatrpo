import numpy as np
import tensorflow as tf

from uatrpo.algs.trpo import TRPO
from uatrpo.common.update_utils import make_F

class UATRPO(TRPO):
    """Algorithm class for UATRPO."""

    def __init__(self,seed,env,actor,critic,runner,ac_kwargs,
        idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
        """Initializes UATRPO class. See TRPO for details."""
        super(UATRPO,self).__init__(seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    
    def _ac_setup(self):
        """Sets up actor and critic kwargs as class attributes."""
        super(UATRPO,self)._ac_setup()
       
        self.m = self.ac_kwargs['m']
        self.c = self.ac_kwargs['c']
        self.alpha = self.ac_kwargs['alpha']
        self.shrink = self.ac_kwargs['shrink']
        self.ua_trust_damp = self.ac_kwargs['ua_trust_damp']
        self.ua_nbatch = self.ac_kwargs['ua_nbatch']
        
        self.vectorized_map = self.ac_kwargs['vectorized_map']
        
        d = np.sum([np.prod(x.shape) for x in self.actor.trainable])
        self.G = tf.random.normal((d,self.m),dtype=tf.float32)
        self.Q = None

    def _actor_update(self,s_all,a_all,adv_all,neglogp_old_all,kl_info_all,
        weights_all):
        """Updates actor."""
        if self.trust_vary:
            neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
            offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)            
            eps_old = tf.reduce_mean(weights_all * tf.abs(offpol_ratio-1.))
            self.eps = np.maximum(self.eps_tv - eps_old,0.0)
            self.delta = np.square(self.eps) / 2

        F = make_F(self.actor,s_all,weights_all,self.trust_sub,damp=0.0)

        # Calculate basis Q for subspace
        if self.vectorized_map:
            Y = tf.linalg.matrix_transpose(
                tf.vectorized_map(F,tf.linalg.matrix_transpose(self.G)))
        else:
            Y = tf.linalg.matrix_transpose(
                tf.map_fn(F,tf.linalg.matrix_transpose(self.G)))

        Q, sing, _ = np.linalg.svd(Y,full_matrices=False)
        Q_thresh = (np.finfo(Y.dtype.as_numpy_dtype).eps * np.max(sing) 
            * np.max(Y.shape))
        Y_rank = np.sum(sing > Q_thresh)
        Q = Q[:,:Y_rank]
        self.Q = tf.convert_to_tensor(Q,dtype=tf.float32)

        # Calculate projected trust region matrix
        if self.vectorized_map:
            FQ = tf.linalg.matrix_transpose(
                tf.vectorized_map(F,tf.linalg.matrix_transpose(self.Q)))
        else:
            FQ = tf.linalg.matrix_transpose(
                tf.map_fn(F,tf.linalg.matrix_transpose(self.Q)))
        F_proj = tf.matmul(self.Q,FQ,transpose_a=True)

        if self.c > 0:
            pg_all_proj = self._get_neg_pg_all(s_all,a_all,adv_all,
                neglogp_old_all,weights_all,self.ua_nbatch,proj=True) * -1
            C_proj = pg_all_proj - tf.reduce_mean(pg_all_proj,axis=0)
            n_ua = C_proj.shape[0]
            S_proj = tf.matmul(C_proj,C_proj,transpose_a=True) / (n_ua-1)
            
            if self.shrink > 0:
                S_diag = tf.linalg.diag(tf.linalg.diag_part(S_proj))
                S_proj = self.shrink * S_diag + (1-self.shrink) * S_proj
        else:
            S_proj = tf.zeros_like(F_proj)
            n_ua = 1

        # Rn2 depends on reduced dimension of subspace spanned by Q
        Rn2 = (Y_rank + 2 * np.sqrt(Y_rank*np.log(1/self.alpha)) 
            + 2*np.log(1/self.alpha)) / n_ua

        M_proj = (F_proj + self.c * Rn2 * S_proj 
            + self.ua_trust_damp * tf.eye(F_proj.shape[0]))

        D,V = tf.linalg.eigh(M_proj)
        V_thresh = (np.finfo(M_proj.dtype.as_numpy_dtype).eps * np.max(D) 
            * np.max(M_proj.shape))
        M_proj_rank = np.sum(D > V_thresh)
        D=D[-M_proj_rank:]
        V = V[:,-M_proj_rank:]

        # Calculate update direction
        pg_vec = self._get_neg_pg(s_all,a_all,adv_all,neglogp_old_all,
            weights_all,flat=True) * -1
        if self.adversary_mult > 0:
            pg_all = self._get_neg_pg_all(s_all,a_all,adv_all,
                neglogp_old_all,weights_all,self.adversary_nbatch) * -1
            C_all = pg_all - tf.reduce_mean(pg_all,axis=0)
            n_adv = C_all.shape[0]
            pg_var = tf.reduce_sum(tf.square(C_all),axis=0) / (n_adv-1)
            pg_se = tf.sqrt(pg_var / n_adv)

            pg_adversary = self.adversary_mult * pg_se * tf.sign(pg_vec)

            pg_vec = pg_vec - pg_adversary.numpy()

        U = tf.matmul(self.Q,V)
        g_proj = tf.tensordot(pg_vec,U,axes=1)
        v_proj = g_proj / D
        v_Qproj = tf.tensordot(V,v_proj,axes=1) 
        v_flat = tf.tensordot(U,v_proj,axes=1)

        # Calculate update
        if np.allclose(g_proj,0) or self.delta==0.0:
            eta_v_flat = np.zeros_like(v_flat)
            eta_v_Qproj = tf.zeros_like(v_Qproj)
        else:
            vMv = tf.tensordot(v_Qproj,
                tf.tensordot(M_proj,v_Qproj,axes=1),axes=1)
            eta = np.sqrt(2*self.delta/vMv.numpy())
            eta_v_flat = eta * v_flat.numpy()
            eta_v_Qproj = eta * v_Qproj

        vFv = 0.5 * tf.tensordot(eta_v_Qproj,
            tf.tensordot(F_proj,eta_v_Qproj,axes=1),axes=1)
        vSv = 0.5 * tf.tensordot(eta_v_Qproj,
            tf.tensordot(S_proj,eta_v_Qproj,axes=1),axes=1)
        vMv = 0.5 * tf.tensordot(eta_v_Qproj,
            tf.tensordot(M_proj,eta_v_Qproj,axes=1),axes=1)

        if np.allclose(g_proj,0):
            delta_mult = 1.0
        else:
            delta_mult = (vFv / (vFv + self.c * Rn2 * vSv)).numpy()

        self._backtrack(eta_v_flat,s_all,a_all,adv_all,neglogp_old_all,
            kl_info_all,weights_all,delta_mult)

        # Log info
        log_info = {
            'vFv_pre':      vFv.numpy(),
            'vSv_pre':      vSv.numpy(),
            'cRn2vSv_pre':  self.c * Rn2 * vSv.numpy(),
            'vMv_pre':      vMv.numpy(),
            'cRn2':         self.c * Rn2,
            'Rn2':          Rn2,
            'Qdim':         Y_rank
        }
        self.logger.log_train(log_info)

    def _get_neg_pg_flatproj(self,inputs_all):
        """Wrapper for batch negative projected gradient calculations.
        
        Args:
            inputs_all (tuple): tuple of inputs for _get_neg_pg
        
        Returns:
            Negative flattened, projected gradient of surrogate objective.
        """

        neg_pg = self._get_neg_pg_flat(inputs_all)
        if (self.Q is not None):
            neg_pg = tf.tensordot(neg_pg,self.Q,axes=1)

        return neg_pg