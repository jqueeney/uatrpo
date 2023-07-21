import numpy as np
import tensorflow as tf

from uatrpo.algs.base_alg import BaseAlg
from uatrpo.common.ac_utils import list_to_flat
from uatrpo.common.update_utils import cg, make_F

class TRPO(BaseAlg):
    """Algorithm class for TRPO."""

    def __init__(self,seed,env,actor,critic,runner,ac_kwargs,
        idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
        """Initializes TRPO class. See BaseAlg for details."""
        super(TRPO,self).__init__(seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    
    def _ac_setup(self):
        """Sets up actor and critic kwargs as class attributes."""
        super(TRPO,self)._ac_setup()
       
        self.delta = np.square(self.eps) / 2

        self.cg_it = self.ac_kwargs['cg_it']
        self.trust_sub = self.ac_kwargs['trust_sub']
        self.kl_maxfactor = self.ac_kwargs['kl_maxfactor']
        self.trust_damp = self.ac_kwargs['trust_damp']

        self.adversary_mult = self.ac_kwargs['adversary_mult']
        self.adversary_nbatch = self.ac_kwargs['adversary_nbatch']

    def _update(self):
        """Updates actor and critic."""
        data_all = self.runner.get_update_info(self.actor,self.critic)
        (s_all, a_all, adv_all, rtg_all, neglogp_old_all, kl_info_all, 
            weights_all) = data_all
        neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
        offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)

        # Advantage normalization
        adv_mean = (np.mean(offpol_ratio * weights_all * adv_all) / 
            np.mean(offpol_ratio * weights_all))
        adv_std = np.std(offpol_ratio * weights_all * adv_all) + 1e-8

        if self.adv_center:
            adv_all = adv_all - adv_mean 
        if self.adv_scale:
            adv_all = adv_all / adv_std
        if self.adv_clip:
            adv_all = np.clip(adv_all,-self.adv_clip,self.adv_clip)

        self._critic_update(s_all,rtg_all,weights_all)
        self._actor_update(s_all,a_all,adv_all,neglogp_old_all,kl_info_all,
            weights_all)

    def _actor_update(self,s_all,a_all,adv_all,neglogp_old_all,kl_info_all,
        weights_all):
        """Updates actor.
        
        Args:
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            kl_info_all (np.ndarray): info needed to calculate KL divergence
            weights_all (np.ndarray): policy weights
        """
        if self.trust_vary:
            neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
            offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)            
            eps_old = tf.reduce_mean(weights_all * tf.abs(offpol_ratio-1.))
            self.eps = np.maximum(self.eps_tv - eps_old,0.0)
            self.delta = np.square(self.eps) / 2

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

        if np.allclose(pg_vec,0) or self.delta==0.0:
            eta_v_flat = np.zeros_like(pg_vec)
        else:
            F = make_F(self.actor,s_all,weights_all,self.trust_sub,
                self.trust_damp)
            v_flat = cg(F,pg_vec,cg_iters=self.cg_it)

            vFv = np.dot(v_flat,F(v_flat))
            eta = np.sqrt(2*self.delta/vFv)
            eta_v_flat = eta * v_flat

        self._backtrack(eta_v_flat,s_all,a_all,adv_all,neglogp_old_all,
            kl_info_all,weights_all)

    def _get_neg_pg(self,s_all,a_all,adv_all,neglogp_old_all,
        weights_all,flat=True):
        """Calculates negative gradient of surrogate objective.

        Args:
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            weights_all (np.ndarray): policy weights
            flat (bool): if True, flattens gradient
        
        Returns:
            Negative gradient of surrogate objective w.r.t. policy parameters.
        """

        with tf.GradientTape() as tape:
            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            
            pg_loss_surr = ratio * adv_all * -1
            pg_loss = tf.reduce_mean(pg_loss_surr*weights_all)
        
        neg_pg = tape.gradient(pg_loss,self.actor.trainable)

        if flat:
            neg_pg = list_to_flat(neg_pg)

        return neg_pg

    def _get_neg_pg_flat(self,inputs_all):
        """Wrapper for batch negative gradient calculations.
        
        Args:
            inputs_all (tuple): tuple of inputs for _get_neg_pg
        
        Returns:
            Negative flattened gradient of surrogate objective.
        """

        s_all, a_all, adv_all, neglogp_old_all, weights_all = inputs_all

        neg_pg = self._get_neg_pg(s_all,a_all,adv_all,neglogp_old_all,
            weights_all,flat=False)

        neg_pg = list_to_flat(neg_pg,use_tf=True)

        return neg_pg

    def _get_neg_pg_flatproj(self,inputs_all):
        """Wrapper for batch negative projected gradient calculations."""
        raise NotImplementedError
    
    def _get_neg_pg_all(self,s_all,a_all,adv_all,neglogp_old_all,
        weights_all,grad_batch=None,proj=False):
        """Calculates batch of negative gradients of surrogate objective.
        
        Args:
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            weights_all (np.ndarray): policy weights
            grad_batch (int): number of gradient minibatches
            proj (bool): if True, project gradient onto subspace
        
        Returns:
            All negative gradients of surrogate objective.
        """

        n_samples = len(adv_all)
        
        if grad_batch and (n_samples > grad_batch):
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            sections = np.arange(0,n_samples,grad_batch)[1:]

            batches = np.array_split(idx,sections)
            if (n_samples % grad_batch != 0):
                batches = batches[:-1]

            s_batch = [s_all[batch_idx] for batch_idx in batches]
            a_batch = [a_all[batch_idx] for batch_idx in batches]
            adv_batch = [adv_all[batch_idx] for batch_idx in batches]
            neglogp_old_batch = [neglogp_old_all[batch_idx] for batch_idx in batches]
            weights_batch = [weights_all[batch_idx] for batch_idx in batches]
        else:
            s_batch = s_all
            a_batch = a_all
            adv_batch = adv_all
            neglogp_old_batch = neglogp_old_all
            weights_batch = weights_all

        inputs_sub = (s_batch,a_batch,adv_batch,neglogp_old_batch,weights_batch)

        if proj:
            return tf.vectorized_map(self._get_neg_pg_flatproj,inputs_sub)
        else:
            return tf.vectorized_map(self._get_neg_pg_flat,inputs_sub)

    def _backtrack(self,eta_v_flat,s_all,a_all,adv_all,neglogp_old_all,
        kl_info_all,weights_all,delta_mult=1.0):
        """Performs backtracking line search and updates policy.

        Args:
            eta_v_flat (np.ndarray): pre backtrack flattened policy update
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            kl_info_all (np.ndarray): info needed to calculate KL divergence
            weights_all (np.ndarray): policy weights
            delta_mult (float): portion of trust region attributed to KL
        """
        # Current policy info
        ent = tf.reduce_mean(weights_all * self.actor.entropy(s_all))
        kl_info_ref = self.actor.get_kl_info(s_all)
        actor_weights_pik = self.actor.get_weights()

        neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
        offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)

        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        surr_before = tf.reduce_mean(ratio * adv_all * weights_all)

        # Update
        self.actor.set_weights(eta_v_flat,from_flat=True,increment=True)
                
        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        surr = tf.reduce_mean(ratio * adv_all * weights_all)
        improve = surr - surr_before

        kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_ref))
        pen_kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_all))
        kl_pre = kl.numpy()
        pen_kl_pre = pen_kl.numpy()

        ratio_diff = tf.abs(ratio - offpol_ratio)
        tv = 0.5 * tf.reduce_mean(weights_all * ratio_diff)
        pen = 0.5 * tf.reduce_mean(weights_all * tf.abs(ratio-1.))
        tv_pre = tv.numpy()
        pen_pre = pen.numpy()

        kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_ref,direction='reverse'))
        pen_kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_all,direction='reverse'))
        kl_reverse_pre = kl_reverse.numpy()
        pen_kl_reverse_pre = pen_kl_reverse.numpy()

        adj = 1
        for _ in range(10):
            if kl > (self.kl_maxfactor * self.delta * delta_mult):
                pass
            elif improve < 0:
                pass
            else:
                break
            
            # Scale policy update
            factor = np.sqrt(2)
            adj = adj / factor
            eta_v_flat = eta_v_flat / factor

            self.actor.set_weights(actor_weights_pik)
            self.actor.set_weights(eta_v_flat,from_flat=True,increment=True)
            
            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            surr = tf.reduce_mean(ratio * adv_all * weights_all)
            improve = surr - surr_before

            kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_ref))
        else:
            # No policy update
            adj = 0
            self.actor.set_weights(actor_weights_pik)

            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            surr = tf.reduce_mean(ratio * adv_all * weights_all)
            improve = surr - surr_before

            kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_ref))
        
        pen_kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_all))
            
        ratio_diff = tf.abs(ratio - offpol_ratio)
        tv = 0.5 * tf.reduce_mean(weights_all * ratio_diff)
        pen = 0.5 * tf.reduce_mean(weights_all * tf.abs(ratio-1.))
        
        kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_ref,direction='reverse'))
        pen_kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_all,direction='reverse'))            
        
        log_actor = {
            'ent':                  ent.numpy(),
            'tv_pre':               tv_pre,
            'kl_pre':               kl_pre,
            'kl_reverse_pre':       kl_reverse_pre,
            'pen_pre':              pen_pre,
            'pen_kl_pre':           pen_kl_pre,
            'pen_kl_reverse_pre':   pen_kl_reverse_pre,
            'tv':                   tv.numpy(),
            'kl':                   kl.numpy(),
            'kl_reverse':           kl_reverse.numpy(),
            'penalty':              pen.numpy(),
            'penalty_kl':           pen_kl.numpy(),
            'penalty_kl_reverse':   pen_kl_reverse.numpy(),
            'adj':                  adj,
            'improve':              improve.numpy(),
            'eps':                  self.eps,
            'delta':                self.delta
        }
        self.logger.log_train(log_actor)

        self.actor.update_pik_weights()