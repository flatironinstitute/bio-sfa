import numpy as np
from tqdm import trange


def fit(
    x,
    out_dim,
    eta_w,
    eta_m,
    lr_regulator,
    num_epochs,
    history_length=1000,
    pre_centered=False,
    colvar=True,
    rg=None,
    dtype=np.float32,
):
    """Fit an online Bio-SFA model

    To fit a network:

    ```
    Minv, W, avg, history = biosfa.fit(...)
    ```

    To extract the learned slow features from new data:

    ```
    y = Minv @ W @ (x - avg)
    ```

    Parameters
    ----------
    X : np.array
        Data to train the model on. Should have shape (T, n),
        where n is the input dimension and T is the number
        of samples.
    out_dim : int
        Output dimensions of network.
    eta_w : float
        Learning rate for W matrix
    eta_m : float
        Learning rate for M matrix
    lr_regulator : float
        Controls learning rate decay, which will follow
            lr_regulator / (t + lr_regulator)
    num_epochs : int
        Number of runs through the training data
    history_length : int
        This function keeps a history of Minv and W. These
        are recorded
    pre_centered : bool
        If the data is already centered, pass True to skip
        the online centering. Returned `avg` will be a 0-vector.
    colvar : bool
        If your data is T x n_features, keep this True. But if
        rows correspond to variables, i.e. the data is n_features x T,
        set to False.
    rg : np.random.Generator, optional
        A numpy rng for reproducibility.
    dtype : np.dtype
        Data type of learned weights etc.

    Returns
    -------
    Minv, W, avg, history : np.array
        Minv is the inverse of M, the matrix of lateral inhibitory
        weights, and W is the matrix of feedforward weights. avg is
        the online-learned mean vector.
        history stores a record of the learning. it is a tuple
            (t, Minv_history, W_history) = history
        where t is an array of timesteps, and Minv_history and
        W_history are the weights saved at those timesteps. The
        timesteps will be log-spaced for better plotting on log-x
        plots.
    """
    # -- are we centering?
    do_center = not pre_centered

    # -- rng
    if rg is None:
        rg = np.random.default_rng()

    # -- row variables?
    if not colvar:
        x = x.T

    # -- model initialization
    # M = np.eye(out_dim, dtype=dtype)
    Minv = np.eye(out_dim, dtype=dtype)
    W = rg.normal(size=(out_dim, x.shape[1])).astype(dtype)
    W /= np.sqrt(x.shape[1])
    avg = np.zeros(x.shape[1], dtype=dtype)

    # -- record learning history
    T = num_epochs * (x.shape[0] - 1)
    t_history = np.unique(
        np.round(np.geomspace(1, T - 1, num=history_length)).astype(int)
    )
    Minv_history = []
    W_history = []

    # -- some storage arrays to reduce # of allocations
    xt = np.zeros_like(avg)
    old_xt = np.zeros_like(xt)
    zt = np.zeros_like(xt)

    # -- training loop
    t_total = 1
    for e in range(num_epochs):
        print("Epoch", e + 1, "/", num_epochs, flush=True)
        
        for t in trange(1, len(x), miniters=10000):
            # position in data
            xt[:] = x[t]

            # LR schedule
            reg = lr_regulator / (lr_regulator + t)
            eta_w_ = reg * eta_w
            eta_m_ = reg * eta_m

            # online centering of inputs
            if do_center:
                avg += eta_w_ * (xt - avg)
                xt -= avg
            zt[:] = xt
            zt += old_xt
            old_xt[:] = xt

            # time-averaged output for learning
            # Minv = np.linalg.inv(M)
            ybart = Minv @ (W @ zt)

            # update W
            yzT = ybart[:, None] @ zt[None, :]
            WxxT = (W @ xt)[:, None] @ xt[None, :]
            W += eta_w_ * (yzT - WxxT)

            # update Minv
            # this is a Sherman-Morrison-Woodbury update,
            # easily derived from the learning rule for M:
            # M += eta_m_ * (ybart[:, None] @ ybart[None, :] - M)
            Minv /= 1.0 - eta_m_
            y_ = Minv @ ybart
            y_y_T = y_[:, None] @ y_[None, :]
            y_Ty_ = ybart @ y_
            Minv -= (eta_m_ / (1.0 + eta_m_ * y_Ty_)) * y_y_T

            if t_total in t_history:
                Minv_history.append(Minv.copy())
                W_history.append(W.copy())
            t_total += 1

        old_xt[:] = x[0] - avg

    history = (t_history, Minv_history, W_history)
    return Minv, W, avg, history
