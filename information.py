import warnings

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from scipy import sparse

class ITPC(BaseEstimator, ClusterMixin):
    """Perform information-bottleneck based clustering.
    When calling ``fit``, an affinity matrix is constructed using either
    kernel function such the Gaussian (aka RBF) kernel of the euclidean
    distanced ``d(X, X)``::
            np.exp(-gamma * d(X,X) ** 2)
    or a k-nearest neighbors connectivity matrix.
    Alternatively, using ``precomputed``, a user-provided affinity
    matrix can be used.
    Parameters
    -----------
    n_clusters : integer, optional
        Number of clusters.
    random_state : int, RandomState instance or None (default)
        A pseudo random number generator used for the initialization.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.
    affinity : string, array-like or callable, default 'rbf'
        If a string, this may be one of 'nearest_neighbors', 'precomputed',
        'rbf' or one of the kernels supported by
        `sklearn.metrics.pairwise_kernels`. List of possible affinities:
        and their parameters: sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS
        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.
    n_neighbors : integer
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : dictionary of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.
    verbose: boolean, default=True
	if True, print number of iteration and number of changed
        labels.
    max_iter: integer, default=50
	maximal number of iterations.


    Attributes
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        Affinity matrix used for clustering. Available only if after calling
        ``fit``.
    labels_ :
        Labels of each point

    Notes
    -----
    If you have an affinity matrix, such as a distance matrix,
    for which 0 means identical elements, and high values means
    very dissimilar elements, it can be transformed in a
    similarity matrix that is well suited for the algorithm by
    applying the Gaussian (RBF, heat) kernel::
        np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
    Where ``delta`` is a free parameter representing the width of the Gaussian
    kernel.
    Another alternative is to take a symmetric version of the k
    nearest neighbors connectivity matrix of the points.
    """

    def __init__(self, n_clusters=8,  random_state=None,
                gamma=1., affinity='rbf', n_neighbors=3,
                 degree=3, coef0=1, kernel_params=None, 
                verbose = True, max_iter=50):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gamma = gamma
        self.affinity = affinity
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = 1
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.max_iter = max_iter


    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies information theory pairwise clustering to this affinity matrix.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)
        y : Ignored
        """
        X = check_array(X, accept_sparse=['csr'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("To use a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            display(self.affinity_matrix_)
            # print(type(self.affinity_matrix_))
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        # Convert affinity matrix to transition matrix of Markov process
        # Divide each element by sum of all
        self.affinity_matrix_ /= self.affinity_matrix_.sum()

	# Divide each element by sum of row
        # if not isinstance(self.affinity_matrix_, scipy.sparse.csr.csr_matrix):
        #     self.affinity_matrix_ =  self.affinity_matrix_ / self.affinity_matrix_.sum(axis=1, keepdims=True)
        # else:
        #    row_sum = np.array(self.affinity_matrix_.sum(axis=1)).flatten()
        #    print(list(1 / row_sum))
        #    c = sparse.diags(1 / row_sum, 0)
        #    display(c.todense())
        #    self.affinity_matrix_ = c @ self.affinity_matrix_
        #print(isinstance(self.affinity_matrix_, scipy.sparse.csr.csr_matrix))
        self.n_samples = X.shape[0]
        
        # Random initialization
        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.randint(0, self.n_clusters, size=self.n_samples)
        _, self.clust_size = np.unique(self.labels_, return_counts=True)
        # print(self.labels_)
        
        # Iterate while number of iterations is less than max_iter
        # or while something changes
        

        iter_n = 0
        changed = True
        # Create matrix of transition probabilitites between clusters
        P = np.zeros((self.n_clusters + 1, self.n_clusters + 1))
        if not isinstance(self.affinity_matrix_, scipy.sparse.csr.csr_matrix):
            for i in range(self.n_samples):
                for n in range(self.n_samples):
                    P[self.labels_[i], self.labels_[n]] += self.affinity_matrix_[i, n]
        else:
            for i, n in zip(*self.affinity_matrix_.nonzero()):
                P[self.labels_[i], self.labels_[n]] += self.affinity_matrix_[i, n]
                    
        # Difference between P and P' in case some sample is moved
        # from its cluster to a different cluster
        delta = np.zeros_like(P)
        
        while iter_n < self.max_iter and changed:
            iter_n += 1
            changed = 0
            if self.verbose:
                print("Iteration ", iter_n)
            
            
            
            for s in range(self.n_samples):
                
                # Save, in which cluster sample was
                # before we try to move it to other clusters
                old_cluster = self.labels_[s]
                # print("Old cluster ", old_cluster)
                self.labels_[s] = self.n_clusters
                self.clust_size[old_cluster] -= 1
                
                P[self.n_clusters, :] *= 0
                P[:, self.n_clusters] *= 0
                delta[self.n_clusters, :] *= 0
                delta[:, self.n_clusters] *= 0
                
                if not isinstance(self.affinity_matrix_, scipy.sparse.csr.csr_matrix):
                    for n in range(self.n_samples):
                    
                        delta[self.n_clusters, self.labels_[n]] += self.affinity_matrix_[s, n]
                        delta[self.labels_[n], self.n_clusters] += self.affinity_matrix_[n, s]
                    
                        if n != s:
                            delta[old_cluster, self.labels_[n]] -= self.affinity_matrix_[s, n]
                            delta[self.labels_[n], old_cluster] -= self.affinity_matrix_[n, s]
                        else:
                            delta[old_cluster, old_cluster] -= self.affinity_matrix_[s, s]
                            delta[self.n_clusters, self.n_clusters] = self.affinity_matrix_[s, s]
                            
                else:
                    for i, j in zip(*self.affinity_matrix_.nonzero()):
                        if i == s:
                            delta[self.n_clusters, self.labels_[j]] += self.affinity_matrix_[s, j]
                            if j != s:
                                delta[old_cluster, self.labels_[j]] -= self.affinity_matrix_[s, j]
                            else:
                                delta[old_cluster, old_cluster] -= self.affinity_matrix_[s, s]
                                
                        if j == s:
                            delta[self.labels_[i], self.n_clusters] += self.affinity_matrix_[i, s]
                    
                            if i != s:
                                delta[self.labels_[i], old_cluster] -= self.affinity_matrix_[i, s]
                            else:
                            
                                delta[self.n_clusters, self.n_clusters] = self.affinity_matrix_[s, s]
                        
                P += delta

                info_score = np.zeros(self.n_clusters)
                
                for c in range(self.n_clusters):

                    if c != self.labels_[s]:
                        P_log_1 = P / P.sum(axis=1, keepdims=True)
                        P_log_1 = P_log_1 * P[[self.n_clusters, c], :].sum()
                        P_log_1 = P_log_1 / P[[self.n_clusters, c], :].sum(axis=0, keepdims=True)
                        
                        P_log_1 = np.log2(P_log_1)
                        np.nan_to_num(P_log_1, copy=False)
                        P_info_1 = P * P_log_1
                        I1 = 2 * np.sum(P_info_1[[self.n_clusters, c], ])
                        
                        P_log_2 = P / P.sum(axis=1, keepdims=True)
                        P_log_2 *= P[[self.n_clusters, c], :].sum()
                        P_log_2 /= P[[self.n_clusters, c], [self.n_clusters, c]].sum(axis=0, keepdims=True)
                        P_log_2 = np.log2(P_log_2)
                        np.nan_to_num(P_log_2, copy=False)
                        P_info_2 = P * P_log_2
                        I2 = np.sum(P_info_2[[self.n_clusters, c], [self.n_clusters, c]])
                        info_score[c] = I1 - I2
                        
                # print("Info_score ", info_score)
                best_c = np.argmin(info_score)
                # print("Best c ", best_c)
                if old_cluster != best_c:
                    changed += 1
                self.labels_[s] = best_c
                
                # Zero matrix of changes and add probabilities to the best cluster
                delta *= 0
                delta[best_c, :] = P[self.n_clusters, :]
                delta[:, best_c] = P[:, self.n_clusters]
                delta[best_c, best_c] = P[best_c, self.n_clusters] + P[self.n_clusters, best_c] + P[self.n_clusters, self.n_clusters]
                self.clust_size[best_c] += 1
            if self.verbose:
                print(changed)
        return self
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
