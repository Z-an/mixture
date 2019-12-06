"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp

import torch

from torch.autograd import Variable

from utils import gpu, iter_none, make_tuple


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)
    length = min(len(x) for x in tensors if hasattr(x, '__len__'))

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, length, batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, length, batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def grouped_minibatch(groupby_array, *tensors):

    values, group_indices = np.unique(groupby_array, return_index=True)
    group_indices = np.concatenate((group_indices, [len(groupby_array)]))

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(len(values)):
            slc = slice(group_indices[i], group_indices[i + 1])
            yield tensor[slc]
    else:
        for i in range(len(values)):
            slc = slice(group_indices[i], group_indices[i + 1])
            yield tuple(x[slc] for x in tensors)

def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor

def _slice_or_none(arg, slc):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(x[slc] for x in arg)
    else:
        return arg[slc]

def _tensor_or_none(arg, use_cuda):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(gpu(torch.from_numpy(x), use_cuda) for x in arg)
    else:
        return gpu(torch.from_numpy(arg), use_cuda)

def _variable_or_none(arg):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return tuple(Variable(x) for x in arg)
    else:
        return Variable(arg)

def _dim_or_zero(arg, axis=1):

    if arg is None:
        return 0
    elif isinstance(arg, tuple):
        return sum(x.shape[axis] for x in arg)
    else:
        return arg.shape[axis]

def _float_or_none(arg):

    if arg is None:
        return None
    elif isinstance(arg, tuple):
        return (x.astype(np.float32) for x in arg)
    else:
        return arg.astype(np.float32)

class Interactions(object):

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 user_features=None,
                 item_features=None,
                 context_features=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.ratings = _float_or_none(ratings)
        self.timestamps = timestamps
        self.weights = _float_or_none(weights)
        self.user_features = _float_or_none(user_features)
        self.item_features = _float_or_none(item_features)
        self.context_features = _float_or_none(context_features)

        self._check()

    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
                .format(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __len__(self):

        return len(self.user_ids)

    def _check(self):

        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater '
                             'than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater '
                             'than declared number of items.')

        if self.ratings is not None and self.ratings.size != len(self):
            raise ValueError('Number of ratings incosistent '
                             'with number of interactions.')

        for feature in make_tuple(self.user_features):
            if feature.shape[0] != self.num_users:
                raise ValueError('Number of user features not '
                                 'equal to number of users.')

        for feature in make_tuple(self.item_features):
            if feature.shape[0] != self.num_items:
                raise ValueError('Number of item features not '
                                 'equal to number of items.')

        for feature in make_tuple(self.context_features):
            if feature.shape[0] != len(self):
                raise ValueError('Number of context features not '
                                 'equal to number of interactions.')

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids),
                            ('ratings', self.ratings),
                            ('timestamps', self.timestamps),
                            ('weights', self.weights)):

            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError('Invalid {} dimensions: length '
                                 'must be equal to number of interactions'
                                 .format(name))

    def _sort(self, indices):

        self.user_ids = self.user_ids[indices]
        self.item_ids = self.item_ids[indices]
        self.ratings = _slice_or_none(self.ratings, indices)
        self.timestamps = _slice_or_none(self.timestamps, indices)
        self.weights = _slice_or_none(self.weights, indices)
        self.context_features = _slice_or_none(self.context_features, indices)

    def shuffle(self, random_state=None):

        if random_state is None:
            random_state = np.random.RandomState()

        shuffle_indices = np.arange(len(self.user_ids))
        random_state.shuffle(shuffle_indices)

        self._sort(shuffle_indices)

    def minibatches(self, use_cuda=False, batch_size=128):

        if use_cuda:
            fnc = lambda x: _tensor_or_none(x, use_cuda)
        else:
            fnc = lambda x: x

        batch_generator = zip(*(minibatch(*fnc(make_tuple(attr)),
                                          batch_size=batch_size)
                                if attr is not None
                                else iter_none()
                                for attr in (self.user_ids,
                                             self.item_ids,
                                             self.ratings,
                                             self.timestamps,
                                             self.weights,
                                             self.context_features)))

        user_features = fnc(self.user_features)
        item_features = fnc(self.item_features)

        for (uids_batch, iids_batch, ratings_batch, timestamps_batch,
             weights_batch, cf_batch) in batch_generator:

            yield InteractionsMinibatch(
                user_ids=uids_batch,
                item_ids=iids_batch,
                ratings=ratings_batch,
                timestamps=timestamps_batch,
                weights=weights_batch,
                user_features=_slice_or_none(user_features, uids_batch),
                item_features=_slice_or_none(item_features, iids_batch),
                context_features=_slice_or_none(context_features, cf_batch),
            )

    def contexts(self):

        if self.num_context_features():
            for batch in self.minibatches(batch_size=1):
                yield batch
        else:
            # Sort by user id
            sort_indices = np.argsort(self.user_ids)
            self._sort(sort_indices)

            batch_generator = zip(*(grouped_minibatch(
                self.user_ids,
                *make_tuple(attr))
                if attr is not None
                else iter_none()
                for attr in (self.user_ids,
                             self.item_ids,
                             self.ratings,
                             self.timestamps,
                             self.weights,
                             self.context_features)))

            user_features = self.user_features
            item_features = self.item_features

            for (uids_batch, iids_batch, ratings_batch, timestamps_batch,
                 weights_batch, cf_batch) in batch_generator:

                yield InteractionsMinibatch(
                    user_ids=uids_batch,
                    item_ids=iids_batch,
                    ratings=ratings_batch,
                    timestamps=timestamps_batch,
                    weights=weights_batch,
                    user_features=_slice_or_none(user_features, uids_batch),
                    item_features=item_features,
                    context_features=cf_batch,
                )

    def num_user_features(self):

        x = 0
        try:
            x = self.user_features.shape[0]
            return x
        except:
            return x

    def num_context_features(self):

        x = 0
        try:
            x = self.context_features.shape[0]
            return x
        except:
            return x

    def num_item_features(self):

        x = 0
        try:
            x = self.item_features.shape[0]
            return x
        except:
            return x

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

class InteractionsMinibatch(object):

    def __init__(self,
                 user_ids,
                 item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 user_features=None,
                 item_features=None,
                 context_features=None):

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self.user_features = user_features
        self.item_features = item_features
        self.context_features = context_features

    def get_item_features(self, item_ids):

        if isinstance(item_ids, Variable):
            item_ids = item_ids.data

        return _slice_or_none(self.item_features, item_ids)

    def torch(self, use_cuda=False):

        fnc = lambda x: _tensor_or_none(x, use_cuda) if not torch.is_tensor(x) else x

        return InteractionsMinibatch(
            user_ids=fnc(self.user_ids),
            item_ids=fnc(self.item_ids),
            ratings=fnc(self.ratings),
            timestamps=fnc(self.timestamps),
            weights=fnc(self.weights),
            user_features=fnc(self.user_features),
            item_features=fnc(self.item_features),
            context_features=fnc(self.context_features)
        )

    def variable(self):

        fnc = lambda x: _variable_or_none(x)

        return InteractionsMinibatch(
            user_ids=fnc(self.user_ids),
            item_ids=fnc(self.item_ids),
            ratings=fnc(self.ratings),
            timestamps=fnc(self.timestamps),
            weights=fnc(self.weights),
            user_features=fnc(self.user_features),
            item_features=fnc(self.item_features),
            context_features=fnc(self.context_features)
        )

    def __len__(self):

        return len(self.user_ids)
