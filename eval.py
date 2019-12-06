import numpy as np

import scipy.stats as st

FLOAT_MAX = np.finfo(np.float32).max

def rmse_score(model, test):

    predictions = model.predict(test.user_ids, test.item_ids)
    return np.sqrt(((test.ratings - predictions) ** 2).mean())

def mrr_score(model, test, train=None, average_per_context=True):

  if train is not None:
      train = train.tocsr()

  mrrs = []

  item_ids = np.arange(test.num_items).astype(np.int64)

  for context in test.contexts():

      user_id = context.user_ids[0]
      target_item_ids = context.item_ids
      predictions = -model.predict(user_id,
                                    item_ids,
                                    user_features=context.user_features,
                                    context_features=context.context_features,
                                    item_features=context.item_features)
      if train is not None:
          predictions[train[user_id].indices] = FLOAT_MAX

      mrr = (1.0 / st.rankdata(predictions)[target_item_ids])

      if average_per_context:
          mrrs.append(mrr.mean())
      else:
          mrrs.extend(mrr.tolist())

  return np.array(mrrs)