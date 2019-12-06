import random

import torch

import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from layers import ScaledEmbedding, ZeroEmbedding

class HybridContainer(nn.Module):

    def __init__(self,
                 latent_module,
                 user_module=None,
                 context_module=None,
                 item_module=None):

        super(HybridContainer, self).__init__()

        self.latent = latent_module
        self.user = user_module
        self.context = context_module
        self.item = item_module

    def forward(self, user_ids,
                item_ids,
                user_features=None,
                context_features=None,
                item_features=None):

        user_representation, user_bias = self.latent.user_representation(user_ids)
        item_representation, item_bias = self.latent.item_representation(item_ids)

        if self.user is not None:
            user_representation += self.user(user_features)
        if self.context is not None:
            user_representation += self.context(context_features)
        if self.item is not None:
            item_representation += self.item(item_features)

        dot = (user_representation * item_representation).sum(1)

        return dot + user_bias + item_bias

class FeatureNet(nn.Module):

    def __init__(self, input_dim, output_dim, bias=False, nonlinearity='tanh'):

        super(FeatureNet, self).__init__()

        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = F.sigmoid
        elif nonlinearity == 'linear':
            self.nonlinearity = lambda x: x
        else:
            raise ValueError('Nonlineariy must be one of '
                             '(tanh, relu, sigmoid, linear)')

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_1 = nn.Linear(self.input_dim,
                              self.output_dim,
                              bias=bias)

    def forward(self, features):

        return self.nonlinearity(self.fc_1(features))

class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                               sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def user_representation(self, user_ids):

        user_embedding = self.user_embeddings(user_ids)
        user_embedding = user_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)

        return user_embedding, user_bias

    def item_representation(self, item_ids):

        item_embedding = self.item_embeddings(item_ids)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        item_bias = self.item_biases(item_ids).view(-1, 1)

        return item_embedding, item_bias

    def forward(self, user_representation, user_bias, item_representation, item_bias):

        dot = (user_representation * item_representation).sum(1)

        return dot + user_bias + item_bias

class MixtureNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32,
                 projection_scale=1.0,
                 num_components=4):

        super(MixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components
        self.projection_scale = projection_scale

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.taste_projection = nn.Linear(embedding_dim,
                                          embedding_dim * self.num_components, bias=False)
        self.attention_projection = nn.Linear(embedding_dim,
                                              embedding_dim * self.num_components, bias=False)

        for layer in (self.taste_projection, self.attention_projection):
            torch.nn.init.xavier_normal(layer.weight, self.projection_scale)

    def user_representation(self, user_ids):

        user_embedding = self.user_embeddings(user_ids).squeeze()
        user_bias = self.user_biases(user_ids).squeeze()

        return user_embedding, user_bias

    def item_representation(self, item_ids):

        item_embedding = self.item_embeddings(item_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return item_embedding, item_bias

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()


        user_tastes = (self.taste_projection(user_embedding)
                       .resize(batch_size,
                               self.num_components,
                               embedding_size))
        user_attention = (self.attention_projection(user_embedding)
                          .resize(batch_size,
                                  self.num_components,
                                  embedding_size))
        user_attention = user_attention #  * user_embedding.unsqueeze(1).expand_as(user_tastes)

        attention = (F.softmax((user_attention *
                                item_embedding.unsqueeze(1).expand_as(user_attention))
                               .sum(2)).unsqueeze(2).expand_as(user_attention))
        weighted_preference = (user_tastes * attention).sum(1)

        dot = (weighted_preference * item_embedding).sum(1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return dot + user_bias + item_bias

class MixtureComponent(nn.Module):

    def __init__(self, embedding_dim, num_components):

        super(MixtureComponent, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.fc_1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.taste_projection = nn.Linear(embedding_dim,
                                          embedding_dim * num_components,
                                          bias=False)
        self.attention_projection = nn.Linear(embedding_dim,
                                              embedding_dim * num_components,
                                              bias=False)

    def forward(self, x):

        batch_size, embedding_size = x.size()

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        user_tastes = (self.taste_projection(x)
                       .resize(batch_size,
                               self.num_components,
                               embedding_size))
        user_attention = (self.attention_projection(x)
                          .resize(batch_size,
                                  self.num_components,
                                  embedding_size))

        return user_tastes, user_attention

class NonlinearMixtureNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32,
                 num_components=4):

        super(NonlinearMixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.mixture = MixtureComponent(embedding_dim, num_components)


    def user_representation(self, user_ids):

        user_embedding = self.user_embeddings(user_ids).squeeze()
        user_bias = self.user_biases(user_ids).squeeze()

        return user_embedding, user_bias

    def item_representation(self, item_ids):

        item_embedding = self.item_embeddings(item_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return item_embedding, item_bias

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes, user_attention = self.mixture(user_embedding)
        item_embedding = item_embedding.unsqueeze(1).expand_as(user_attention)

        attention = F.softmax((user_attention * item_embedding).sum(2))

        preference = ((user_tastes * item_embedding)
                      .sum(2))
        weighted_preference = (attention * preference).sum(1).squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return weighted_preference + user_bias + item_bias

class EmbeddingMixtureNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32,
                 num_components=4):

        super(EmbeddingMixtureNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_components = num_components

        self.taste_embeddings = ScaledEmbedding(num_users, embedding_dim * num_components)
        self.attention_embeddings = ScaledEmbedding(num_users, embedding_dim * num_components)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

    def forward(self, user_ids, item_ids):

        item_embedding = self.item_embeddings(item_ids)

        batch_size, embedding_size = item_embedding.size()

        user_tastes = (self.taste_embeddings(user_ids)
                       .resize(batch_size,
                               self.num_components,
                               embedding_size))
        user_attention = (self.attention_embeddings(user_ids)
                          .resize(batch_size,
                                  self.num_components,
                                  embedding_size))

        attention = (F.softmax((user_attention *
                                item_embedding.unsqueeze(1).expand_as(user_attention))
                               .sum(2)).unsqueeze(2).expand_as(user_attention))
        weighted_preference = (user_tastes * attention).sum(1)

        dot = (weighted_preference * item_embedding).sum(1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        return dot + user_bias + item_bias