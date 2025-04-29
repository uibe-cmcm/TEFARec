import torch
from torch import nn

class FactorizationMachine(nn.Module):
    def __init__(self, pimplicit, pexplicit, k):
        super().__init__()
        self.vexplicit = nn.Parameter(torch.rand(pexplicit, k) / 100)
        self.vimplicit = nn.Parameter(torch.rand(pimplicit, k) / 100)
        self.linearexplicit = nn.Linear(pexplicit, 1, bias=False)
        self.linearsimplicit = nn.Linear(pimplicit, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.7)
        self.dropout_1 = nn.Dropout(0.1)

    def forward(self, ximplicit):
        linear_partimplicit = self.linearsimplicit(ximplicit)
        linear_part = self.dropout_1(linear_partimplicit) + self.bias
        inter_part1implicit = torch.mm(ximplicit, self.vimplicit) ** 2
        inter_part2implicit = torch.mm(ximplicit ** 2, self.vimplicit ** 2)
        pair_interactionsimplicit = torch.sum(inter_part1implicit - inter_part2implicit, dim=1, keepdim=True)
        pair_interactions = pair_interactionsimplicit
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output

class TEFARec(nn.Module):
    def __init__(self, config, num_users, num_items):
        super(TEFARec, self).__init__()
        if config.data_name=='dm':
            aspect_size = 5
        elif config.data_name=='yelp':
            aspect_size = 6
        else:
            raise ValueError('no dataname')
        self.user_embeddings = nn.Embedding(num_users, config.embed_size)
        self.item_embeddings = nn.Embedding(num_items, config.embed_size)
        self.user_embeddings_explicit = nn.Embedding(num_users, 64)
        self.item_embeddings_explicit = nn.Embedding(num_items, 64)
        self.fm = FactorizationMachine(config.embed_size * 2, aspect_size, 4)

        self.att_qual_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * aspect_size, 128), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 16), torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1), torch.nn.ReLU()
        )

    def forward(self, user_id, item_id, X, Y):
        user_embeds = self.user_embeddings(user_id)
        item_embeds = self.item_embeddings(item_id)
        concat_latent = torch.cat((user_embeds, item_embeds), dim=1)
        concat_latent_xy = torch.cat((X.float(), Y.float()), dim=1)
        att_qual = self.att_qual_mlp(concat_latent_xy)
        prediction = self.fm(concat_latent)
        rating = att_qual + prediction
        return rating