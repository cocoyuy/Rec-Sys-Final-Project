import torch
from torch.utils.data import DataLoader
from Datasets import InitDataset, EpinionDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # print(i)
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        #print('grad')
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        #print(loss)
        loss.backward(retain_graph=True) #
        #print('back')
        optimizer.step()
        #print('step')
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            # print('val_o', val_output)
            tmp_pred.append([i for i in list(np.atleast_1d(val_output.data.cpu().numpy()))])
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    print('i-expected_rmse:', expected_rmse)
    return expected_rmse, mae

def main():
    # hyperparameters for training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # load dataset
    ds = InitDataset()
    # ds.describe_location()

    
    train_df, validation_df, test_df = ds.split_train_test() # valid and test portion in 0-1
    history_u_lists, history_ur_lists = ds.get_history_lists(ds.filtered_rating_df,'user','item')
    #with open("history_u_listsuseritem", 'rb') as f:
        #history_u_lists = pickle.load(f)
        #f.close()
    #with open("history_ur_listsuseritem", 'rb') as f:
        #history_ur_lists = pickle.load(f)
        #f.close()
    
    history_v_lists, history_vr_lists = ds.get_history_lists(ds.filtered_rating_df,'item','user')
    #with open("history_u_listsitemuser", 'rb') as f:
        #history_v_lists = pickle.load(f)
        #f.close()
    #with open("history_ur_listsitemuser", 'rb') as f:
        #history_vr_lists = pickle.load(f)
        #f.close()
    social_adj_lists = ds.get_social_adj_lists()

    batch_size = args.batch_size #
    #batch_size = 1000
    train_ds = EpinionDataset(train_df)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_ds = EpinionDataset(validation_df)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_ds = EpinionDataset(test_df)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    # print(next(iter(train_dl)))

    num_users, num_items, num_ratings = ds.describe() # TB debugged - maybe can cause index error
    #print('3 nums:', num_users, num_items, num_ratings)
    num_users = max(ds.user_dict.values())#train_df.user.unique().max()
    num_items = max(max(history_u_lists.values()))+1 # ds.filtered_rating_df.item.unique().max() #
    #print('num_users', num_users)
    #print('num_items', num_items)
    
    embed_dim = args.embed_dim

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    if os.path.exists('best_model.pth'):
        graphrec.load_state_dict(torch.load('best_model.pth'))

    else:
        optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

        best_rmse = 9999.0
        best_mae = 9999.0
        endure_count = 0

        for epoch in tqdm(range(1, args.epochs + 1)):

            train(graphrec, device, train_dl, optimizer, epoch, best_rmse, best_mae)
            expected_rmse, mae = test(graphrec, device, test_dl)
            # please add the validation set to tune the hyper-parameters based on your datasets.

            # early stopping (no validation set in toy dataset)
            if best_rmse > expected_rmse:
                best_rmse = expected_rmse
                best_mae = mae
                endure_count = 0
                torch.save(graphrec.state_dict(), 'best_model.pth')
            else:
                endure_count += 1
            print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

            if endure_count > 5:
                break

    ##group evaluation - location
    test_df_group = ds.create_test_location()
    grouped = test_df_group.groupby('location')
    groups_rmse = {}
    groups_mae = {}

    for l, group_df in grouped: 
        test_ds_group = EpinionDataset(group_df.drop(columns=['location']))
        test_dl_group = DataLoader(test_ds_group, batch_size=5, shuffle=True)
        expected_rmse, mae = test(graphrec, device, test_dl_group)
        groups_rmse[l] = expected_rmse
        groups_mae[l] = mae
        
    print("groups_rmse\n", groups_rmse, "\n")
    print("groups_mae\n", groups_mae, "\n") 


    # plot
    keys = list(groups_rmse.keys())
    rmse_values = list(groups_rmse.values())
    mae_values = list(groups_mae.values())

    # Sort keys and values based on RMSE values
    sorted_indices = sorted(range(len(rmse_values)), key=lambda k: rmse_values[k])
    sorted_keys = [keys[i] for i in sorted_indices]
    sorted_rmse_values = [rmse_values[i] for i in sorted_indices]
    sorted_mae_values = [mae_values[i] for i in sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_keys, sorted_rmse_values, label='RMSE', marker='o')
    plt.plot(sorted_keys, sorted_mae_values, label='MAE', marker='o')
    plt.xlabel('States')
    plt.ylabel('Error')
    plt.title('RMSE and MAE by State')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Get the sample sizes for each state
    df = pd.read_csv('Epinion/processed_data/profileLoc.csv')
    sample_sizes = df.groupby('location').size().to_dict()

    # Sort keys and values based on RMSE values
    sorted_sample_sizes = [sample_sizes.get(key, 0) for key in sorted_keys]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_keys, sorted_rmse_values, s=np.array(sorted_sample_sizes) * 2, c='blue', alpha=0.5)
    plt.scatter(sorted_keys, sorted_mae_values, s=np.array(sorted_sample_sizes) * 2, c='red', alpha=0.5)

    plt.plot(sorted_keys, sorted_rmse_values, color='black', label='RMSE' )
    plt.plot(sorted_keys, sorted_mae_values, color='gray', label='MAE')
    plt.xlabel('States')
    plt.ylabel('Error')
    plt.title('RMSE and MAE by State with Sample Sizes as Radius')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Extract keys and values from the dictionaries
    groups_rmse_social={'Alabama': 0.5373963577683013, 'Alaska': 0.27593869261953724, 'Arizona': 0.23620852289216335, 'Arkansas': 0.13254029062895545, 'California': 0.5692589055027747, 'Colorado': 1.2971548674689404, 'Connecticut': 0.48665693573280916, 'Delaware': 0.1255669625454351, 'District of Columbia': 0.5917687488078925, 'Florida': 0.5675522621810065, 'Georgia': 0.4172036664309281, 'Idaho': 0.2555298758357046, 'Illinois': 0.7258263006282042, 'Indiana': 0.9472199419155124, 'Iowa': 0.8506015990750126, 'Kansas': 0.2596770400822414, 'Kentucky': 0.3708405484165555, 'Louisiana': 0.19833603878306405, 'Maine': 0.1516079311020993, 'Maryland': 0.9461232011465218, 'Massachusetts': 0.2731126899694833, 'Michigan': 0.35949798221181034, 'Minnesota': 0.6710472774734905, 'Mississippi': 1.6938068366279961, 'Missouri': 0.4296854019113816, 'Montana': 0.13763618552498605, 'Nebraska': 0.20623138047472087, 'Nevada': 0.4448484043618812, 'New Hampshire': 0.20672035265681174, 'New Jersey': 0.27903325637829157, 'New Mexico': 0.33848424815825456, 'New York': 0.8140914443246503, 'North Carolina': 1.1563045901221851, 'Ohio': 0.6228422588276756, 'Oklahoma': 0.1317367533827764, 'Oregon': 0.528752599624123, 'Pennsylvania': 0.9398022992132962, 'Rhode Island': 1.3499357543775778, 'South Carolina': 0.7952681403539001, 'South Dakota': 0.26987743260240044, 'Tennessee': 0.7474982338823869, 'Texas': 0.8212816598674262, 'Utah': 0.2944982783432394, 'Vermont': 0.15748644018271538, 'Virginia': 0.8383878997338132, 'Washington': 0.76099729361786, 'Wisconsin': 0.20774243220455857, 'Wyoming': 0.23814062753095064}
    groups_mae_social={'Alabama': 0.40179202, 'Alaska': 0.24642622, 'Arizona': 0.20037194, 'Arkansas': 0.10279608, 'California': 0.3474253, 'Colorado': 1.0298887, 'Connecticut': 0.40520075, 'Delaware': 0.12556696, 'District of Columbia': 0.59176874, 'Florida': 0.37020144, 'Georgia': 0.32618466, 'Idaho': 0.25552988, 'Illinois': 0.41369843, 'Indiana': 0.5456296, 'Iowa': 0.6975808, 'Kansas': 0.2415124, 'Kentucky': 0.27018446, 'Louisiana': 0.17290497, 'Maine': 0.14970668, 'Maryland': 0.73390824, 'Massachusetts': 0.24007325, 'Michigan': 0.27310598, 'Minnesota': 0.41652727, 'Mississippi': 1.2602792, 'Missouri': 0.33298284, 'Montana': 0.13763618, 'Nebraska': 0.20508885, 'Nevada': 0.32839894, 'New Hampshire': 0.20672035, 'New Jersey': 0.21349378, 'New Mexico': 0.3065132, 'New York': 0.49700263, 'North Carolina': 0.74079424, 'Ohio': 0.40631026, 'Oklahoma': 0.12681627, 'Oregon': 0.292626, 'Pennsylvania': 0.63916546, 'Rhode Island': 1.0119146, 'South Carolina': 0.63400084, 'South Dakota': 0.26987743, 'Tennessee': 0.4497685, 'Texas': 0.5131105, 'Utah': 0.26548305, 'Vermont': 0.15748644, 'Virginia': 0.4742157, 'Washington': 0.5342401, 'Wisconsin': 0.16955525, 'Wyoming': 0.23797405}
    keys = list(groups_rmse_social.keys())
    rmse_values = list(groups_rmse.values())
    rmse_values_social = list(groups_rmse_social.values())

    # Sort keys and values based on RMSE values
    sorted_indices = sorted(range(len(rmse_values_social)), key=lambda k: rmse_values_social[k])
    sorted_keys = [keys[i] for i in sorted_indices]
    sorted_rmse_values = [rmse_values[i] for i in sorted_indices]
    sorted_rmse_values_social = [rmse_values_social[i] for i in sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_keys, sorted_rmse_values_social, label='RMSE_Social', marker='o')
    plt.plot(sorted_keys, sorted_rmse_values, label='RMSE', marker='o')
    plt.xlabel('States')
    plt.ylabel('Error')
    plt.title('RMSE with and w/o Social Network')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


    ##group evaluation - title
    test_df_group = ds.create_test_title()
    grouped = test_df_group.groupby('title')
    groups_rmse = {}
    groups_mae = {}

    for l, group_df in grouped: 
        print(l)
        test_ds_group = EpinionDataset(group_df.drop(columns=['title']))
        test_dl_group = DataLoader(test_ds_group, batch_size=5, shuffle=True)
        expected_rmse, mae = test(graphrec, device, test_dl_group)
        groups_rmse[l] = expected_rmse
        groups_mae[l] = mae
        
    print("groups_rmse\n", groups_rmse, "\n")
    print("groups_mae\n", groups_mae, "\n")        
    

if __name__ == '__main__':
    main()