import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os

class InitDataset():
    def __init__(self, rating_file='Epinion/processed_data/full_ratings2.csv', profile_file='Epinion/processed_data/profileLoc.csv', user_dict_file = 'Epinion/processed_data/user_dict.txt'):
        self.rating_df = self._get_rating(rating_file)
        self.user_df, self.profile_user_df = self._get_user(profile_file)
        self.user_dict = self._get_user_dict(user_dict_file)
        self.filtered_rating_df = self._get_filtered_rating_df()
        # _,self.location_user_df = self.get_location_group()
        print(int(self.filtered_rating_df['rating'].max()))

    def describe_location(self):
        x = self.profile_user_df.groupby("location")["username"].count().index.tolist()
        print("x",x)
        y_rating_cnts = self.profile_user_df.groupby("location")["username"].count().tolist()
        print("y",y_rating_cnts)
        location_dic = {}
        for i in range (len(x)):
            location_dic[x[i]] = y_rating_cnts[i]
        location_dic = {k: v for k, v in sorted(location_dic.items(), key=lambda item: item[1])}
        plt.pie(location_dic.values(), labels = location_dic.keys(), autopct='%1.1f%%',textprops={'size': 'smaller'})
        # plt.legend(title = "Locations:")
        plt.show()

    def describe(self):
        return int(self.filtered_rating_df['user'].max())+1, int(self.filtered_rating_df['item'].max())+1, int(self.filtered_rating_df['rating'].max())+1
    
    def _get_rating(self, rating_file):
        rating_df = pd.read_csv(rating_file, header=0)
        rating_df['user'] = rating_df['user'].astype(int)
        rating_df['item'] = rating_df['item'].astype(int)
        return rating_df
    
    def _get_user(self, profile_file):
        user_df = pd.read_csv(profile_file) # header=0, names=['username','title','location','realname','register_time', 'homepage', 'selfdescriptions','num_reviews_written','num_member_visits','num_visits','fav_websites']
        # print(user_df.head()) # debug
        profile_user_df = user_df[user_df['location'].notna()] # self.profile_user_df: user df with non-empty location info # len=40742
        return user_df, profile_user_df
    
    def _get_user_dict(self, user_dict_file):
        with open(user_dict_file, 'r') as file:
            file_contents = file.read()
            user_dict = eval(file_contents) # self.user_dict: key is userID and value is username
        return user_dict
    
    def _get_filtered_rating_df(self): 
        print('Filtering rating df ...')
        user_ratings_count = self.rating_df.groupby('user')['rating'].count()
        users_with_valid_rating = set(user_ratings_count[user_ratings_count >= 5].index) # users with at least 5 ratings
        users_with_valid_profile = set() # get a set with profile userID
        for username in set(self.profile_user_df['username']):
            if username in self.user_dict: # in case of expectation
                users_with_valid_profile.add(self.user_dict[username])
        
        filtered_rating_df = self.rating_df[self.rating_df['user'].isin(users_with_valid_rating & users_with_valid_profile)]
        # print('filtered rating df len =',filtered_rating_df['user'].nunique()) # 22732 
        # filtered_rating_df = filtered_rating_df.head(15000) 
        return filtered_rating_df

        ##filter group evaluation user with >=5 ratings
    def get_filtered_user_df(self, any_user_df):
        any_user_df['username'] = any_user_df['username'].replace(self.user_dict)
        valid_user = self.filtered_rating_df['user'].unique()
        filtered_user_df = any_user_df[any_user_df['username'].isin((valid_user))]
        filtered_user_df = filtered_user_df.rename(columns = {'username':'user'})
        return filtered_user_df 
    
    ## user location matching, only>5 & notna
    def get_location_group(self): 
        if os.path.exists('filtered_u_df.csv'):
            filtered_u_df = pd.read_csv('filtered_u_df.csv')
        else:
            profile_user_df_copy = self.profile_user_df.copy()     
            profile_user_df_copy = profile_user_df_copy[['username','location']] 
            profile_user_df_copy = profile_user_df_copy.rename({'username':'user'})
            filtered_u_df = self.get_filtered_user_df(profile_user_df_copy)
            filtered_u_df.to_csv('filtered_u_df.csv', index=False)  # Set index=False to avoid saving row indices
        return filtered_u_df
    
    ## user title matching, only>5 & notna
    def get_title_group(self): 
        if os.path.exists('filtered_u_df2.csv'):
            filtered_u_df = pd.read_csv('filtered_u_df2.csv')
        else:
            profile_user_df_copy = self.profile_user_df.copy()     
            profile_user_df_copy = profile_user_df_copy[['username','title']] 
            profile_user_df_copy = profile_user_df_copy.rename({'username':'user'})
            filtered_u_df = self.get_filtered_user_df(profile_user_df_copy)
            filtered_u_df.to_csv('filtered_u_df2.csv', index=False)  # Set index=False to avoid saving row indices
        return filtered_u_df
            
    ## new test set with different user, LOO
    def create_test_location(self):
        print('Getting test data with location ...')
        train_set, validation_set, test_set = [], [], []
        location_df = self.get_title_group()
        print('location', location_df)
        rating_df = self.filtered_rating_df
        print('rating', rating_df)
        merged_df = pd.merge(location_df, rating_df, on='user', how='inner')
        
        print("location group prev",merged_df)
        grouped = merged_df.groupby('user')
        # print("location group",g)

        # Function to split data for each user
        def split_user_data(user_df):
            test_data = user_df.iloc[-1:]
            return test_data

        # Iterate over groups
        print("grouped", len(grouped))
        for _, group_df in tqdm(grouped):
            test_data = split_user_data(group_df)
            # train_set.append(train_data)
            # validation_set.append(validation_data)
            test_set.append(test_data)
        # train_df = pd.concat(train_set, ignore_index=True)
        # validation_df = pd.concat(validation_set, ignore_index=True)
        test_df = pd.concat(test_set, ignore_index=True)
        
        return test_df
    

    def create_test_title(self):
        print('Getting test data with title ...')
        test_set = []
        title_df = self.get_title_group()
        rating_df = self.filtered_rating_df
        merged_df = pd.merge(title_df, rating_df, on='user', how='inner')
        print('tt', merged_df['title'])
        merged_df['title'] = merged_df['title'].isna().astype(int)  # Replace NaN values with 0 and non-NaN values with 1 in the 'title' column
        print('tt aft', merged_df['title'])
        grouped = merged_df.groupby('user')
        for _, group_df in tqdm(grouped):
            test_data = group_df.iloc[-1:]
            test_set.append(test_data)
        test_df = pd.concat(test_set, ignore_index=True)
        
        return test_df
    
    def split_train_test(self):
        print('Spliting train, validation, test data ...')
        train_set, validation_set, test_set = [], [], []
        grouped = self.filtered_rating_df.groupby('user')

        for _, group_df in tqdm(grouped):
            test_set.append(group_df.iloc[-1])
            validation_set.append(group_df.iloc[-2])
            train_set.append(group_df.iloc[:-2])
        # print("sets",train_set, validation_set, test_set)
        train_df = pd.concat(train_set, ignore_index=True)
        validation_df = pd.DataFrame(validation_set).reset_index(drop=True)
        test_df = pd.DataFrame(test_set).reset_index(drop=True)

        return train_df, validation_df, test_df

    ## cold start
    # def split_train_test(self, valid_portion, test_portion):
    #     print('Spliting train, validation, test data ...')
    #     train_set, validation_set, test_set = [], [], []
    #     grouped = self.filtered_rating_df.groupby('user')

    #     # Function to split data for each user
    #     def split_user_data(user_df):
    #         total_rows = len(user_df)
    #         test_size = int(total_rows * test_portion)  
    #         validation_size = int(total_rows * valid_portion)  
    #         train_size = total_rows - test_size - validation_size
    #         train_data = user_df.iloc[:train_size]
    #         validation_data = user_df.iloc[train_size:train_size + validation_size]
    #         test_data = user_df.iloc[train_size + validation_size:]
    #         return train_data, validation_data, test_data

    #     # Iterate over groups
    #     for _, group_df in grouped:
    #         train_data, validation_data, test_data = split_user_data(group_df)
    #         train_set.append(train_data)
    #         validation_set.append(validation_data)
    #         test_set.append(test_data)
    #     train_df = pd.concat(train_set, ignore_index=True)
    #     validation_df = pd.concat(validation_set, ignore_index=True)
    #     test_df = pd.concat(test_set, ignore_index=True)
        
    #     return train_df, validation_df, test_df
    
    def get_history_lists(self, train_df, index1, index2): # 
        history_u_lists, history_ur_lists = {}, {}
        print("Getting history lists ...")
        for user in tqdm(set(train_df[index1])):
            history_u_lists[user] = list((train_df[train_df[index1] == user][index2])) # set
            history_ur_lists[user] = list(train_df[train_df[index1] == user]['rating'])
        return history_u_lists, history_ur_lists

    def get_social_adj_lists(self, social_adj_file='Epinion/processed_data/trust_rel.txt'):
        social_adj_lists = defaultdict(set)
        with open(social_adj_file, 'r') as file:
            file_contents = file.read()
        parsed_file_content = eval(file_contents) 
        for key, value in parsed_file_content.items():
            # social_adj_lists[int(key)].add(tuple(value))
            for item in value:
                social_adj_lists[int(key)].add(item)
        return social_adj_lists


class EpinionDataset(Dataset):
    def __init__(self, df, user_col=0, item_col=1, rating_col=2):
        self.df = df
        self.user_tensor = torch.tensor(self.df.iloc[:,user_col].values.astype(int), dtype=torch.long)
        self.item_tensor = torch.tensor(self.df.iloc[:,item_col].values.astype(int), dtype=torch.long)
        self.target_tensor = torch.tensor(self.df.iloc[:,rating_col].values.astype(int), dtype=torch.float32)

    def __getitem__(self, index):
        return(self.user_tensor[index], self.item_tensor[index], self.target_tensor[index])

    def __len__(self):
        return(self.target_tensor.shape[0])