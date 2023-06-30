from itertools import combinations
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, FFMpegWriter



class Polarity:
    def __init__(self, df, vecs, period_length):
        vecs = (vecs - vecs.mean(axis=0))/vecs.std(axis=0)
        df['vecs'] = vecs.tolist()
        df['vecs'] = df['vecs'].apply(lambda x: np.array(x))
        df['date'] = pd.to_datetime(df['date'])

        self.df = df
        self.period_length = period_length
        self.gov_media = ['imedinews.ge', '1tv.ge', 'postv.media']
        self.opp_media = ['tvpirveli.ge', 'formulanews.ge', 'mtavari.tv']
        self.combs = list(combinations(self.gov_media+self.opp_media, 2))
        self.within_combs = list(combinations(self.gov_media, 2)) + list(combinations(self.opp_media, 2))
        self.between_combs = list(set(self.combs) - set(self.within_combs))
        self.source_to_logo = {
            'imedinews.ge':mpimg.imread('logos/imedinews.png'),
            'postv.media':mpimg.imread('logos/postv.png'),
            '1tv.ge':mpimg.imread('logos/1tv.jpg'),
            'formulanews.ge':mpimg.imread('logos/formulanews.png'),
            'mtavari.tv':mpimg.imread('logos/mtavari.png'),
            'tvpirveli.ge':mpimg.imread('logos/tvpirveli.png')
            }
        

    def get_representative_vectors(self):
        repr_vecs = (self.df
                     .groupby([pd.Grouper(freq=self.period_length, key='date'), 'source'])
                     .agg({'vecs':'mean'})
                     .reset_index()
                     .rename({'vecs':'repr_vecs'}, axis=1))
        return repr_vecs
    
    def get_media_weights(self, ratings_path='../ratings.csv'):

        def explode_row(row):
            new_df = pd.DataFrame(columns=['date', 'comb', 'weight_i_j'])
            for n_days in range(7): # 7 days in week
                for media_i, media_j in self.combs:

                    new_row = pd.DataFrame([[None]*3], columns=new_df.columns)
                    new_row['date'] = row['week'] + timedelta(days=n_days)
                    new_row['comb'] = [[media_i, media_j]]
                    new_row['weight_i_j'] = row[media_i]*row[media_j]     

                    new_df = pd.concat([new_df, new_row])
            return new_df
        
        ratings = pd.read_csv(ratings_path, parse_dates=['week'])
        final_df = pd.DataFrame()
        for index, row in ratings.iterrows():
            new_df = explode_row(row)
            final_df = pd.concat([final_df, new_df])
            final_df['comb'] = final_df['comb'].apply(tuple)
        return final_df

    def get_D(self):
        def cosine_dissimilarity(vec1, vec2):
            cosine_dis = 1-np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            return cosine_dis/2
        
        D_df = pd.DataFrame(columns=['date', 'comb', 'D'])
        repr_vecs = self.get_representative_vectors()

        for date in repr_vecs['date'].unique():
            for comb in self.combs:
                temp = repr_vecs[(repr_vecs['date'] == date) & (repr_vecs['source'].isin(comb))]
                if temp.shape[0] == 2: # If we have statistics for both media sources
                    D = cosine_dissimilarity(temp['repr_vecs'].iloc[0], temp['repr_vecs'].iloc[1])
                    new_row = pd.DataFrame({'date':date, 'comb':[comb], 'D':D})
                    D_df = pd.concat([D_df, new_row])
        D_df = D_df.reset_index(drop=True)

        return D_df
    
    def get_L(self, D_df=None):
        if D_df is None:
            D_df = self.get_D()

        L_df = (D_df[D_df['comb'].isin(self.within_combs)]
                .groupby('date')
                .agg({'D':'mean'})
                .reset_index()
                .rename({'D':'L'}, axis=1))

        return L_df

    
    def get_B(self, D_df=None, L_df=None):
        if D_df is None:
            D_df = self.get_D()
        if L_df is None:
            L_df = (D_df[D_df['comb'].isin(self.within_combs)]
                    .groupby('date').agg({'D':'mean'})
                    .reset_index()
                    .rename({'D':'L'}, axis=1))

        B_df = D_df.merge(L_df, how='left', on='date')
        B_df['B'] = B_df['D'] - B_df['L']
        B_df.loc[B_df['comb'].isin(self.within_combs), 'B'] = 0
        B_df = B_df.reset_index(drop=True)

        return B_df


    def get_polarity(self, B_df=None):
        if B_df is None:
            B_df = self.get_B()

        weights_df = self.get_media_weights()

        B_df = B_df.merge(weights_df, how='left', on=['date', 'comb'])
        B_df['weight_i_j'] = B_df.groupby('date', group_keys=False)['weight_i_j'].apply(lambda x: x/x.sum())
        B_df['polarity'] = B_df['B']*B_df['weight_i_j']

        return B_df 

    def reduce_daily_dimension(self):
        pca = PCA(n_components=2, random_state=7)
        repr_vecs = (self.df
                     .groupby([pd.Grouper(freq='1d', key='date'), 'source'])
                     .agg({'vecs':'mean'})
                     .reset_index())
        repr_vecs[['x','y']] = pca.fit_transform(np.vstack(repr_vecs['vecs'].values))
        repr_vecs[['x_ewma', 'y_ewma']] = (repr_vecs
                                                 .groupby('source', group_keys=False)[['x', 'y']]
                                                 .apply(lambda x :x.ewm(alpha=0.05).mean()))
        return repr_vecs
    
    def save_animation(self, file_name='animation.mp4', im_size=0.4):
        repr_vecs = self.reduce_daily_dimension()
        repr_vecs = repr_vecs.dropna()
        unique_dates = repr_vecs['date'].unique()
        num_iterations = unique_dates.shape[0]
        fig, ax = plt.subplots()
        
        def annimate(i):
            if i >= num_iterations:
                return
            
            date = unique_dates[i]
            filtered = repr_vecs[repr_vecs['date'] == date]
            
            ax.cla()
            ax.scatter(filtered['x_ewma'], filtered['y_ewma'], alpha=0)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim([-3, 3])
            ax.set_ylim([-2, 2])
            ax.set_title(date.astype(str)[0:10], fontsize=20, fontweight='bold')
            for j in range(filtered.shape[0]):
                current_img = self.source_to_logo.get(filtered['source'].iloc[j])
                img_y, img_x = current_img.shape[0:2]
                img_y, img_x = img_y/(img_y+img_x), img_x/(img_y+img_x)
                ax.imshow(current_img,
                            extent=[filtered['x_ewma'].iloc[j] - im_size * img_x,
                                    filtered['x_ewma'].iloc[j] + im_size * img_x,
                                    filtered['y_ewma'].iloc[j] - im_size * img_y,
                                    filtered['y_ewma'].iloc[j] + im_size * img_y])
            for x,y in zip(filtered['x_ewma'], filtered['y_ewma']):
                ax.arrow(0, 0, x*0.7, y*0.7, width=0.03, color='black')
    
            ax.scatter(0, 0, color='red', marker='o', edgecolor='black', s=140)
 
        animation = FuncAnimation(fig, annimate, interval=200, frames=num_iterations)
        writer = FFMpegWriter(fps=15)
        animation.save(file_name, writer=writer)