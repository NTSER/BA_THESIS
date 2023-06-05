import pandas as pd
from datetime import datetime

#Read separately
formulanewsge = pd.read_json('formulanewsge/formulanewsge.json')
formulanewsge['source'] = 'formulanews.ge'

imedinewsge = pd.read_json('imedinewsge/imedinewsge.json')
imedinewsge['source'] = 'imedinews.ge'

mtavaritv = pd.read_json('mtavaritv/mtavaritv.json')
mtavaritv['source'] = 'mtavari.tv'

tvpirvelige = pd.read_csv('tvpirvelige/tvpirvelige.csv')
tvpirvelige['source'] = 'tvpirveli.ge'

tv1ge = pd.read_json('tv1ge/tv1ge.json')
tv1ge['source'] = '1tv.ge'

postvmedia = pd.read_json('postvmedia/postvmedia.json')
postvmedia['source'] = 'postv.media'


#concatenate and clean
df = pd.concat([formulanewsge, imedinewsge, mtavaritv, tvpirvelige, tv1ge, postvmedia])
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= datetime(2020, 1, 1)) & (df['date'] < datetime(2023, 5, 1))]
df = df.sort_values(by='date')
df = df.dropna()
content_len = df['content'].apply(len)
df = df[(content_len > 100) & (content_len < 10000)]

print(df.groupby('source').agg({'date':'first', 'content':'count'}).sort_values(by='date'))

#save
df.to_csv('../political_news.csv', index=False)