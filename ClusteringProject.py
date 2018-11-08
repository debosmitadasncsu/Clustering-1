
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

calendar = pd.read_csv('/Users/Connor/Desktop/HW/Clustering/calendar.csv')
listings = pd.read_csv('/Users/Connor/Desktop/HW/Clustering/listings.csv')
reviews = pd.read_csv('/Users/Connor/Desktop/HW/Clustering/reviews.csv')


# In[ ]:


# listing data management
one = [re.sub('.00','',i) for i in listings['price']]
two = [re.sub('\$','',i) for i in one]
len(two)
x = []
for i in two:
    try:
        x.append(float(i))
    except:
        x.append(np.nan)
        
listings['price'] = pd.Series(x).rename('price')


# In[4]:


# reviews data management
from langdetect import detect
from tqdm import tqdm
from scipy import stats

comments = list(reviews['comments'])
languages = []
for i in tqdm(comments):
    try:
        languages.append(detect(i))
    except:
        languages.append('unknown')


# In[324]:


df = pd.concat([reviews, pd.Series(languages).rename('language')], axis=1)

en_reviews = df[df['language']=='en'].sort_values('comments').reset_index(drop=True)
en_comments = list(en_reviews['comments'])

en_reviews = en_reviews[0:64489]
en_reviews['comments'] = [re.sub('\\n','',i) for i in en_reviews['comments']]
en_reviews['comments'] = [re.sub('\\r','',i) for i in en_reviews['comments']]
en_reviews['comments'] = [re.sub('\\t','',i) for i in en_reviews['comments']]


# In[6]:


import tensorflow_hub as hub

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")   


# In[286]:


import tensorflow as tf
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(en_comments))


# In[287]:


x = pd.DataFrame(en_reviews['listing_id'], ).reset_index(drop=True)
y = pd.DataFrame(message_embeddings)


# In[288]:


new = pd.concat([x,y],axis=1)
house_reviews = new.groupby('listing_id').mean()


# In[289]:


from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


# In[290]:


Z = linkage(house_reviews,'ward')


# In[291]:


def plot_dendrogram(distances):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()
plot_dendrogram(Z)


# In[294]:


from scipy.cluster.hierarchy import fcluster

assignments = fcluster(Z,3.5,'distance')
clusters = pd.concat([pd.Series(house_reviews.index.values).rename('listing_id'),pd.Series(assignments).rename('cluster')],axis=1)


# In[296]:


import statistics as st

def find_cluster_value(cluster = 4, value = 'review_scores_rating'):
    subclust = clusters[clusters['cluster'] == cluster]

    id_to_rating = dict(zip(listings['id'],listings[value]))
    lists = []
    for i in subclust['listing_id']:
        lists.append(id_to_rating[i])
    
    return np.nanmean(np.array(lists))


# In[325]:


for i in range(1,8):
    print('Cluster %d    Cost:' %(i),round(find_cluster_value(cluster=i,value = 'price'),1), 
          '  Rating:',round(find_cluster_value(cluster=i),1),
          '  Location Rating:',round(find_cluster_value(cluster=i,value = 'review_scores_location'),1),
          '  Cleanliness Rating:',round(find_cluster_value(cluster=i,value = 'review_scores_cleanliness'),1),
          '  Reviews per Month:',round(find_cluster_value(cluster=i,value = 'reviews_per_month'),1))

