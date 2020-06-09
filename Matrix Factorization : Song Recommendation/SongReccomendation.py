import turicreate 

#Load music data
song_data = turicrate.SFrame('song_data.sframe')

#Data exploration
song_data['song'].show()
song_data.head()

#Count number of unique users
users = song_data['user_id'].unique()

#Create a Song Recommender
training_data,testing_data = song_data.random_split(0.75,seed=0)

#Simple popularity-based recommender
popularity_model = turicreate.popularity_recommender.create(training_data,user_id='user_id',item_id='song')
#Use the popularity model to make some predictions
popularity_model.recommend(users=[users[0]])
popularity_model.recommend(users=[users[1]])

#Build a song-recommender with personalization
personalized_model = turicreate.item_similarity_recommender.create(training_data,user_id='user_id',item_id='song')
#Applying the personalized recommender 
personalized_model.recommend(users=[users[0]])
personalized_model.recommend(users=[users[1]])

personalized_model.get_similar_items(['With Or Without You - U2'])

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])

#Quantitative comparision between models
model_performance = turicreate.recommender.util.compare_models(testing_data, [popularity_model, personalized_model], user_sample=.05)


#Counting the unique artists
song_data.groupby('artist',operations={'total_count':turicreate.aggregate.SUM('listen_count')}).sort('total_count',ascending=False)

subset_test_users = testing_data['user_id'].unique()[0:10000]
personalized_model.recommend(subset_test_users,k=1)
