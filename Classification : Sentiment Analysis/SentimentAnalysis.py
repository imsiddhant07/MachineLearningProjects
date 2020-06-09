import turicreate 

#Importing the dataset
products = turicreate.SFrame('amazon_baby.sframe')

#Exploring the dataset
print(products)

#Grouping the reviews by name of product
products.groupby('name',operations={'count':turicreate.aggregate.COUNT()}).sort('count',ascending=False)

#Examine the reviews for most reviewed product Vulli Sophie the Giraffe Teether..
giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']
print(len(giraffe_reviews))

#Ratings for Giraffe
giraffe_reviews['rating'].show()




#Build a sentiment classifier
#Build word count vector for each review
products['word_count'] = turicreate.text_analytics.count_words(products['review'])
print(products)

#Defining what is positive and negative sentiment
products['rating'].show()

#We'll ignore 3 star ratings
products = products[products['rating']!=3]

#positive sentiment = 4-star or 5-star reviews
products['sentiment'] = products['rating']>=4
print(products)

#Lets train the sentiment classifier
training_data,testing_data = products.random_split(0.75,seed=0)
sentiment_model = turicreate.logistic_classifier.create(training_data,target='sentiment',features=['word_count'],validation_set=testing_data)

# Apply the sentiment classifier to better understand the Giraffe reviews
products['predicted_sentiment'] = sentiment_model.predict(products,output_type='probability')
giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']

# Sort the Giraffe reviews according to predicted sentiment
giraffe_reviews = giraffe_reviews.sort('predicted-sentiment',ascending=False)
print(giraffe_reviews)

#Reading positive reviews
giraffe_reviews[0]['review']
giraffe_reviews[1]['review']

#Reading negative comments
giraffe_reviews[-1]['review']
giraffe_reviews[-2]['review']



##Quiz
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

for word in selected_words:
	products[word] = products['word_count'].apply(lambda counts: counts.get(word,0))

new_sentiments = turicreate.logistic_classifier.create(training_data,target='sentiment',features=['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate'],validation_set=testing_data)


