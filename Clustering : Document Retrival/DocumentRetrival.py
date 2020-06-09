import turicreate 

#Load some text data
people = turicreate.SFrame('people_wiki.sframe')
people.show()

#Select some rows and checkout the text it contains
obama = people[people['name']=='Barack Obama']
print(obama)
print(obama['text'])

clooney = people[people['name']=='George Clooney']
print(clooney)
print(clooney['text'])


# Word counts for Obama acticle
obama['word_count'] = turicreate.text_analytics.count_words(obama['text'])
print (obama['word_count'])

#Sorting Word count
obama.stack('word_count',new_column_name=['word','count'])
obama_word_count_table = obama[['word_count']].stack('word_count',new_column_name=['word','count'])
obama_word_count_table
obama_word_count_table.sort('count',ascending=False)

#Compute TF-IDF for the corpus
people['word_count'] = turicreate.text_analytics.count_words(people['text'])
people.head()

people['tfidf'] = turicreate.text_analytics.tf_idf(people['word_count'])

#Examine tf-idf for obama article
obama = people[people['name']=='Barack Obama']
obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

#Manually compute distances between people randomly
clinton = people[people['name']=='Bill Clinton']
beckham = people[people['name']=='David Beckham']

#Similarities between Obama && (Clinton || Beckham)

#turicreate.distances.manhattan(obama['tfidf'][0],clinton['tfidf'][0])
#2041.2192911159102
#turicreate.distances.manhattan(obama['tfidf'][0],beckham['tfidf'][0])
#2036.1554824662835

turicreate.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])		#0.8339854936884277
turicreate.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])		#0.9791305844747478


#Build a nearest-neighbor model
knn_model = turicreate.nearest_neighbors.create(people,features=['tfidf'],label='name')

#Applying th nearest-neighbor model to retrive data
knn_model.query(obama)

#Applying it further
swift = people[people['name']=='Taylor Swift']
knn_model.query(swift)

jolie = people[people['name']=='Angelina Jolie']
knn_model.query(jolie)

arnold = people[people['name'] == 'Arnold Schwarzenegger']
knn_model.query(arnold)






















