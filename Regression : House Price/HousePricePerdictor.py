import turicreate
sales = turicreate.SFrame('home_data.sframe')

#displaying imported data in graphical view
sales.show()
#displaying data in tabular view
print(sales[:])

#Plotting sqft_living(x) vs price(y)
turicreate.show(sales[1:5000]['sqft_living'],sales[1:5000]['price'])





##Simple Linear Regression model 
#Spliting data into training data and testing data
training_data,testing_data = sales.random_split(0.75,seed=0)

#Defining our linear regression model
sqft_model = turicreate.linear_regression.create(training_data,target='price',features='sqft_living')

#Evaluate the quality of model
print(testing_data['price'].mean())
print(sqft_model.evaluate(testing_data))
sqft_model.coefficients

#Exploring a little further
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(testing_data['sqft_living'],testing_data['price'],'.',
		 testing_data['sqft_living'],sqft_model.predict(testing_data),'-')
 




#Exploring other features 
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
sales[my_features].show()
turicreate.show(sales['zipcode'],sales['price'])

#Build a model with additional features
my_feature_model = turicreate.linear_regression.create(training_data,target='price',features=my_features)





#Apply learned models to make predictions
#House1
house1 = sales[sales['id']=='5309101200']
print(house1['price'])
print(sqft_model.predict(house1))
print(my_feature_model.predict(house1))

#House2
house2 = sales[sales['id']=='1925069082']
print(house2['price'])
print(sqft_model.predict(house2))
print(my_feature_model.predict(house2))

#House3
bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}
print (my_features_model.predict(turicreate.SFrame(bill_gates)))




##turicreate.show(sales['zipcode'],sales['price'])

