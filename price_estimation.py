import numpy as np
import pandas as pd
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.model_selection import (train_test_split, GridSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures)
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('data/home_price.csv')

data.date = pd.to_datetime(data.date)
data['data_year'] = data['date'].dt.year

# Set renovation year to built year where renovation didn't take place
data['yr_renovated'][data['yr_renovated']==0] = data['yr_built'][data['yr_renovated']==0]

data['yrs_since_built'] = data['data_year']-data['yr_built']
data['yrs_since_renovated'] = data['data_year']-data['yr_renovated']


seattle = np.array([[47.6131746,-122.4821478]],dtype="float64")
loc = np.array(data[['lat', 'long']])
from scipy.spatial import distance

dist = []
for i in np.arange(len(loc)):
	dist.append(round(distance.euclidean(loc[i], seattle), 2))
data['dist'] = dist

# droping these columns to get our optimal set of variables for exploratory analysis.
data = data.drop(['id','date','lat','long','yr_built','yr_renovated','data_year'],axis=1)

y=data['price']
x = data.drop('price', axis=1)

#create train and test set before adding zip code average price
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


def zip_code_avg(X, y):
	temp = pd.concat([X,y],axis=1)
	grouped = temp.groupby('zipcode').agg({'price':'mean'})
	grouped = grouped.reset_index(level = 'zipcode')
	temp = pd.merge(X,grouped,how='left',on='zipcode')
	temp.rename(columns={'price':'zipcodeavgprice'},inplace=True)
	map = temp.groupby(['zipcode','zipcodeavgprice']).size().reset_index().rename(columns={0:'count'})
	del temp['zipcode']
	del map['count']
	return temp,map

#Calling the function to add the zipcodeavg X_train    
X_train,map = zip_code_avg(X_train,y_train)

#using the train data to map avg zipcode price to X_test
X_test  = pd.merge(X_test,map,on='zipcode',how='left')
X_test = X_test.drop('zipcode',axis=1)

# correlation_mat = data.corr()
# sns.heatmap(correlation_mat, vmax=.8, square=True);

# Based on the corelation plot, we see strong correlation between the price (target variable)
# with the variables sqft_living, grade, sqft_above, view and bathrooms.
 # Lets check the relation between them through a details correlation plot.

# corr_mat = X_train.corr()
# sns.set()
# cols = ['price', 'sqft_living', 'grade', 'sqft_above', 'view', 'bathrooms']
# sns.pairplot(data[cols], size = 2.5)
# plt.show()



#multiple linear regression 
scaler = MinMaxScaler()

scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


linreg = LinearRegression().fit(X_train_scaled, y_train)

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test_scaled, y_test)))

u = pd.DataFrame(X_train.columns.copy())
v = pd.DataFrame(linreg.coef_.copy())
pd.merge(u,v,left_index=True,right_index=True)

y_pred = linreg.predict(X_test_scaled)
#print("Mean squared error: %.2f"
#% mean_squared_error(y_test, y_pred))        
#print('Variance score: %.2f' % r2_score(y_test, y_pred))
RSS = ((y_test - y_pred )*(y_test - y_pred)).sum()
# square the residuals and add them up
print ("RSS\n",RSS)

# plot the polynomial regression relation between the price and variables which we found correlated in our exploratory analysis.

def polyreg():
    poly_df = pd.DataFrame(np.arange(2,5),columns=['degree'])
    train_sc = []
    test_sc = []
    for i in np.arange(2,5):
        poly = PolynomialFeatures(degree=i)
        X_F1_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)
        reg = clf.fit(X_F1_poly, y_train)
        train_sc.append(reg.score(X_F1_poly,y_train))
        test_sc.append(reg.score(X_test_poly,y_test))
    poly_df['training_score'] = train_sc
    poly_df['testing_score'] = test_sc
    print(pd.DataFrame(poly_df))
    plt.close()
    plt.xlabel('Degrees')
    plt.ylabel('R-squared')
    plt.title('Variation in train/test accuracies with degrees')
    plt.grid(True)
    plt.plot(poly_df['degree'],poly_df['training_score'])
    plt.plot(poly_df['degree'],poly_df['testing_score'])
    plt.legend(['train','test'], loc='upper left')
    plt.savefig('polyreg.png')
    return None

# Use a polynomial regression as a mean to examine this dataset.
poly_df = pd.DataFrame(np.arange(2,5),columns=['degree'])

clf  = LinearRegression()
polyreg()


# Setup the pipeline steps: Scaling using MinMaxScaler and running the model
steps = [('scaler', MinMaxScaler()),
         ('Ridge', Ridge(tol=0.1))]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'Ridge__alpha': np.logspace(-4, 0, 50)}

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline,param_grid=parameters,cv=5)

# Fit to the training set
gm_cv.fit(X_train,y_train)
    
# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned Ridge Alpha: {}".format(gm_cv.best_params_))
print("Tuned Ridge R squared: {0:.4f}".format(r2))


# Custom function to plot the hyperparameters against the cross validated scores
def display_plot(cv_scores, cv_scores_std):
	fig = plit.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(alpha_space, cv_scores)

	std_error = cv_scores / np.sqrt(10)

	ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
	ax.set_ylabel('CV Score +/- Std Error')
	ax.set_xlabel('Alpha')
	ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
	ax.set_xlim([alpha_space[0], alpha_space[-1]])
	ax.set_xscale('log')
	plt.savefig('display_plot.png')
	# plt.show()

# Plotting the hyperparameters against the cross validated scores
scaler = MinMaxScaler()
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True,tol=0.1)

for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge,X_test_scaled,y_test,cv=10)
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

#display the plot
display_plot(ridge_scores, ridge_scores_std)


# Polynomial Ridge Regression
ridge = Ridge(alpha = gm_cv.best_params_['Ridge__alpha'],normalize=True,tol=0.1)

#clf indicates the model on which to run polynomial regression
clf = ridge

polyreg()


# LASSO Regression

# Setup the pipeline steps: steps
steps = [('scaler', MinMaxScaler()),
         ('Lasso', Lasso(tol=0.1))]


# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'Lasso__alpha': np.logspace(-4, 3, 50)}

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline,param_grid=parameters,cv=5)

# Fit to the training set
gm_cv.fit(X_train,y_train)


# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned Lasso L1 penalty: {}".format(gm_cv.best_params_))
print("Tuned Lasso R squared: {0:.4f}".format(r2))



# Plotting the hyperparameters against the cross validated scores


