# Text-Classification-using-LDA-and-Word2Vec

We collected the COVID-19 related donation data from GoFundMe platform and classified campaigns using the supervised text classification. Specifically, we categorized campaigns based on beneficiaries: raised for selves and immediate family members (self) or for the organizations, institutes and healthcare workers (community). Here are the steps of the analysis:  

1. Data preprocessing  

For those columns with missing value, We imputed missing numeric data using column medians and replaced missing categorical data with a constant. We scaled all the numerical variables and converted all the categorical data to binary forms using one-hot encoding. We dropped zero variance columns and highly correlated columns. For the text data, we removed all the punctuations and numbers, tokenized the sentences into words, removed stopwords and stemmed the words. To reduce the modeling bias, we balanced the training data by down sampling the majority class.  

2. Feature Engineering  

To increase the predictive power of machine learning algorithms, we conducted extensive feature engineering and created new features from data. We extracted state information from the 'location' column, extracted components of dates such as day of the week/month label from the existing 'launchdate' column, concatenated campaign 'name' column and 'description' column to be a whole 'text' column. For text mining, we generated text features by domain lexicon, Latent Dirichlet Allocation (LDA), neural word embedding (Word2Vec). For the domain lexicon, we manually identified and concluded some top frequent keywords for different classes. For topic modeling, the LDA training summarized all the documents into 15 different themes. The Word2Vec techniques transformed texts into vectors of 85 dimensions to produce word embeddings by a two-layer neural net.  

3. Model Fitting and Hyperparameter Tuning  

We experimented seven popular classifiers, including Decision Tree, AdaBoost, Bernoulli Naive Bayes, Logistic Regression, Random Forest, XGBoost, and Neural Net. We used grid-search to find the optimal hyperparameters for the models which results in the most 'accurate' predictions.  

4. Evaluation and Error Analysis  

As for the evaluation, we chose the over accuracy, as well as the class-specific precision, recall, F1 measure as metrics. Logistic Regression (Accuracy: 86.3%) and Neural Net (Accuracy: 86.0%) turned out to be the best two classifiers, which were then applied to classify all the COVID-19 related GoFundMe campaign data in 1/1/2020-4/30/2020. During the modeling process, error analysis helped us figure out the reason why some instances were wrongly classified by the machine and gave us clues on how to improve the model performance.
