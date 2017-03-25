from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)



vectorizer = CountVectorizer()
string1 = 'hi Katie the self driving car will be late Best Sebastian'
string2 = 'Hi Sebastian the machine learning class will be great great great Best Katie'
string3 = 'Hi Katie the machine learning class will be most excellent'
email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit(email_list)
print bag_of_words
bag_of_words = vectorizer.transform(email_list)
print bag_of_words

from nltk.corpus import stopwords
sw = stopwords.words('english')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
stemmer.stem('responsiveness')
stemmer.stem('responsivity')

