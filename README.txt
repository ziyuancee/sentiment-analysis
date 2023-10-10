A sentiment analysis engine that rates how positive or negative a movie review is. Classified with Naive Bayes, with Laplace smoothing if needed. Preprocessing methods include: lowercasing, selective punctuation removal, NLTK stemming, and NLTK stop list filtering.

# Installation
Assuming a Windows installation, do
pip install pandas
pip install numpy
pip install matplotlib
pip install nltk
If there's an error after those installs, do:

python -m nltk.downloader stopwords
python -m nltk.downloader punkt
