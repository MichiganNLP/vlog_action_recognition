#! /bin/bash

# Download the Stanford NLP tools if needed
if [ ! -d "stanford-postagger-full-2018-10-16" ]; then
    wget http://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip
    # Extract the zip file.
    unzip stanford-postagger-full-2018-10-16.zip
fi

# install python packages
pip install -r requirements.txt

#setup nltk
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"


