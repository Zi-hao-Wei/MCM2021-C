import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


class MyNLTK():
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.ps=PorterStemmer()
        self.wnl=WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        


    def stopWordsFilter(self,words):
        filtered=[]
        for w in words:
            if len(w)<3:
                continue
            if w not in self.stop_words:
                filtered.append(w)
        return filtered

    def pos_tag(self,wordList):
        return nltk.pos_tag(wordList) #filterWords

    def Lemmatization(self,wordList):
        filteredWords=[]
        for w in wordList:
                wordnet_pos = self._get_wordnet_pos(w[1]) or wordnet.NOUN
                fwords=self.wnl.lemmatize(w[0],pos=wordnet_pos) #Lemmatization
                filteredWords.append(fwords)
        return filteredWords


    def _get_wordnet_pos(self,tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def process(self,sentence):
        originWords=self.tokenizer.tokenize(sentence)
        discardStopWords=self.stopWordsFilter(originWords)
        pos_tagged=self.pos_tag(discardStopWords)
        lem=self.Lemmatization(pos_tagged)
        pos_lem=self.pos_tag(lem)

        for i in range(0,len(pos_lem)):
            wordnet_pos=self._get_wordnet_pos(pos_lem[i][1]) or wordnet.NOUN
            pos_lem[i]=(pos_lem[i][0],wordnet_pos)

        return pos_lem

def main():
    sen="'football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.'"
    myNLTK=MyNLTK()
    processed=myNLTK.process(sen)
    print(processed)
if __name__ == '__main__':
    main()