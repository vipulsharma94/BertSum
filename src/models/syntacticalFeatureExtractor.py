import os
import pandas as pd
import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, RegexpParser
from pickle import dump, load
import numpy as np
import nltk
nltk.download('all')


class SyntacticalFeatureExtractorForPT(object):

    def __init__(self,fileType, ptFileName, ptPath, ptSavePath, scalarObjsPath):
        self.ptFileName = ptFileName
        self.ptPath = ptPath
        self.ptSavePath = ptSavePath
        self.fileType = fileType
        self.scalarObjsPath = scalarObjsPath
        self.scalarObjs = []
        for i in range(1,19):
            
            readObj = open(os.path.join(scalarObjsPath,f"scaler_{i}.pkl"), "rb")
            self.scalarObjs.append(load(readObj))
            readObj.close()
            #print("Loaded Scalar Obj for Feature ", i)
            
        self.dataset = torch.load(os.path.join(self.ptPath, ptFileName))
        print("Loaded %s dataset from %s, number of examples: %d' ",(self.fileType, self.ptPath, len(self.dataset)))
        
    
    
    
    def savePtFile(self, finalFeatList):
        
        torch.save(finalFeatList, os.path.join(self.ptSavePath, self.ptFileName))
        
        
    def extractSyntacticalFeatures(self):
        
        
        for i, data in enumerate(self.dataset):
            print("Working on i", i+1)
            synFeats = SyntacticalProcessorForText(data['src_txt']).finalFeatureExtractor()
            scaledSynFeats = [ self.scalarObjs[i].transform(np.array(synFeat).reshape(-1,1)).reshape(1,-1).tolist()[0] for i, synFeat in enumerate(synFeats)]
            scaledSynFeats = [ [round(value, 4) for value in feat ] for feat in scaledSynFeats]
            self.dataset[i]['sync'] = []
            for ind in range(len(data['src_txt'])):
                ithSentFeats = [feat[ind] for feat in scaledSynFeats]
                self.dataset[i]['sync'].append(ithSentFeats)
            self.runValidation(self.dataset[i]['sync'], len(data['src_txt']))
                
    def runValidation(self, syncData, numOfSent):
        #print("data type", type(syncData))
        assert np.array(syncData).shape == (numOfSent,18)
                  
        assert np.any(np.isnan(np.array(syncData))) == False
            

        
class SyntacticalProcessorForText(object):
    
    def __init__(self, sentList):
        
        self.sentList = sentList
        
        self.cleanSentList = self.cleanSentListGen()
        
        self.originalText = self.originalTextMaker(self.sentList)
        
        self.originalCleanText = self.originalTextMaker(self.cleanSentList)
        
        self.countTotalWords = self.countNumberOfWordsInText(self.originalText)
        self.countTotalCleanWords = self.countNumberOfWordsInText(self.originalCleanText)
        
        self.frequencyOfEachWord = self.countFreqOfEachWordInText(self.originalCleanText)
        
    def convertToTensor(self, listValue):
        
        return torch.transpose(torch.tensor([listValue]),0,1)
    
    def sentWordsInTextGen(self, sentList):
        
        return [ [word for word in sent.split()] for sent in sentList]
    
    def removePunctuation(self, sentList):
        #print([ "<s>" + value + "<e>" for value in  sentList])
        return [re.sub(r'[^\w\s]','',sent)  for sent in sentList]
    
    
    def removeStopWords(self, sentList):
        
        stopWords = list(stopwords.words('english'))
        
        return [" ".join(word for word in sent.split() if word not in stopWords) for sent in sentList]
        
    def cleanSentListGen(self):
        
        return self.removePunctuation(self.removeStopWords(self.sentList))

    
    def originalTextMaker(self, sentList):
        
        return " ".join(sentList)
    
    def countFreqOfEachWordInText(self, text):
        
        words = word_tokenize(text)
        fdist = FreqDist(words)
        return dict(fdist)        
        
    
    def countNumberOfWordsInText(self, text):
        #print(text)
        removePunc = re.sub(r'[^\w\s]','',text)
        return len(removePunc.split())
    
    def feat1_SumOfWordsFreqInSent(self, sentList):
        sentWordsList = self.sentWordsInTextGen(sentList)
        freqDict = self.countFreqOfEachWordInText(self.originalTextMaker(sentList))
        return [ sum([freqDict[word]  for word in sentWords if word in freqDict]) for sentWords in sentWordsList ]
    
    def feat2_AvgOfWeightedWordsFreqInSent(self, sentList):
        sentWordsList = self.sentWordsInTextGen(sentList)
        #print("cc",self.countTotalCleanWords)
        #print([(sentWords,len(sentWords)) for sentWords in sentWordsList])
        freqDict = self.countFreqOfEachWordInText(self.originalTextMaker(sentList))
        return [ sum([freqDict[word]/self.countNumberOfWordsInText(self.originalTextMaker(sentList)) for word in sentWords if word in freqDict])/len(sentWords) if len(sentWords) != 0 else 0 for sentWords in sentWordsList ]
    
    def feat3_tfisf(self, sentList):
        
        sentWordsList = self.sentWordsInTextGen(sentList)
    
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentList)
        tfIsf = X.toarray()        
        
        df = pd.DataFrame(tfIsf, columns = vectorizer.get_feature_names())
        
        sumOfTfIsfOfWordsInSent = [sum([df.iloc[i][word] for word in sentWords if word in list(df.columns)]) for i, sentWords in enumerate(sentWordsList)]
        if sumOfTfIsfOfWordsInSent == []:
            return []
        maxValue = max(sumOfTfIsfOfWordsInSent)
        return [value/maxValue for value in sumOfTfIsfOfWordsInSent]
    
    def feat4_posTags(self, sentList):
        
        sentWordsList = self.sentWordsInTextGen(sentList)
        taggedSentList = [nltk.pos_tag(sentWords) for sentWords in sentWordsList]
        
        #NounTags
        NN_Total = [sum([1 for word,tag in taggedSent if 'NN' in tag]) for taggedSent in taggedSentList]

        #Verb Tags
        VB_Related_Total = [sum([1 for word,tag in taggedSent if 'VB' in tag]) for taggedSent in taggedSentList]
        
        #Adjective Tags
        JJ_Related_Total = [sum([1 for word,tag in taggedSent if 'JJ' in tag]) for taggedSent in taggedSentList]
        
        #Preposition Tags
        IN_Related_Total = [sum([1 for word,tag in taggedSent if tag == 'IN']) for taggedSent in taggedSentList]
        
        #Pronoun Tags
        PR_Related_Total = [sum([1 for word,tag in taggedSent if 'PR' in tag]) for taggedSent in taggedSentList]
        
        #Adverb Tags
        RB_Related_Total = [sum([1 for word,tag in taggedSent if 'RB' in tag]) for taggedSent in taggedSentList]
        
        #interjection Tags
        UH_Related_Total = [sum([1 for word,tag in taggedSent if 'UH' in tag]) for taggedSent in taggedSentList]
        
        NN_feat = [value/sum(NN_Total) if sum(NN_Total) != 0 else 0 for value in NN_Total]
        VB_feat = [value/sum(VB_Related_Total) if sum(VB_Related_Total) != 0 else 0 for value in VB_Related_Total]
        JJ_feat = [value/sum(JJ_Related_Total) if sum(JJ_Related_Total) != 0 else 0 for value in JJ_Related_Total]
        IN_feat = [value/sum(IN_Related_Total) if sum(IN_Related_Total) != 0 else 0 for value in IN_Related_Total]
        PR_feat = [value/sum(PR_Related_Total) if sum(PR_Related_Total) != 0 else 0 for value in PR_Related_Total]
        RB_feat = [value/sum(RB_Related_Total) if sum(RB_Related_Total) != 0 else 0 for value in RB_Related_Total]
        UH_feat = [value/sum(UH_Related_Total) if sum(UH_Related_Total) != 0 else 0 for value in UH_Related_Total]
        
        return taggedSentList, NN_feat,VB_feat,JJ_feat,IN_feat,PR_feat,RB_feat,UH_feat
    
    def feat5_SentPositionLabel(self, sentList):
        
        N = len(sentList) 
        return [ -1 if (i+1) <= N*0.2 else 1 if (i+1) >= N*0.8 else 0  for i,sent in enumerate(sentList)]
    
    def feat6_SentPositionWeight(self, sentList):
        
        N = len(sentList) 
        return [ 1/(i+1) if (i+1) <= N*0.3 else 1/(N-(i+1)+1) if (i+1) >= N*0.7 else 0  for i,sent in enumerate(sentList)]
    

    def feat7_SentLengthCharacters(self):
        
        totalCount = len(self.originalCleanText)
    
        return [len(sent)/totalCount for sent in self.cleanSentList if totalCount != 0]
    
    def feat8_SentLengthWords(self):
        
        sentWordsList = self.sentWordsInTextGen(self.cleanSentList)
        
        if sentWordsList != 0:
            maxCount = max([ len(sentWords) for sentWords in sentWordsList]) 
        else:
            return []
        return [ len(value)/maxCount for value in  sentWordsList if maxCount != 0]
    
    def feat9_SentLengthStd(self):
        import statistics, math
        sentWordsList = self.sentWordsInTextGen(self.cleanSentList)
        if len(self.cleanSentList) != 0:
            avgWordsPerSent = self.countTotalCleanWords/len(self.cleanSentList)
            stdWordsPerSent = statistics.stdev([len(sent) for sent in sentWordsList])
        else:
            return []
            
        return [1/(1+math.log(abs(avgWordsPerSent- abs((avgWordsPerSent - len(sentWords)) /stdWordsPerSent)))) for sentWords in sentWordsList]
    
    
    def feat10_PhrasesInSent(self):
        
        outputPos = self.feat4_posTags(self.sentList)
        posTags = outputPos[0]
        chunker = RegexpParser(""" 
            NP: {<DT>?<JJ>*<NN>}    #Noun Phrases 
            P: {<IN>}               #Prepositions 
            V: {<V.*>}              #Verbs
            PP: {<P> <NP>}          #Prepostional Phrases 
            VP: {<V> <NP|PP>*}      #Verb Phrases 
                       """) 
        
        output = [str(chunker.parse(posTag)) for posTag in posTags]
        
        NP_Count = [ len(re.findall(r'\(NP',value)) for value in output]
        NP_Feat = [value/sum(NP_Count) if value != 0 else 0 for value in NP_Count]
        
        PP_Count = [ len(re.findall(r'\(PP',value)) for value in output]
        PP_Feat = [value/sum(PP_Count) if value != 0 else 0 for value in PP_Count]
        
        VP_Count = [ len(re.findall(r'\(VP',value)) for value in output]
        VP_Feat = [value/sum(VP_Count) if value != 0 else 0 for value in VP_Count]
    
        return NP_Feat, PP_Feat, VP_Feat
    
    def finalFeatureExtractor(self):
        
        finalFeat1 = self.feat1_SumOfWordsFreqInSent(self.cleanSentList)
        #print("finalFeat1 Completed")
        finalFeat2 = self.feat2_AvgOfWeightedWordsFreqInSent(self.cleanSentList)
        #print("finalFeat2 Completed")
        finalFeat3 = self.feat3_tfisf(self.cleanSentList)
        #print("finalFeat3 Completed")
        finalFeat_pos_parent = self.feat4_posTags(self.sentList)
        #print("finalFeat_pos_parent Completed")
        finalFeat4 = finalFeat_pos_parent[1]
        #print("finalFeat4 Completed")
        finalFeat5 = finalFeat_pos_parent[2]
        #print("finalFeat5 Completed")
        finalFeat6 = finalFeat_pos_parent[3]
        #print("finalFeat6 Completed")
        finalFeat7 = finalFeat_pos_parent[4]
        #print("finalFeat7 Completed")
        finalFeat8 = finalFeat_pos_parent[5]
        #print("finalFeat8 Completed")
        finalFeat9 = finalFeat_pos_parent[6]
        #print("finalFeat9 Completed")
        finalFeat10= finalFeat_pos_parent[7]        
        #print("finalFeat10 Completed")
        finalFeat11 = self.feat5_SentPositionLabel(self.sentList)
        #print("finalFeat11 Completed")
        finalFeat12 = self.feat6_SentPositionWeight(self.sentList)
        #print("finalFeat12 Completed")
        finalFeat13 = self.feat7_SentLengthCharacters()
        #print("finalFeat13 Completed")
        finalFeat14 = self.feat8_SentLengthWords()
        #print("finalFeat14 Completed")
        finalFeat15 = self.feat9_SentLengthStd()
        #print("finalFeat15 Completed")
        finalFeat_phrase_parent =  self.feat10_PhrasesInSent()
        #print("finalFeat_phrase_parent Completed")
        finalFeat16 = finalFeat_phrase_parent[0]
        #print("finalFeat16 Completed")
        finalFeat17 = finalFeat_phrase_parent[1]
        #print("finalFeat17 Completed")
        finalFeat18 = finalFeat_phrase_parent[2]
        #print("finalFeat18 Completed")
        
        return [finalFeat1,\
                finalFeat2,\
                finalFeat3,\
                finalFeat4,\
                finalFeat5,\
                finalFeat6,\
                finalFeat7,\
                finalFeat8,\
                finalFeat9,\
                finalFeat10,\
                finalFeat11,\
                finalFeat12,\
                finalFeat13,\
                finalFeat14,\
                finalFeat15,\
                finalFeat16,\
                finalFeat17,
                finalFeat18]
        