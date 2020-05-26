import pandas as pd
import numpy as np
import math

train_data = open("train.txt")
test_data = open("train.txt")
count ={}
totaltokens = 0
twocount ={}

def preprocess(filename, count, trainTrue):
    if trainTrue == "true":
        file = open(filename)
        for next in file:
            array = next.split()
            for i in array:
                i = i.lower()
                if i not in count:
                    count[i]=1
                else:
                    count[i]+= 1

        filenew = open("after-"+filename,"w")
        file = open(filename)
        for next in file:
            array = next.split()
            filenew.write("<s> ")
            if "<s>" not in count:
                count["<s>"]=1
            else:
                count["<s>"]+=1

            for i in array:
                i = i.lower()
                if count[i] == 1:
                    filenew.write("<unk> ")
                    if "<unk>" not in count:
                        count["<unk>"] = 1
                    else:
                        count["<unk>"] += 1
                else:
                    filenew.write(i+" ")
            filenew.write("</s>\n")
            if "</s>" not in count:
                count["</s>"] = 1
            else:
                count["</s>"] += 1

        filenew.close()
    elif trainTrue == "no":
        file = open(filename)
        for next in file:
            array = next.split()
            for i in array:
                i = i.lower()
                if i not in count:
                    count[i] = 1
                else:
                    count[i] += 1
    else:
        filenew = open("after-" + filename, "w")
        file = open(filename)
        notintraining =0
        total =0.0
        for next in file:
            array = next.split()
            filenew.write("<s> ")
            for i in array:
                i = i.lower()
                total+=1
                if i not in count:
                    notintraining +=1
                    filenew.write("<unk> ")

                else:
                    filenew.write(i+" ")
            filenew.write("</s>\n")

        filenew.close()
        answer = np.arange(2)
        answer[0]= notintraining
        answer[1]=total
        return answer

def bigramcount(filename, twocount, trainTrue):
    bigramtype =0.0
    bigramtoken =0.0
    nobigramtype =0.0
    nobigramtoken =0.0
    if trainTrue == "true":
        file = open(filename)
        for next in file:
            array = next.split()
            for i in range(len(array)-1):
                bigram = (array[i],array[i+1])
                if bigram not in twocount:
                    twocount[bigram] = 1
                else:
                    twocount[bigram] += 1

    else:
        file = open(filename)
        check = {} #for nobigramtype
        checkseen={} #for bigramtype
        for next in file:
            array = next.split()
            for i in range(len(array)-1):
                bigram = (array[i],array[i+1])
                bigramtoken+=1
                if(bigram not in checkseen):
                    checkseen[bigram]=1
                    bigramtype+=1
                if bigram not in twocount:
                    nobigramtoken+=1
                    if bigram not in check:
                        nobigramtype+=1
                        check[bigram]=1

        answer = np.arange(4)
        answer[0] = bigramtoken
        answer[1] = nobigramtoken
        answer[2] = bigramtype
        answer[3] = nobigramtype
        return answer


preprocess("train.txt",count,"true")
question3 =preprocess("test.txt",count,"false")
words =0
tokens =0;

for i in count:
    words+=1
    totaltokens+=count[i]
    if(i != "<s>" and i!= "</s>"):
        tokens += count[i]


def UnigramProbabilty(x, count, sizeofcount,t):
    p = 1.0
    paramters ="Unigram: "
    paramtervalue ="Unigram: "
    m=0
    for ii in x.split():
        m+=1
        p *= (count[ii] / sizeofcount)
        if t == "false":
            if (paramters == "Unigram: "):
                paramters += ("p(" + str(ii) + ")")
                paramtervalue += ("p(" + str(p) + ")")
            else:
                paramters += ("+p(" + str(ii) + ")")
                paramtervalue += ("+p(" + str(p) + ")")
    if t == "false":
        print(paramters)
        print(paramtervalue)
    return p, m

def logUnigramProbability(x, count, sizeofcount,t):
    p = 0.0
    st = ""
    if t == "true":
        file = open(x)
        for next in file:
            st = st + next + "\n"
        file.close()
        m =0
        for ii in st.split("\n"):  # for multiple sentences
            xp, xq= UnigramProbabilty(ii, count, sizeofcount, t)
            m+=xq
            if xp == 0:
                return print("Log Probability Unigram: undefined","The Perplexity of Unigram Model is: Undefined")
            else:
                p += math.log(xp, 2)
        print("Log Probability Unigram: " + str(p))
        print("The Perplexity of Unigram Model is: (1/" + str(m) + ")" + " * " + str(p) + " = " + str((1 / m) * p))
    else:
        for ii in x.split("\n"): #for multiple sentences
            xp,m = UnigramProbabilty(ii,count,sizeofcount,t)
            if xp == 0:
                return print("Log Probability Unigram: undefined","The Perplexity of Unigram Model is: Undefined")
            else:
                p += math.log(xp, 2)
        print("Log Probability Unigram: " + str(p))
        print("The Perplexity of Unigram Model is: (1/" +str(m)+")"+ " * "+str(p)+" = "+str((1/m)*p))



def bigramProbability(x, count, twocount,t):
    array = x.split()
    p = 1.0
    m=0
    parameters ="Bigram: "
    parametervalue ="Bigram: "
    for i in range(len(array) - 1):
        bigram = (array[i], array[i+1])
        m+=1
        if bigram not in twocount:
            p=0
        else:
            p *= (twocount[bigram] / np.double(count[array[i]]))
        if t == "false":
            if (parameters == "Bigram: "):
                parameters += ("p(" + str(bigram) + ")")
                parametervalue += ("p(" + str(p) + ")")
            else:
                parameters += ("+p(" + str(bigram) + ")")
                parametervalue += ("+p(" + str(p) + ")")
    if t == "false":
        print(parameters)
        print(parametervalue)
    return p,m


def logBigramProbability(x, count, twocount,t):
    p = 0.0
    st = ""
    if t == "true":
        file = open(x)
        for next in file:
            st = st + next + "\n"
        file.close()
        m=0
        for i in st.split("\n"):  # for multiple sentences
            xp, xq = bigramProbability(i, count, twocount, t)
            m+=xq
            if xp == 0:
                return print("Log Probability Bigram: undefined","The Perplexity of Bigram Model is: Undefined")
            else:
                p += math.log(xp, 2)
        print("Log Probability Bigram: " + str(p))
        print("The Perplexity of Bigram Model is: (1/" + str(m) + ")" + " * " + str(p) + " = " + str((1 / m) * p))
    else:
        for i in x.split("\n"): # for multiple sentences
            xp,m = bigramProbability(i, count, twocount,t)
            if xp == 0:
                return print("Log Probability Bigram: undefined","The Perplexity of Bigram Model is: Undefined")
            else:
                p += math.log(xp, 2)
        print("Log Probability Bigram: " + str(p))
        print("The Perplexity of Bigram Model is: (1/" + str(m)+")" + " * " + str(p) + " = " + str((1 / m) * p))


def bigramAddOneProbability(x, count, twocount,words,t):
    array = x.split()
    p = 1.0
    m = 0
    parameters = "BigramAdd1: "
    parametervalue = "BigramAdd1: "
    for i in range(len(array) - 1):
        bigram = (array[i], array[i+1])
        m+=1
        if bigram not in twocount:
            p *= (1.0 / (float(count[array[i]]) + words))
        else:
            p *= (twocount[bigram] + 1.0) / (float(count[array[i]]) + words)
        if(t=="false"):
            if (parameters == "BigramAdd1: "):
                parameters += ("p(" + str(bigram) + ")")
                parametervalue += ("p(" + str(p) + ")")
            else:
                parameters += ("+p(" + str(bigram) + ")")
                parametervalue += ("+p(" + str(p) + ")")
    if t =="false":
        print(parameters)
        print(parametervalue)
    return p,m


def logBigramAddOneProbability(x, count, twocount,words,t):
    p = 0.0
    st = ""
    if t == "true":
        file = open(x)
        for next in file:
            st= st +next +"\n"
        file.close()
        m=0
        for ii in st.split("\n"): # for multiple sentences
            xp,xq = bigramAddOneProbability(ii, count, twocount,words,t)
            m+=xq
            if xp == 0: x
                #return print("Log Probability Bigram Add One is: undefined","The Perplexity of Bigram Add One Model is: Undefined")
            else: p += math.log(xp, 2)
        print("Log Probability of Bigram with Add-One Smoothing:  " + str(p))
        print("The Perplexity of Bigram with Add-One Smoothing Model is: (1/" + str(m)+")" + " * " + str(p) + " = " + str((1 / m) * p))
    else:
        for ii in x.split("\n"): # for multiple sentences
            xp,m = bigramAddOneProbability(ii, count, twocount,words,t)
            if xp == 0: x
                #return print("Log Probability Bigram Add One is: undefined","The Perplexity of Bigram Add One Model is: Undefined")
            else: p += math.log(xp, 2)
        print("Log Probability of Bigram with Add-One Smoothing:  " + str(p))
        print("The Perplexity of Bigram with Add-One Smoothing Model is: (1/" + str(m)+")" + " * " + str(p) + " = " + str((1 / m) * p))


print("Answers to Part 1")
input="am sam"
part1count={}
part1twocount={}
preprocess("part1.txt",part1count,"no")
bigramcount("part1.txt",part1twocount,"true")
p,m = bigramAddOneProbability(input,part1count,part1twocount,25,"false")

print("\nAnswers to Part 2")
print("The number of unique words in the training corpus including the padding symbols and the unknown token is: " + str(words))
print("The number of word tokens not including padding symbols is: "+ str(tokens))
print("The percentage of word tokens in the test corpus that did not occur in training is: "+ str(question3[0]/question3[1]))
print("The percentage of word types in the test corpus that did not occur in training is: "+str(question3[0]/words))

bigramcount("after-train.txt",twocount,"true")
question4 =bigramcount("after-test.txt",twocount,"false")

print("The percentage of bigram types in the test corpus that did not occur in training is: "+ str(question4[3]/question4[2]))
print("The percentage of bigram tokens in the test corpus that did not occur in training is: "+ str(question4[1]/question4[0]))

phrase = "I look forward to hearing your reply ."
phrase = phrase.lower()
phrase= "<s> "+phrase+ " </s>"
logUnigramProbability(phrase, count, totaltokens,"false")
logBigramProbability(phrase, count, twocount,"false")
logBigramAddOneProbability(phrase, count, twocount,words,"false")

print("\n The Whole Test Corpus")

logUnigramProbability("after-test.txt", count, totaltokens,"true")
logBigramProbability("after-test.txt", count, twocount,"true")
logBigramAddOneProbability("after-test.txt", count, twocount,words,"true")

