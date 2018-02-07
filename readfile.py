import jieba
import re
import collections
import os



filename="wctest.txt"
vocabulary_size = 100

def getfiletxt():
    if not os.path.exists(filename):
        print("can not find "+filename)
        return
    data=list()
    with open(filename,encoding='UTF-8') as f:
        temp=f.read()
        words = re.sub("[\s+’!“”\"#$%&\'(（）)*,，\-.。·/:：;；《》、<=>?@[\\]【】^_`{|}…~]+","", temp)
    words_split=jieba.cut(words)
    for i,word  in enumerate(words_split):
        data.append(word)
    return data


def build_dataset(words):
    count = [['Unknow', 0]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary=dict()
    for word,num in count:
        dictionary[word]=len(dictionary)
    data=list()
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            count[0][1] += 1
        data.append(index)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data,count,dictionary,reversed_dictionary


if __name__ == "__main__":
    words=getfiletxt()
    data, count, dictionary, reversed_dictionary = build_dataset(words)
    del words
    with open("G:/data.txt",'w') as f1:
        f1.write(str(data))
        f1.close()
        print("data list has been written")
    with open("G:/count.txt",'w') as f2:
        f2.write(str(count))
        f2.close()
        print("count list has been written")
    with open("G:/dictionary.txt",'w') as f3:
        f3.write(str(dictionary))
        f3.close()
        print("dictionary has been written")
    with open("G:/reversed_dictionary.txt",'w') as f4:
        f4.write(str(reversed_dictionary))
        f4.close()
        print("reversed_dictionary has been written")