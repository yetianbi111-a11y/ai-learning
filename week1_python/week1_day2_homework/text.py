text = "python is easy and python is powerful"
words = text.split()

words_map ={}
for word in words:
    if(word in words_map):
        words_map[word]+=1
    else:
        words_map[word]=1
for i in words_map:
    print(f"{i}:{words_map[i]}")
