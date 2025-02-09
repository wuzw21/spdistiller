with open('/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/tools/files.txt','r') as f :
    lines = f.readlines()
for line in lines :
    if '|mmlu' in line  or '|perplexity' in line or '|word_perplexity' in line :
        print(line)