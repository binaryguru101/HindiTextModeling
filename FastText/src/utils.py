import unicodedata
from collections import Counter
from itertools import chain

class WordPieceTokenizer:
    def __init__(self):
        self.vocmap = {}
        self.vocmap_inv = {}
        self.special_token = ["<UNK>", "<s>", "</s>"]
        
    def preprocess(self,text):
        norm = unicodedata.normalize("NFKC",text)
        res = ""
        for w in norm:
            category = unicodedata.category(w)
            if category.startswith("P"):
                res += " "+ w+ " "
            elif category in ["Cf", "Cc"]:
                continue
            else:
                res += w
        return res 
        
    def createcorpus(self,text):
        text = text.split()
        corpus = []
        for w in text:
            word_token = [w[0]]+[f"##{c}" for c in w[1:]]
            corpus.append(word_token)
        return corpus
        
    # def getcharcounts(self,corpus):
    #     counts = {}
    #     for w in corpus:
    #         for ch in w:
    #             counts[ch] = counts.get(ch,0)+1
    #     return counts 
    def getcharcounts(self,corpus):
        return Counter(chain.from_iterable(corpus))

    
    # def getpairwisecounts(self,corpus):
    #     pair_counts = {}
    #     for lis in corpus:
    #         for i in range(0,len(lis)-1):
    #             pair = (lis[i],lis[i+1])
    #             pair_counts[pair] = pair_counts.get(pair,0)+1
    #     return pair_counts

    def getpairwisecounts(self,corpus):
        pair_counts = Counter()
        for word in corpus:
            if len(word) < 2:
                continue
            pair_counts.update(zip(word, word[1:]))
        return pair_counts

    def getpairscores(self,pair_counts,char_counts):
        scores = {}
        for (a,b),ct in pair_counts.items():
            scores[(a,b)] = ct /(char_counts[a]*char_counts[b])
        return scores
    
    def gethighestliklihood(self,scores):
        mx = max(scores.values())
        max_like = [k for k,v in scores.items() if v==mx]
        max_like.sort()
        return max_like[0] 
        
    def merge(self,corpus,pair):
        a,b = pair
        new_corpus = []
        for word in corpus:
            if a not in word:
                new_corpus.append(word)
                continue
            new_word = []
            i = 0 
            while i < len(word):
                if i < len(word)-1 and (word[i],word[i+1])==pair:
                    if b.startswith("##"):
                        cln_s = b[2:]
                    else:
                        cln_s = b
                    merged = a+cln_s
                    new_word.append(merged)
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_corpus.append(new_word) 
        return new_corpus    

    def train_wordpiece(self,text,VOCAB_SIZE):
        text = self.preprocess(text)
        corpus = self.createcorpus(text)
    
        vocab = set()
        for word in corpus:
            for token in word:
                vocab.add(token)
        print("LEN OF VOCAB",len(vocab))
    
        while len(vocab) < VOCAB_SIZE:
            char_counts = self.getcharcounts(corpus)
            pair_counts = self.getpairwisecounts(corpus)
    
            if not pair_counts:
                break
            scores = self.getpairscores(pair_counts,char_counts)
            best_pair = self.gethighestliklihood(scores)
            fr,sec = best_pair
            if sec.startswith("##"):
                cl_sec= sec[2:]
            else:
                cl_sec = sec
            nwtk = fr+cl_sec
            vocab.add(nwtk)
            corpus = self.merge(corpus,best_pair)
            print(f"Merged: {best_pair} -> {nwtk} | Vocab Size: {len(vocab)}")
        sorted_voc = self.special_token + sorted(list(vocab))
        self.vocmap = {token: idx for idx, token in enumerate(sorted_voc)}
        self.vocmap_inv = {idx: token for idx, token in enumerate(sorted_voc)}
        return sorted_voc  

    def encode(self,sen):
        unk = self.vocmap["<UNK>"]
        bos = self.vocmap["<s>"]
        eos = self.vocmap["</s>"]
        token_id = [bos]
        words = sen.split()
        for w in words:
            start = 0 
            while start < len(w):
                end = len(w)
                found = False
                while end > start:
                    subword = w[start:end]
                    if start>0:
                        subword = "##"+subword
                    if subword in self.vocmap:
                        token_id.append(self.vocmap[subword])
                        start = end
                        found = True
                        break
                    end -=1
                if found == False:
                    token_id.append(unk)
                    start+=1
        token_id.append(eos)
        return token_id

    def decode(self,ids):
        decoded_tokens = []
        for i in ids:
            token = self.vocmap_inv.get(i, "<UNK>")
            if token in ["<s>", "</s>"]:
                continue
            elif token == "<UNK>":
                decoded_tokens.append("[UNK]")
            elif token.startswith("##"):
                if decoded_tokens:
                    decoded_tokens[-1] += token[2:]
                else:
                    decoded_tokens.append(token[2:])
            else:
                decoded_tokens.append(token)
        return " ".join(decoded_tokens)

    def preproc_file(self,inp_path,out_path=None):
        if out_path is None:
            out_path = "preprocessed.txt"
        with open(inp_path,"r", encoding = "utf-8") as inp, open(out_path,"w",encoding="utf-8") as out:
            for line in inp:
                clean = self.preprocess(line)
                out.write(clean+"\n")
        print(f"Success! Preprocessed data saved to: {out_path}")      

    def vocabtraining(self,inp_path,VOCABSIZE,out_path=None):
        if out_path is None:
            out_path = "vocab.txt"
        with open(inp_path,"r",encoding = "utf-8") as f:
            raw_text = f.read()
            
        sorted_vocab = self.train_wordpiece(raw_text,VOCABSIZE)

        with open(out_path,"w",encoding = "utf-8") as out:
            for token in sorted_vocab:
                out.write(token+"\n")
        print(f"Success! Vocab of size {len(sorted_vocab)} saved to: {out_path}")        


    def loadvocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
        self.vocmap = {token: idx for idx, token in enumerate(tokens)}
        self.vocmap_inv = {idx: token for idx, token in enumerate(tokens)}
        
    def tokenizetostrings(self,inp_path,vocab_path,out_path=None):
        if out_path is None: 
            out_path = "stringtokenized.txt"
        self.loadvocab(vocab_path)

        with open(inp_path,"r",encoding = "utf-8") as inp,open(out_path,"w",encoding = "utf-8") as out:
            for line in inp:
                clean = self.preprocess(line)
                ids = self.encode(clean)
                tokens = [self.vocmap_inv[i] for i in ids]
                out.write(" ".join(tokens) + "\n")
        print(f"Success! Tokens saved to: {out_path}") 


    def tokenize(self,inp_path,vocab_path,out_path=None):
        if out_path is None:
            out_path = "tokenized.txt"
        self.loadvocab(vocab_path)

        with open(inp_path,"r",encoding = "utf-8") as inp, open(out_path,"w",encoding = "utf-8") as out:
            for line in inp:
                clean = self.preprocess(line)
                ids = self.encode(clean)
                id_str = " ".join(map(str, ids))
                out.write(id_str + "\n")
        print(f"Success! Token IDs saved to: {out_path}")    

    def detokenize(self,inp_path,vocab_path,out_path=None):
        if out_path is None:
            out_path = "detokenized.txt"
        self.loadvocab(vocab_path)
        with open(inp_path,"r",encoding = "utf-8") as inp, open(out_path,"w",encoding="utf-8") as out:
            for line in inp:
                try:
                    ids = list(map(int, line.strip().split()))
                    decoded_text = self.decode(ids)
                    out.write(decoded_text + "\n")
                except ValueError:
                    out.write("\n")
        print(f"Success! Detokenized text saved to: {out_path}")  