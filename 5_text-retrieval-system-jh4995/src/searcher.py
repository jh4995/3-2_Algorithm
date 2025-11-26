import os
import json
import struct
import math
from .tokenizer import extract_terms

class Searcher:
    
    def __init__(self, index_dir, doc_table_file, term_dict_file, postings_file):
        self.index_dir = os.path.abspath(index_dir)
        
        term_dict_path = os.path.join(self.index_dir, term_dict_file)
        with open(term_dict_path, 'r', encoding='utf8') as f:
            self.term_dict = json.load(f)
        
        doc_table_path = os.path.join(self.index_dir, doc_table_file)
        with open(doc_table_path, 'r', encoding='utf8') as f:
            self.doc_table = json.load(f)
        
        self.N = len(self.doc_table)
        
        postings_path = os.path.join(self.index_dir, postings_file)
        self.fp = open(postings_path, "rb")
    
    def get_postings(self, term):
        if term not in self.term_dict:
            return []
        
        entry = self.term_dict[term]
        df = entry["df"]
        start_offset = entry["start"]
        
        postings = []
        self.fp.seek(start_offset)
        for _ in range(df):
            data = self.fp.read(8)
            if len(data) != 8:
                raise ValueError(f"Incomplete data read at offset {start_offset}")
            doc_id, freq = struct.unpack("ii", data)
            postings.append((doc_id, freq))
        
        return postings
    
    def process_query(self, user_query):
        # query term 전처리
        query_terms = extract_terms(user_query)
        
        # search main
        doc_scores = {}
        
        for term in query_terms:
            if term not in self.term_dict:
                continue
            
            df = self.term_dict[term]["df"]
            idf = math.log((self.N + 1) / (df + 1)) + 1
            
            postings = self.get_postings(term)
            for doc_id, tf in postings:
                tf_idf = tf * idf
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + tf_idf
        
        # document ranking
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 결과 출력
        print("\nRESULT:")
        print(f"검색어: [{user_query}]")
        print(f"총 {len(ranked_docs)}개 문서 검색")
        print("상위 5개 문서:")
        
        for doc_id, score in ranked_docs[:5]:
            doc_info = self.doc_table[str(doc_id)]
            filename = os.path.basename(doc_info["filename"])
            print(f"  {filename}  {score:.2f}")