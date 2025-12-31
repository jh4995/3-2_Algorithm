import os
import json
import struct
from collections import defaultdict, Counter
from .tokenizer import extract_terms

class Indexer:
    
    def __init__(self, data_dir, output_dir, doc_table_file, term_dict_file, 
                 postings_file_T, postings_file_A, postings_file_C):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.doc_table_file = os.path.join(self.output_dir, doc_table_file)
        self.term_dict_file = os.path.join(self.output_dir, term_dict_file)
        self.postings_file_T = os.path.join(self.output_dir, postings_file_T)
        self.postings_file_A = os.path.join(self.output_dir, postings_file_A)
        self.postings_file_C = os.path.join(self.output_dir, postings_file_C)

    def build_index(self):
        def read_json_file(file_path):
            with open(file_path, encoding='utf8') as f:
                return json.load(f)

        def extract_fields_from_document(data):
            dataset = data['dataset']
            title = dataset.get('invention_title', '')
            abstract = dataset.get('abstract', '')
            claims = dataset.get('claims', '')
            return title, abstract, claims
        
        def calculate_term_frequencies_by_field(title, abstract, claims):
            tf_title = Counter(extract_terms(title))
            tf_abstract = Counter(extract_terms(abstract))
            tf_claims = Counter(extract_terms(claims))
            return tf_title, tf_abstract, tf_claims
        
        def update_postings_and_doc_table(tf_title, tf_abstract, tf_claims, 
                                         doc_id, title_text, abstract_text, claims_text, 
                                         relative_path,
                                         term_postings_T, term_postings_A, term_postings_C,
                                         doc_table):
            doc_table[doc_id] = {
                "doc_id": doc_id,
                "relpath": relative_path,
                "len_T": sum(tf_title.values()),
                "len_A": sum(tf_abstract.values()),
                "len_C": sum(tf_claims.values()),
                "T_text": title_text,
                "A_text": abstract_text,
                "C_text": claims_text
            }
            
            for term, freq in tf_title.items():
                term_postings_T[term].append((doc_id, freq))
            
            for term, freq in tf_abstract.items():
                term_postings_A[term].append((doc_id, freq))
            
            for term, freq in tf_claims.items():
                term_postings_C[term].append((doc_id, freq))
        
        def calculate_average_lengths(doc_table):
            total_docs = len(doc_table)
            if total_docs == 0:
                return 0, 0, 0
            
            sum_len_T = sum(doc["len_T"] for doc in doc_table.values())
            sum_len_A = sum(doc["len_A"] for doc in doc_table.values())
            sum_len_C = sum(doc["len_C"] for doc in doc_table.values())
            
            avg_len_T = sum_len_T / total_docs
            avg_len_A = sum_len_A / total_docs
            avg_len_C = sum_len_C / total_docs
            
            return avg_len_T, avg_len_A, avg_len_C
        
        def save_doc_table_with_metadata(doc_table, avg_len_T, avg_len_A, avg_len_C):
            output_data = {
                "_meta": {
                    "total_docs": len(doc_table),
                    "avg_len_T": avg_len_T,
                    "avg_len_A": avg_len_A,
                    "avg_len_C": avg_len_C
                }
            }
            
            for doc_id, doc_info in doc_table.items():
                output_data[str(doc_id)] = doc_info
            
            with open(self.doc_table_file, 'w', encoding='utf8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        def save_postings_and_term_dict(term_postings_T, term_postings_A, term_postings_C):
            term_dict = {}
            
            all_terms = set(term_postings_T.keys()) | \
                       set(term_postings_A.keys()) | \
                       set(term_postings_C.keys())
            
            for term in all_terms:
                docs_T = set(doc_id for doc_id, _ in term_postings_T.get(term, []))
                docs_A = set(doc_id for doc_id, _ in term_postings_A.get(term, []))
                docs_C = set(doc_id for doc_id, _ in term_postings_C.get(term, []))
                
                global_df = len(docs_T | docs_A | docs_C)
                term_dict[term] = {"df": global_df}
            
            offset_T = 0
            with open(self.postings_file_T, "wb") as f:
                for term in sorted(all_terms):
                    if term in term_postings_T:
                        plist = term_postings_T[term]
                        start = offset_T
                        for doc_id, freq in plist:
                            f.write(struct.pack("ii", doc_id, freq))
                            offset_T += 8
                        term_dict[term]["T"] = {"start": start, "length": len(plist)}
            
            offset_A = 0
            with open(self.postings_file_A, "wb") as f:
                for term in sorted(all_terms):
                    if term in term_postings_A:
                        plist = term_postings_A[term]
                        start = offset_A
                        for doc_id, freq in plist:
                            f.write(struct.pack("ii", doc_id, freq))
                            offset_A += 8
                        term_dict[term]["A"] = {"start": start, "length": len(plist)}
            
            offset_C = 0
            with open(self.postings_file_C, "wb") as f:
                for term in sorted(all_terms):
                    if term in term_postings_C:
                        plist = term_postings_C[term]
                        start = offset_C
                        for doc_id, freq in plist:
                            f.write(struct.pack("ii", doc_id, freq))
                            offset_C += 8
                        term_dict[term]["C"] = {"start": start, "length": len(plist)}
            
            with open(self.term_dict_file, 'w', encoding='utf8') as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=4)
            
            return term_dict
        
        doc_table = {}
        term_postings_T = defaultdict(list)
        term_postings_A = defaultdict(list)
        term_postings_C = defaultdict(list)
        
        doc_id = 0
        processed_files = 0
        
        print("인덱싱 시작...")
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.data_dir)
                    
                    json_data = read_json_file(file_path)
                    title, abstract, claims = extract_fields_from_document(json_data)
                    tf_title, tf_abstract, tf_claims = calculate_term_frequencies_by_field(
                        title, abstract, claims
                    )
                    
                    update_postings_and_doc_table(
                        tf_title, tf_abstract, tf_claims,
                        doc_id, title, abstract, claims, relative_path,
                        term_postings_T, term_postings_A, term_postings_C,
                        doc_table
                    )
                    
                    doc_id += 1
                    processed_files += 1
                    
                    if processed_files % 1000 == 0:
                        print(f"처리된 파일: {processed_files:,}개")
        
        avg_len_T, avg_len_A, avg_len_C = calculate_average_lengths(doc_table)
        save_doc_table_with_metadata(doc_table, avg_len_T, avg_len_A, avg_len_C)
        term_dict = save_postings_and_term_dict(term_postings_T, term_postings_A, term_postings_C)
        
        print(f"\n인덱싱 완료: 총 {processed_files:,}개 파일 처리")
        print(f"총 {len(term_dict):,}개 unique terms")
        print(f"필드별 평균 길이:")
        print(f"  - Title: {avg_len_T:.2f}")
        print(f"  - Abstract: {avg_len_A:.2f}")
        print(f"  - Claims: {avg_len_C:.2f}")
        print(f"결과 파일:")
        print(f"  - {self.doc_table_file}")
        print(f"  - {self.term_dict_file}")
        print(f"  - {self.postings_file_T}")
        print(f"  - {self.postings_file_A}")
        print(f"  - {self.postings_file_C}")