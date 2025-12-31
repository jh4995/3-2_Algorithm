import os
import json
import struct
from collections import defaultdict, Counter
from .tokenizer import extract_terms

class Indexer:
    
    def __init__(self, data_dir, output_dir, doc_table_file, term_dict_file, postings_file):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.doc_table_file = os.path.join(self.output_dir, doc_table_file)
        self.term_dict_file = os.path.join(self.output_dir, term_dict_file)
        self.postings_file = os.path.join(self.output_dir, postings_file)

    def build_index(self):
        """인덱스 구축 메인 함수"""
        
        def read_json_file(file_path):
            """JSON 파일을 읽어서 데이터를 반환"""
            with open(file_path, encoding='utf8') as f:
                return json.load(f)

        def extract_fields_from_document(data):
            """invention_title, abstract, claims를 각각 분리하여 추출"""
            dataset = data['dataset']
            title = dataset.get('invention_title', '')
            abstract = dataset.get('abstract', '')
            claims = dataset.get('claims', '')
            return title, abstract, claims
        
        def calculate_field_term_frequencies(title, abstract, claims):
            """각 필드별 단어 추출 및 빈도 계산"""
            title_terms = extract_terms(title)
            abstract_terms = extract_terms(abstract)
            claims_terms = extract_terms(claims)
            
            title_freq = Counter(title_terms)
            abstract_freq = Counter(abstract_terms)
            claims_freq = Counter(claims_terms)
            
            return title_freq, abstract_freq, claims_freq, len(title_terms), len(abstract_terms), len(claims_terms)
        
        def update_postings_and_doc_table(title_freq, abstract_freq, claims_freq, 
                                          len_t, len_a, len_c, title_text,
                                          doc_id, term_postings_t, term_postings_a, term_postings_c,
                                          doc_table, file, file_path):
            """포스팅 리스트 업데이트 및 문서 테이블에 정보 저장"""
            # 문서 테이블에 doc_id를 키로 저장
            doc_table[doc_id] = {
                "doc_id": doc_id,
                "filename": file,
                "path": file_path,
                "len_T": len_t,
                "len_A": len_a,
                "len_C": len_c,
                "T_text": title_text
            }
            
            # 필드별 postings 리스트에 추가
            for term, freq in title_freq.items():
                term_postings_t[term].append((doc_id, freq))
            for term, freq in abstract_freq.items():
                term_postings_a[term].append((doc_id, freq))
            for term, freq in claims_freq.items():
                term_postings_c[term].append((doc_id, freq))
        
        def save_doc_table(doc_table, total_len_t, total_len_a, total_len_c, num_docs):
            """doc_table.json 파일 저장 (평균 길이 정보 포함)"""
            metadata = {
                "avgdl_T": total_len_t / num_docs if num_docs > 0 else 0,
                "avgdl_A": total_len_a / num_docs if num_docs > 0 else 0,
                "avgdl_C": total_len_c / num_docs if num_docs > 0 else 0
            }
            output = {
                "metadata": metadata,
                "documents": doc_table
            }
            with open(self.doc_table_file, 'w', encoding='utf8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
        
        def save_postings_and_term_dict(term_postings_t, term_postings_a, term_postings_c):
            """postings.bin과 term_dict.json 파일 생성 및 저장"""
            # 모든 term 수집
            all_terms = set(term_postings_t.keys()) | set(term_postings_a.keys()) | set(term_postings_c.keys())
            
            term_dict = {}
            offset = 0
            
            with open(self.postings_file, "wb") as pbin:
                for term in all_terms:
                    term_entry = {"df": 0}
                    
                    # 해당 term이 등장하는 모든 문서 ID 수집 (global df 계산용)
                    doc_ids = set()
                    
                    # Title 필드 포스팅
                    if term in term_postings_t:
                        plist = term_postings_t[term]
                        start = offset
                        for doc_id, freq in plist:
                            pbin.write(struct.pack("ii", doc_id, freq))
                            doc_ids.add(doc_id)
                            offset += 8
                        term_entry["T"] = {"start": start, "length": len(plist)}
                    
                    # Abstract 필드 포스팅
                    if term in term_postings_a:
                        plist = term_postings_a[term]
                        start = offset
                        for doc_id, freq in plist:
                            pbin.write(struct.pack("ii", doc_id, freq))
                            doc_ids.add(doc_id)
                            offset += 8
                        term_entry["A"] = {"start": start, "length": len(plist)}
                    
                    # Claims 필드 포스팅
                    if term in term_postings_c:
                        plist = term_postings_c[term]
                        start = offset
                        for doc_id, freq in plist:
                            pbin.write(struct.pack("ii", doc_id, freq))
                            doc_ids.add(doc_id)
                            offset += 8
                        term_entry["C"] = {"start": start, "length": len(plist)}
                    
                    # global df는 해당 term이 등장하는 문서 수
                    term_entry["df"] = len(doc_ids)
                    term_dict[term] = term_entry
            
            # term_dict.json 저장
            with open(self.term_dict_file, 'w', encoding='utf8') as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=4)
            
            return term_dict
        
        # 메인 로직 시작
        doc_table = {}
        term_postings_t = defaultdict(list)  # Title 포스팅
        term_postings_a = defaultdict(list)  # Abstract 포스팅
        term_postings_c = defaultdict(list)  # Claims 포스팅
        seen_filenames = set()  # 중복 파일명 체크용
        
        doc_id = 0
        processed_files = 0
        total_len_t = 0
        total_len_a = 0
        total_len_c = 0
        
        # JSON 파일들 처리
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    # 중복 파일명 건너뛰기
                    if file in seen_filenames:
                        continue
                    seen_filenames.add(file)
                    
                    file_path = os.path.join(root, file)
                    
                    # JSON 파일 읽기
                    json_data = read_json_file(file_path)
                    
                    # 필드별 텍스트 추출
                    title, abstract, claims = extract_fields_from_document(json_data)
                    
                    # 필드별 단어 빈도 계산
                    title_freq, abstract_freq, claims_freq, len_t, len_a, len_c = calculate_field_term_frequencies(title, abstract, claims)
                    
                    # 평균 길이 계산용 누적
                    total_len_t += len_t
                    total_len_a += len_a
                    total_len_c += len_c
                    
                    # 포스팅 및 문서 테이블 업데이트
                    update_postings_and_doc_table(
                        title_freq, abstract_freq, claims_freq,
                        len_t, len_a, len_c, title,
                        doc_id, term_postings_t, term_postings_a, term_postings_c,
                        doc_table, file, file_path
                    )
                    
                    doc_id += 1
                    processed_files += 1
                    
                    # 테스트용: 1000개 파일만 처리
                    # if processed_files >= 1000:
                    #     break
                    
                    if processed_files % 1000 == 0:
                        print(f"처리된 파일: {processed_files:,}개")
            
            # 테스트용: 1000개 파일만 처리 (외부 루프 break)
            # if processed_files >= 1000:
            #     break
        
        # 결과 파일들 저장
        save_doc_table(doc_table, total_len_t, total_len_a, total_len_c, processed_files)
        term_dict = save_postings_and_term_dict(term_postings_t, term_postings_a, term_postings_c)
        
        # 완료 메시지 출력
        print(f"인덱싱 완료: 총 {processed_files:,}개 파일 처리")
        print(f"총 {len(term_dict):,}개 unique terms")
        print(f"결과 파일:")
        print(f"  - {self.doc_table_file}")
        print(f"  - {self.term_dict_file}")
        print(f"  - {self.postings_file}")