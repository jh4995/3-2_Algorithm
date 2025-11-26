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

        def extract_text_from_document(data):
            """invention_title, abstract, claims를 추출하여 concatenate"""
            dataset = data['dataset']
            text_parts = []
            
            if 'invention_title' in dataset:
                text_parts.append(dataset['invention_title'])
            if 'abstract' in dataset:
                text_parts.append(dataset['abstract'])
            if 'claims' in dataset:
                text_parts.append(dataset['claims'])
            
            return " ".join(text_parts)
        
        def calculate_term_frequencies(document_text):
            """문서 텍스트에서 단어 추출 및 빈도 계산"""
            document_terms = extract_terms(document_text)
            term_freq = Counter(document_terms)
            return term_freq
        
        def update_postings_and_doc_table(term_freq, doc_id, term_postings, doc_table, file, file_path):
            """포스팅 리스트 업데이트 및 문서 테이블에 정보 저장"""
            # 문서 테이블에 doc_id를 키로 저장
            doc_table[doc_id] = {
                "doc_id": doc_id,
                "filename": file,
                "path": file_path
            }
            
            # postings 리스트에 추가
            for term, freq in term_freq.items():
                term_postings[term].append((doc_id, freq))
        
        def save_doc_table(doc_table):
            """doc_table.json 파일 저장"""
            with open(self.doc_table_file, 'w', encoding='utf8') as f:
                json.dump(doc_table, f, ensure_ascii=False, indent=4)
        
        def save_postings_and_term_dict(term_postings):
            """postings.bin과 term_dict.json 파일 생성 및 저장"""
            term_dict = {}
            offset = 0
            
            with open(self.postings_file, "wb") as pbin:
                for term, plist in term_postings.items():
                    start = offset
                    for doc_id, freq in plist:
                        pbin.write(struct.pack("ii", doc_id, freq))
                        offset += 8
                    
                    term_dict[term] = {
                        "df": len(plist),
                        "start": start,
                        "length": len(plist)
                    }
            
            # term_dict.json 저장
            with open(self.term_dict_file, 'w', encoding='utf8') as f:
                json.dump(term_dict, f, ensure_ascii=False, indent=4)
            
            return term_dict
        
        # 메인 로직 시작
        doc_table = {}
        term_postings = defaultdict(list)
        
        doc_id = 0
        processed_files = 0
        
        # JSON 파일들 처리
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    # JSON 파일 읽기
                    json_data = read_json_file(file_path)
                    
                    # 텍스트 추출
                    document_text = extract_text_from_document(json_data)
                    
                    # 단어 빈도 계산
                    term_freq = calculate_term_frequencies(document_text)
                    
                    # 포스팅 및 문서 테이블 업데이트
                    update_postings_and_doc_table(term_freq, doc_id, term_postings, doc_table, file, file_path)
                    
                    doc_id += 1
                    processed_files += 1
                    
                    # 테스트용: 1000개 파일만 처리
                    if processed_files >= 1000:
                        break
                    
                    if processed_files % 1000 == 0:
                        print(f"처리된 파일: {processed_files:,}개")
            
            #테스트용: 1000개 파일만 처리 (외부 루프 break)
            if processed_files >= 1000:
                break
        
        # 결과 파일들 저장
        save_doc_table(doc_table)
        term_dict = save_postings_and_term_dict(term_postings)
        
        # 완료 메시지 출력
        print(f"인덱싱 완료: 총 {processed_files:,}개 파일 처리")
        print(f"총 {len(term_dict):,}개 unique terms")
        print(f"결과 파일:")
        print(f"  - {self.doc_table_file}")
        print(f"  - {self.term_dict_file}")
        print(f"  - {self.postings_file}")