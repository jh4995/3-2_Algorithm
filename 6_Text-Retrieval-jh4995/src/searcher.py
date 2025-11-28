import os
import json
import struct
import math
import re
from .tokenizer import extract_terms

class Searcher:
    
    # BM25F 파라미터 (8페이지 참조)
    K1 = 1.1
    FIELD_WEIGHTS = {'T': 2.5, 'A': 1.5, 'C': 1.1}
    FIELD_B = {'T': 0.3, 'A': 0.75, 'C': 0.8}
    
    def __init__(self, index_dir, doc_table_file, term_dict_file, postings_file):
        self.index_dir = os.path.abspath(index_dir)
        
        term_dict_path = os.path.join(self.index_dir, term_dict_file)
        with open(term_dict_path, 'r', encoding='utf8') as f:
            self.term_dict = json.load(f)
        
        doc_table_path = os.path.join(self.index_dir, doc_table_file)
        with open(doc_table_path, 'r', encoding='utf8') as f:
            doc_data = json.load(f)
            self.metadata = doc_data["metadata"]
            self.doc_table = doc_data["documents"]
        
        self.N = len(self.doc_table)
        self.avgdl = {
            'T': self.metadata["avgdl_T"],
            'A': self.metadata["avgdl_A"],
            'C': self.metadata["avgdl_C"]
        }
        
        postings_path = os.path.join(self.index_dir, postings_file)
        self.fp = open(postings_path, "rb")
    
    def get_postings(self, term, field):
        """특정 term의 특정 field 포스팅 리스트 반환"""
        if term not in self.term_dict:
            return []
        
        entry = self.term_dict[term]
        if field not in entry:
            return []
        
        field_entry = entry[field]
        start_offset = field_entry["start"]
        length = field_entry["length"]
        
        postings = []
        self.fp.seek(start_offset)
        for _ in range(length):
            data = self.fp.read(8)
            if len(data) != 8:
                raise ValueError(f"Incomplete data read at offset {start_offset}")
            doc_id, freq = struct.unpack("ii", data)
            postings.append((doc_id, freq))
        
        return postings
    
    def parse_query(self, user_query):
        """쿼리 파싱: Prefix와 Field 추출"""
        verbose = False
        and_mode = False
        phrase_mode = False
        explicit_fields = []  # 명시적으로 지정된 필드
        
        # [...]로 둘러싸인 부분 추출
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, user_query)
        
        for match in matches:
            match_upper = match.upper()
            if match_upper == 'VERBOSE':
                verbose = True
            elif match_upper == 'AND':
                and_mode = True
            elif match_upper == 'PHRASE':
                phrase_mode = True
            elif match_upper.startswith('FIELD='):
                field_char = match_upper[6:]
                if field_char in ['T', 'A', 'C']:
                    explicit_fields.append(field_char)
        
        # 브래킷 부분 제거하여 순수 쿼리 추출
        pure_query = re.sub(pattern, '', user_query).strip()
        
        # Field가 지정되지 않으면 전체 필드 사용
        if not explicit_fields:
            fields = ['T', 'A', 'C']
        else:
            fields = explicit_fields
        
        return {
            'verbose': verbose,
            'and_mode': and_mode,
            'phrase_mode': phrase_mode,
            'fields': fields,
            'explicit_fields': explicit_fields,
            'query_text': pure_query,
            'original_query': user_query
        }
    
    def validate_query(self, parsed):
        """쿼리 조합 유효성 검증. 오류 시 오류 메시지 반환, 정상이면 None 반환"""
        if parsed['phrase_mode']:
            # PHRASE + AND 불가
            if parsed['and_mode']:
                return "오류: [PHRASE]와 [AND]는 동시에 사용할 수 없습니다."
            # PHRASE + FIELD=A 또는 FIELD=C 불가
            explicit = parsed['explicit_fields']
            if 'A' in explicit or 'C' in explicit:
                return "오류: [PHRASE]는 Title에서만 검색하므로 [FIELD=A] 또는 [FIELD=C]와 함께 사용할 수 없습니다."
        return None
    
    def calculate_idf(self, df):
        """BM25 IDF 계산 (8페이지 공식)"""
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def calculate_bm25f_score(self, query_terms, doc_id, fields):
        """BM25F 점수 계산"""
        score = 0.0
        doc_info = self.doc_table[str(doc_id)]
        
        for term in query_terms:
            if term not in self.term_dict:
                continue
            
            df = self.term_dict[term]["df"]
            idf = self.calculate_idf(df)
            
            # tf_tilde 계산: 각 필드별 정규화된 TF의 가중합
            tf_tilde = 0.0
            
            for field in fields:
                postings = self.get_postings(term, field)
                tf = 0
                for d_id, freq in postings:
                    if d_id == doc_id:
                        tf = freq
                        break
                
                if tf > 0:
                    dl = doc_info[f"len_{field}"]
                    avgdl = self.avgdl[field]
                    b_f = self.FIELD_B[field]
                    w_f = self.FIELD_WEIGHTS[field]
                    
                    if avgdl > 0:
                        normalized_tf = tf / ((1 - b_f) + b_f * (dl / avgdl))
                    else:
                        normalized_tf = tf
                    
                    tf_tilde += w_f * normalized_tf
            
            if tf_tilde > 0:
                term_score = idf * ((self.K1 + 1) * tf_tilde) / (self.K1 + tf_tilde)
                score += term_score
        
        return score
    
    def get_candidate_docs(self, query_terms, fields, and_mode):
        """검색 대상 문서 ID 집합 반환"""
        if and_mode:
            # AND 모드: 모든 term이 존재하는 문서만
            doc_sets = []
            for term in query_terms:
                if term not in self.term_dict:
                    return set()
                
                term_docs = set()
                for field in fields:
                    postings = self.get_postings(term, field)
                    for doc_id, freq in postings:
                        term_docs.add(doc_id)
                
                if not term_docs:
                    return set()
                
                doc_sets.append(term_docs)
            
            if not doc_sets:
                return set()
            
            result = doc_sets[0]
            for doc_set in doc_sets[1:]:
                result = result & doc_set
            return result
        else:
            # OR 모드: 하나라도 존재하는 문서
            result = set()
            for term in query_terms:
                if term not in self.term_dict:
                    continue
                for field in fields:
                    postings = self.get_postings(term, field)
                    for doc_id, freq in postings:
                        result.add(doc_id)
            return result
    
    def phrase_search(self, query_text, query_terms):
        """PHRASE 검색: Title에서 exact matching (substring matching)"""
        # 1단계: query의 모든 term이 Title에 존재하는 문서 필터링 (속도 최적화)
        candidate_docs = self.get_candidate_docs(query_terms, ['T'], and_mode=True)
        
        # 2단계: 해당 문서들의 Title에서 query_text가 substring으로 존재하는지 확인
        matched_docs = []
        for doc_id in candidate_docs:
            doc_info = self.doc_table[str(doc_id)]
            title_text = doc_info.get("T_text", "")
            if query_text in title_text:
                matched_docs.append(doc_id)
        
        return matched_docs
    
    def process_query(self, user_query):
        """쿼리 처리 메인 함수"""
        # 쿼리 파싱
        parsed = self.parse_query(user_query)
        
        # 쿼리 조합 유효성 검증
        error_msg = self.validate_query(parsed)
        if error_msg:
            print(error_msg)
            return
        
        query_terms = extract_terms(parsed['query_text'])
        
        if not query_terms:
            print("\nRESULT:")
            print(f"검색어 입력: {user_query}")
            print("총 0개 문서 검색")
            return
        
        # PHRASE 모드 처리
        if parsed['phrase_mode']:
            # PHRASE: Title에서 exact matching
            matched_docs = self.phrase_search(parsed['query_text'], query_terms)
            
            # BM25F 점수 계산 (Title 필드만 사용)
            doc_scores = {}
            for doc_id in matched_docs:
                score = self.calculate_bm25f_score(query_terms, doc_id, ['T'])
                if score > 0:
                    doc_scores[doc_id] = score
        else:
            # 일반 검색 (OR 또는 AND)
            candidate_docs = self.get_candidate_docs(query_terms, parsed['fields'], parsed['and_mode'])
            
            doc_scores = {}
            for doc_id in candidate_docs:
                score = self.calculate_bm25f_score(query_terms, doc_id, parsed['fields'])
                if score > 0:
                    doc_scores[doc_id] = score
        
        # 랭킹
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 결과 출력 (10페이지 형식)
        print("\nRESULT:")
        print(f"검색어 입력: {user_query}")
        print(f"총 {len(ranked_docs)}개 문서 검색")
        
        top_k = min(5, len(ranked_docs))
        if top_k > 0:
            print(f"상위 {top_k}개 문서:")
            for doc_id, score in ranked_docs[:top_k]:
                doc_info = self.doc_table[str(doc_id)]
                filename = doc_info["filename"]
                print(f"  {filename}  {score:.2f}")