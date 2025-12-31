import os
import json
import struct
import math
import re
from .tokenizer import extract_terms

class Searcher:
    
    # BM25F 파라미터
    K1 = 1.1
    FIELD_WEIGHTS = {'T': 2.5, 'A': 1.5, 'C': 1.1}
    FIELD_B = {'T': 0.3, 'A': 0.75, 'C': 0.8}
    FIELD_NAMES = {'T': 'TITLE', 'A': 'ABSTRACT', 'C': 'CLAIMS'}
    WINDOW_SIZE = 80
    
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
        
        self.postings_cache = {}
    
    def get_postings(self, term, field):
        """특정 term의 특정 field 포스팅을 Dictionary로 반환 (캐싱 적용)"""
        cache_key = (term, field)
        if cache_key in self.postings_cache:
            return self.postings_cache[cache_key]
        
        if term not in self.term_dict:
            self.postings_cache[cache_key] = {}
            return {}
        
        entry = self.term_dict[term]
        if field not in entry:
            self.postings_cache[cache_key] = {}
            return {}
        
        field_entry = entry[field]
        start_offset = field_entry["start"]
        length = field_entry["length"]
        
        postings = {}
        self.fp.seek(start_offset)
        for _ in range(length):
            data = self.fp.read(8)
            if len(data) != 8:
                raise ValueError(f"Incomplete data read at offset {start_offset}")
            doc_id, freq = struct.unpack("ii", data)
            postings[doc_id] = freq
        
        self.postings_cache[cache_key] = postings
        return postings
    
    def clear_cache(self):
        """포스팅 캐시 초기화"""
        self.postings_cache = {}
    
    def parse_query(self, user_query):
        """쿼리 파싱: Prefix와 Field 추출"""
        verbose = False
        and_mode = False
        phrase_mode = False
        explicit_fields = []
        invalid_prefixes = []
        
        ## valid_prefixes = {'VERBOSE', 'V', 'AND', 'A', 'PHRASE', 'P'}
        
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, user_query)
        
        for match in matches:
            match_upper = match.upper()
            if match_upper in ('VERBOSE', 'V'):
                verbose = True
            elif match_upper in ('AND', 'A'):
                and_mode = True
            elif match_upper in ('PHRASE', 'P'):
                phrase_mode = True
            elif match_upper.startswith('FIELD='):
                field_char = match_upper[6:]
                if field_char in ['T', 'A', 'C']:
                    explicit_fields.append(field_char)
                else:
                    invalid_prefixes.append(f"[{match}]")
            else:
                invalid_prefixes.append(f"[{match}]")
        
        pure_query = re.sub(pattern, '', user_query).strip()
        
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
            'original_query': user_query,
            'invalid_prefixes': invalid_prefixes
        }
    
    def validate_query(self, parsed):
        """쿼리 조합 유효성 검증"""
        if parsed['phrase_mode']:
            if parsed['and_mode']:
                return "오류: [PHRASE]와 [AND]는 동시에 사용할 수 없습니다."
            explicit = parsed['explicit_fields']
            if 'A' in explicit or 'C' in explicit:
                return "오류: [PHRASE]는 Title에서만 검색하므로 [FIELD=A] 또는 [FIELD=C]와 함께 사용할 수 없습니다."
        return None
    
    def calculate_idf(self, df):
        """BM25 IDF 계산"""
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
            
            tf_tilde = 0.0
            
            for field in fields:
                postings = self.get_postings(term, field)
                tf = postings.get(doc_id, 0)
                
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
            doc_sets = []
            for term in query_terms:
                if term not in self.term_dict:
                    return set()
                
                term_docs = set()
                for field in fields:
                    postings = self.get_postings(term, field)
                    term_docs.update(postings.keys())
                
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
            result = set()
            for term in query_terms:
                if term not in self.term_dict:
                    continue
                for field in fields:
                    postings = self.get_postings(term, field)
                    result.update(postings.keys())
            return result
    
    def phrase_search(self, query_text, query_terms):
        """PHRASE 검색: Title에서 exact matching"""
        candidate_docs = self.get_candidate_docs(query_terms, ['T'], and_mode=True)
        
        matched_docs = []
        for doc_id in candidate_docs:
            doc_info = self.doc_table[str(doc_id)]
            title_text = doc_info.get("T_text", "")
            if query_text in title_text:
                matched_docs.append(doc_id)
        
        return matched_docs
    
    # ========== VERBOSE 관련 함수들 ==========
    
    def get_document_fields(self, doc_id):
        """원본 JSON 파일에서 title, abstract, claims 텍스트 가져오기"""
        doc_info = self.doc_table[str(doc_id)]
        file_path = doc_info["path"]
        
        with open(file_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        dataset = data['dataset']
        title = dataset.get('invention_title', '')
        abstract = dataset.get('abstract', '')
        claims = dataset.get('claims', '')
        
        return {'T': title, 'A': abstract, 'C': claims}
    
    def find_term_positions(self, text, terms):
        """텍스트에서 각 term의 모든 위치 찾기"""
        positions = []
        for term in terms:
            start = 0
            while True:
                idx = text.find(term, start)
                if idx == -1:
                    break
                positions.append((idx, idx + len(term), term))
                start = idx + 1
        return sorted(positions, key=lambda x: x[0])
    
    def highlight_text(self, text, terms):
        """텍스트에서 검색어들을 <<...>>로 감싸기"""
        positions = self.find_term_positions(text, terms)
        if not positions:
            return text
        
        result = []
        last_end = 0
        for start, end, term in positions:
            if start >= last_end:
                result.append(text[last_end:start])
                result.append(f"<<{term}>>")
                last_end = end
        result.append(text[last_end:])
        
        return ''.join(result)
    
    def find_best_window_or(self, text, query_terms):
        """OR/일반 검색용: 가장 다양한 검색어가 많이 등장하는 window 찾기"""
        positions = self.find_term_positions(text, query_terms)
        if not positions:
            return None, set()
        
        best_start = 0
        best_unique_terms = set()
        
        for pos_start, pos_end, term in positions:
            window_start = max(0, pos_start - self.WINDOW_SIZE // 2)
            window_end = min(len(text), window_start + self.WINDOW_SIZE)
            
            if window_end - window_start < self.WINDOW_SIZE and window_end == len(text):
                window_start = max(0, window_end - self.WINDOW_SIZE)
            
            unique_terms = set()
            for p_start, p_end, p_term in positions:
                if p_start >= window_start and p_end <= window_end:
                    unique_terms.add(p_term)
            
            if len(unique_terms) > len(best_unique_terms):
                best_unique_terms = unique_terms
                best_start = window_start
        
        return best_start, best_unique_terms
    
    def create_snippet_or(self, text, query_terms):
        """OR/일반 검색용: 다양한 검색어가 많은 window의 snippet 생성"""
        best_start, unique_terms = self.find_best_window_or(text, query_terms)
        if not unique_terms:
            return None, set()
        
        window_end = min(len(text), best_start + self.WINDOW_SIZE)
        snippet = text[best_start:window_end]
        highlighted = self.highlight_text(snippet, query_terms)
        
        return highlighted, unique_terms
    
    def create_snippet_phrase(self, title_text, query_text):
        """PHRASE 검색용: 검색어 전체를 가운데에 배치한 snippet 생성"""
        idx = title_text.find(query_text)
        if idx == -1:
            return None
        
        phrase_center = idx + len(query_text) // 2
        half_window = self.WINDOW_SIZE // 2
        
        window_start = max(0, phrase_center - half_window)
        window_end = min(len(title_text), window_start + self.WINDOW_SIZE)
        
        if window_end - window_start < self.WINDOW_SIZE and window_end == len(title_text):
            window_start = max(0, window_end - self.WINDOW_SIZE)
        
        snippet = title_text[window_start:window_end]
        highlighted = snippet.replace(query_text, f"<<{query_text}>>")
        
        return highlighted
    
    def find_snippets_and(self, doc_fields, query_terms, fields):
        """AND 검색용: 모든 검색어가 나올 때까지 여러 field의 snippet 수집"""
        found_terms = set()
        snippets = []
        
        field_info = []
        for field in fields:
            text = doc_fields[field]
            if not text:
                continue
            _, unique_terms = self.find_best_window_or(text, query_terms)
            if unique_terms:
                field_info.append((field, len(unique_terms), unique_terms))
        
        field_info.sort(key=lambda x: x[1], reverse=True)
        
        for field, _, unique_terms in field_info:
            new_terms = unique_terms - found_terms
            if not new_terms:
                continue
            
            text = doc_fields[field]
            highlighted, _ = self.create_snippet_or(text, query_terms)
            if highlighted:
                snippets.append((field, highlighted))
                found_terms.update(unique_terms)
            
            if found_terms >= set(query_terms):
                break
        
        return snippets
    
    def print_verbose_or(self, doc_id, score, query_terms, fields):
        """OR/일반 검색의 VERBOSE 출력"""
        doc_info = self.doc_table[str(doc_id)]
        filename = doc_info["filename"]
        doc_fields = self.get_document_fields(doc_id)
        
        print(f"파일명: {filename}, 점수: {score:.2f}")
        
        best_field = None
        best_snippet = None
        best_unique_count = 0
        
        for field in ['T', 'A', 'C']:
            if field not in fields:
                continue
            text = doc_fields[field]
            if not text:
                continue
            
            snippet, unique_terms = self.create_snippet_or(text, query_terms)
            if snippet and len(unique_terms) > best_unique_count:
                best_unique_count = len(unique_terms)
                best_field = field
                best_snippet = snippet
        
        if best_field and best_snippet:
            print(f"[{self.FIELD_NAMES[best_field]}] {best_snippet}")
    
    def print_verbose_phrase(self, doc_id, score, query_text):
        """PHRASE 검색의 VERBOSE 출력"""
        doc_info = self.doc_table[str(doc_id)]
        filename = doc_info["filename"]
        title_text = doc_info.get("T_text", "")
        
        print(f"파일명: {filename}, 점수: {score:.2f}")
        
        snippet = self.create_snippet_phrase(title_text, query_text)
        if snippet:
            print(f"[{self.FIELD_NAMES['T']}] {snippet}")
    
    def print_verbose_and(self, doc_id, score, query_terms, fields):
        """AND 검색의 VERBOSE 출력"""
        doc_info = self.doc_table[str(doc_id)]
        filename = doc_info["filename"]
        doc_fields = self.get_document_fields(doc_id)
        
        print(f"파일명: {filename}, 점수: {score:.2f}")
        
        snippets = self.find_snippets_and(doc_fields, query_terms, fields)
        for field, snippet in snippets:
            print(f"[{self.FIELD_NAMES[field]}] {snippet}")
    
    def print_verbose_results(self, ranked_docs, query_terms, query_text, parsed):
        """VERBOSE 모드 결과 출력"""
        print("-" * 50)
        
        top_k = min(5, len(ranked_docs))
        
        for doc_id, score in ranked_docs[:top_k]:
            if parsed['phrase_mode']:
                self.print_verbose_phrase(doc_id, score, query_text)
            elif parsed['and_mode']:
                self.print_verbose_and(doc_id, score, query_terms, parsed['fields'])
            else:
                self.print_verbose_or(doc_id, score, query_terms, parsed['fields'])
            print()
    
    def process_query(self, user_query):
        """쿼리 처리 메인 함수"""
        self.clear_cache()
        
        parsed = self.parse_query(user_query)
        
        if parsed['invalid_prefixes']:
            invalid_str = ', '.join(parsed['invalid_prefixes'])
            print(f"잘못된 형식이 입력되었습니다: {invalid_str}")
            return
        
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
        
        if parsed['phrase_mode']:
            matched_docs = self.phrase_search(parsed['query_text'], query_terms)
            
            doc_scores = {}
            for doc_id in matched_docs:
                score = self.calculate_bm25f_score(query_terms, doc_id, ['T'])
                if score > 0:
                    doc_scores[doc_id] = score
        else:
            candidate_docs = self.get_candidate_docs(query_terms, parsed['fields'], parsed['and_mode'])
            
            doc_scores = {}
            for doc_id in candidate_docs:
                score = self.calculate_bm25f_score(query_terms, doc_id, parsed['fields'])
                if score > 0:
                    doc_scores[doc_id] = score
        
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
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
        
        if parsed['verbose'] and top_k > 0:
            self.print_verbose_results(ranked_docs, query_terms, parsed['query_text'], parsed)