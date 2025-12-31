import os
import json
import struct
import math
import re
from .tokenizer import extract_terms

class Searcher:
    
    def __init__(self, index_dir, doc_table_file, term_dict_file, 
                 postings_file_T, postings_file_A, postings_file_C):
        self.index_dir = os.path.abspath(index_dir)
        
        term_dict_path = os.path.join(self.index_dir, term_dict_file)
        with open(term_dict_path, 'r', encoding='utf8') as f:
            self.term_dict = json.load(f)
        
        doc_table_path = os.path.join(self.index_dir, doc_table_file)
        with open(doc_table_path, 'r', encoding='utf8') as f:
            doc_table_data = json.load(f)
            self.meta = doc_table_data.pop("_meta")
            self.doc_table = doc_table_data
        
        self.N = self.meta["total_docs"]
        self.avg_len_T = self.meta["avg_len_T"]
        self.avg_len_A = self.meta["avg_len_A"]
        self.avg_len_C = self.meta["avg_len_C"]
        
        postings_path_T = os.path.join(self.index_dir, postings_file_T)
        postings_path_A = os.path.join(self.index_dir, postings_file_A)
        postings_path_C = os.path.join(self.index_dir, postings_file_C)
        
        self.fp_T = open(postings_path_T, "rb")
        self.fp_A = open(postings_path_A, "rb")
        self.fp_C = open(postings_path_C, "rb")
        
        self.k1 = 1.1
        self.w_T = 2.5
        self.w_A = 1.5
        self.w_C = 1.1
        self.b_T = 0.3
        self.b_A = 0.75
        self.b_C = 0.8
    
    def parse_query(self, user_query):
        """
        쿼리 파싱: prefix와 field 추출
        
        Returns:
            dict: {
                'is_and': bool,
                'is_phrase': bool,
                'is_verbose': bool,
                'fields': list,  # ['T', 'A', 'C'] 또는 일부
                'query_text': str
            }
        """
        query_info = {
            'is_and': False,
            'is_phrase': False,
            'is_verbose': False,
            'fields': ['T', 'A', 'C'],
            'query_text': user_query
        }
        
        remaining = user_query
        
        # Prefix 파싱
        while True:
            if remaining.startswith('[AND]'):
                query_info['is_and'] = True
                remaining = remaining[5:].strip()
            elif remaining.startswith('[PHRASE]'):
                query_info['is_phrase'] = True
                remaining = remaining[8:].strip()
            elif remaining.startswith('[VERBOSE]'):
                query_info['is_verbose'] = True
                remaining = remaining[9:].strip()
            elif remaining.startswith('[V]'):
                query_info['is_verbose'] = True
                remaining = remaining[3:].strip()
            elif remaining.startswith('[A]'):
                query_info['is_and'] = True
                remaining = remaining[3:].strip()
            elif remaining.startswith('[P]'):
                query_info['is_phrase'] = True
                remaining = remaining[3:].strip()
            else:
                break
        
        # Field 파싱
        specified_fields = []
        while True:
            match = re.match(r'\[FIELD=([TAC])\]', remaining)
            if match:
                specified_fields.append(match.group(1))
                remaining = remaining[match.end():].strip()
            else:
                break
        
        if specified_fields:
            query_info['fields'] = specified_fields
        
        # PHRASE는 무조건 Title만
        if query_info['is_phrase']:
            query_info['fields'] = ['T']
        
        query_info['query_text'] = remaining
        
        return query_info
    
    def get_postings(self, term, field):
        if term not in self.term_dict:
            return []
        
        if field not in self.term_dict[term]:
            return []
        
        entry = self.term_dict[term][field]
        start_offset = entry["start"]
        length = entry["length"]
        
        fp_map = {"T": self.fp_T, "A": self.fp_A, "C": self.fp_C}
        fp = fp_map[field]
        
        postings = []
        fp.seek(start_offset)
        for _ in range(length):
            data = fp.read(8)
            if len(data) != 8:
                raise ValueError(f"Incomplete data read at offset {start_offset}")
            doc_id, freq = struct.unpack("ii", data)
            postings.append((doc_id, freq))
        
        return postings
    
    def get_tf_for_doc(self, term, doc_id, field):
        postings = self.get_postings(term, field)
        for d_id, freq in postings:
            if d_id == doc_id:
                return freq
        return 0
    
    def calculate_bm25f_score(self, term, doc_id):
        df = self.term_dict[term]["df"]
        idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        
        doc_info = self.doc_table[str(doc_id)]
        len_T = doc_info["len_T"]
        len_A = doc_info["len_A"]
        len_C = doc_info["len_C"]
        
        tf_T = self.get_tf_for_doc(term, doc_id, "T")
        tf_A = self.get_tf_for_doc(term, doc_id, "A")
        tf_C = self.get_tf_for_doc(term, doc_id, "C")
        
        if self.avg_len_T > 0:
            norm_tf_T = tf_T / ((1 - self.b_T) + self.b_T * len_T / self.avg_len_T)
        else:
            norm_tf_T = 0
        
        if self.avg_len_A > 0:
            norm_tf_A = tf_A / ((1 - self.b_A) + self.b_A * len_A / self.avg_len_A)
        else:
            norm_tf_A = 0
        
        if self.avg_len_C > 0:
            norm_tf_C = tf_C / ((1 - self.b_C) + self.b_C * len_C / self.avg_len_C)
        else:
            norm_tf_C = 0
        
        tf_tilde = (self.w_T * norm_tf_T + 
                   self.w_A * norm_tf_A + 
                   self.w_C * norm_tf_C)
        
        score = idf * (self.k1 + 1) * tf_tilde / (self.k1 + tf_tilde)
        
        return score
    
    def search_and_query(self, query_terms, fields):
        """AND 쿼리 처리: 모든 term이 존재하는 문서만"""
        candidate_docs_per_term = []
        
        for term in query_terms:
            if term not in self.term_dict:
                return {}
            
            docs_for_this_term = set()
            for field in fields:
                postings = self.get_postings(term, field)
                for doc_id, _ in postings:
                    docs_for_this_term.add(doc_id)
            
            candidate_docs_per_term.append(docs_for_this_term)
        
        if not candidate_docs_per_term:
            return {}
        
        # 교집합: 모든 term이 있는 문서
        candidate_docs = set.intersection(*candidate_docs_per_term)
        
        # BM25F 스코어 계산
        doc_scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                if term in self.term_dict:
                    score += self.calculate_bm25f_score(term, doc_id)
            doc_scores[doc_id] = score
        
        return doc_scores
    
    def search_phrase_query(self, query_text):
        """PHRASE 쿼리 처리: Title에서 정확히 일치하는 문서만"""
        query_terms = extract_terms(query_text)
        
        if not query_terms:
            return {}
        
        # 첫 번째 term으로 후보 문서 수집
        first_term = query_terms[0]
        if first_term not in self.term_dict:
            return {}
        
        candidate_docs = set()
        postings = self.get_postings(first_term, 'T')
        for doc_id, _ in postings:
            candidate_docs.add(doc_id)
        
        # 각 문서의 Title에서 phrase matching 확인
        matched_docs = {}
        
        for doc_id in candidate_docs:
            doc_info = self.doc_table[str(doc_id)]
            title_text = doc_info["T_text"]
            title_terms = extract_terms(title_text)
            
            # Phrase 매칭 확인
            if self._is_phrase_match(title_terms, query_terms):
                # BM25F 스코어 계산
                score = 0.0
                for term in query_terms:
                    if term in self.term_dict:
                        score += self.calculate_bm25f_score(term, doc_id)
                matched_docs[doc_id] = score
        
        return matched_docs
    
    def _is_phrase_match(self, doc_terms, query_terms):
        """문서의 term 리스트에서 query_terms가 연속으로 나타나는지 확인"""
        if len(query_terms) > len(doc_terms):
            return False
        
        for i in range(len(doc_terms) - len(query_terms) + 1):
            if doc_terms[i:i+len(query_terms)] == query_terms:
                return True
        
        return False
    
    def search_or_query(self, query_terms, fields):
        """OR 쿼리 처리: 기본 검색 (하나라도 있으면)"""
        candidate_docs = set()
        
        for term in query_terms:
            if term in self.term_dict:
                for field in fields:
                    postings = self.get_postings(term, field)
                    for doc_id, _ in postings:
                        candidate_docs.add(doc_id)
        
        doc_scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            for term in query_terms:
                if term in self.term_dict:
                    score += self.calculate_bm25f_score(term, doc_id)
            doc_scores[doc_id] = score
        
        return doc_scores
    
    def highlight_results(self, ranked_docs, query_terms, query_info):
        """상위 5개 문서에 대해 highlighting 수행"""
        top_docs = ranked_docs[:5]
        
        print()
        for doc_id, score in top_docs:
            doc_info = self.doc_table[str(doc_id)]
            filename = os.path.basename(doc_info["relpath"])
            
            print("-" * 50)
            print(f"파일명: {filename}, 점수: {score:.2f}")
            
            if query_info['is_phrase']:
                self._highlight_phrase(doc_info, query_info['query_text'], query_terms)
            elif query_info['is_and']:
                self._highlight_and(doc_info, query_terms, query_info['fields'])
            else:
                self._highlight_or(doc_info, query_terms, query_info['fields'])
        
        print("-" * 50)
    
    def _highlight_phrase(self, doc_info, query_text, query_terms):
        """PHRASE highlighting: Title에서 정확히 일치하는 부분"""
        title_text = doc_info["T_text"]
        title_terms = extract_terms(title_text)
        
        # Phrase 위치 찾기
        phrase_start = -1
        for i in range(len(title_terms) - len(query_terms) + 1):
            if title_terms[i:i+len(query_terms)] == query_terms:
                phrase_start = i
                break
        
        if phrase_start == -1:
            return
        
        # 원본 텍스트에서 phrase 위치 찾기 및 highlighting
        highlighted = self._highlight_in_text(title_text, query_terms, window_size=80)
        
        if highlighted:
            print(f"[TITLE] {highlighted}")
    
    def _highlight_and(self, doc_info, query_terms, fields):
        """AND highlighting: 모든 term이 나타날 때까지 여러 부분 출력"""
        field_priority = []
        if 'T' in fields:
            field_priority.append(('T', doc_info["T_text"], "TITLE"))
        if 'A' in fields:
            field_priority.append(('A', doc_info["A_text"], "ABSTRACT"))
        if 'C' in fields:
            field_priority.append(('C', doc_info["C_text"], "CLAIMS"))
        
        highlighted_terms = set()
        
        for field_code, text, field_name in field_priority:
            if not text:
                continue
            
            remaining_terms = [t for t in query_terms if t not in highlighted_terms]
            if not remaining_terms:
                break
            
            snippet = self._find_snippet_with_terms(text, remaining_terms)
            if snippet:
                highlighted_snippet = self._highlight_in_text(snippet, remaining_terms, window_size=80)
                if highlighted_snippet:
                    print(f"[{field_name}] {highlighted_snippet}")
                    
                    # 이번에 출력된 term들 기록
                    for term in remaining_terms:
                        if term in snippet:
                            highlighted_terms.add(term)
    
    def _highlight_or(self, doc_info, query_terms, fields):
        """OR highlighting: 최대한 많은 서로 다른 term을 포함한 한 부분만 출력"""
        field_priority = []
        if 'T' in fields:
            field_priority.append(('T', doc_info["T_text"], "TITLE"))
        if 'A' in fields:
            field_priority.append(('A', doc_info["A_text"], "ABSTRACT"))
        if 'C' in fields:
            field_priority.append(('C', doc_info["C_text"], "CLAIMS"))
        
        best_snippet = None
        best_field_name = None
        best_term_count = 0
        
        for field_code, text, field_name in field_priority:
            if not text:
                continue
            
            snippet, term_count = self._find_best_snippet(text, query_terms)
            if term_count > best_term_count:
                best_snippet = snippet
                best_field_name = field_name
                best_term_count = term_count
        
        if best_snippet:
            highlighted = self._highlight_in_text(best_snippet, query_terms, window_size=80)
            if highlighted:
                print(f"[{best_field_name}] {highlighted}")
    
    def _find_best_snippet(self, text, query_terms):
        """텍스트에서 가장 많은 서로 다른 query term을 포함한 snippet 찾기"""
        if not text:
            return "", 0
        
        text_terms = extract_terms(text)
        
        best_start = 0
        best_unique_terms = 0
        window_size = 40
        
        for i in range(len(text_terms)):
            end = min(i + window_size, len(text_terms))
            window_terms = text_terms[i:end]
            unique_terms = len(set(window_terms) & set(query_terms))
            
            if unique_terms > best_unique_terms:
                best_unique_terms = unique_terms
                best_start = i
        
        # 원본 텍스트에서 해당 부분 추출
        snippet_terms = text_terms[best_start:min(best_start + window_size, len(text_terms))]
        snippet = self._reconstruct_text(text, snippet_terms)
        
        return snippet, best_unique_terms
    
    def _find_snippet_with_terms(self, text, terms):
        """텍스트에서 주어진 term들을 포함한 부분 찾기"""
        if not text or not terms:
            return ""
        
        text_terms = extract_terms(text)
        
        for i, t in enumerate(text_terms):
            if t in terms:
                start = max(0, i - 20)
                end = min(len(text_terms), i + 20)
                snippet_terms = text_terms[start:end]
                return self._reconstruct_text(text, snippet_terms)
        
        return ""
    
    def _reconstruct_text(self, original_text, terms):
        """term 리스트로부터 원본 텍스트의 해당 부분 재구성"""
        if not terms:
            return ""
        
        # 간단한 방법: 원본 텍스트에서 첫 term이 나타나는 위치부터 80자 추출
        first_term = terms[0]
        idx = original_text.find(first_term)
        if idx == -1:
            return original_text[:80]
        
        start = max(0, idx - 20)
        end = min(len(original_text), idx + 60)
        
        return original_text[start:end]
    
    def _highlight_in_text(self, text, query_terms, window_size=80):
        """텍스트에서 query_terms를 <<term>> 형식으로 highlighting"""
        if not text:
            return ""
        
        # Window 크기 조정
        if len(text) > window_size:
            text_terms = extract_terms(text)
            for i, term in enumerate(text_terms):
                if term in query_terms:
                    start = max(0, i - 10)
                    end = min(len(text_terms), i + 10)
                    snippet_terms = text_terms[start:end]
                    text = self._reconstruct_text(text, snippet_terms)
                    break
        
        # Highlighting
        highlighted = text
        for term in query_terms:
            pattern = re.escape(term)
            highlighted = re.sub(f'({pattern})', r'<<\1>>', highlighted, flags=re.IGNORECASE)
        
        return highlighted
    
    def process_query(self, user_query):
        """쿼리 처리 메인 함수"""
        # 쿼리 파싱
        query_info = self.parse_query(user_query)
        query_text = query_info['query_text']
        
        if not query_text.strip():
            print("\n검색어를 찾을 수 없습니다.")
            return
        
        # 쿼리 실행
        if query_info['is_phrase']:
            doc_scores = self.search_phrase_query(query_text)
            query_terms = extract_terms(query_text)
        elif query_info['is_and']:
            query_terms = extract_terms(query_text)
            doc_scores = self.search_and_query(query_terms, query_info['fields'])
        else:
            query_terms = extract_terms(query_text)
            doc_scores = self.search_or_query(query_terms, query_info['fields'])
        
        # 랭킹
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 기본 결과 출력
        print("\nRESULT:")
        print(f"검색어 입력: {user_query}")
        print(f"총 {len(ranked_docs)}개 문서 검색")
        
        top_n = min(5, len(ranked_docs))
        print(f"상위 {top_n}개 문서:")
        
        if not query_info['is_verbose']:
            for doc_id, score in ranked_docs[:5]:
                doc_info = self.doc_table[str(doc_id)]
                filename = os.path.basename(doc_info["relpath"])
                print(f"  {filename}  {score:.2f}")
        else:
            for doc_id, score in ranked_docs[:5]:
                doc_info = self.doc_table[str(doc_id)]
                filename = os.path.basename(doc_info["relpath"])
                print(f"  {filename}  {score:.2f}")
            
            # Verbose: highlighting
            self.highlight_results(ranked_docs, query_terms, query_info)