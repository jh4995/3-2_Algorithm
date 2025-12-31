# 3-2_Algorithm

202101976 유종호

수행 가능한 QUERY: BM25F, AND, FIELD, PHRASE, VERBOSE

특이사항:
1.PHRASE와 AND 동시 사용시 에러 메시지 출력
2.PHRASE와 FIELD=A 동시 사용시 에러 메시지 출력
3.PHRASE와 FIELD=C 동시 사용시 에러 메시지 출력
4.AND, FIELD, PHRASE, VERBOSE를 제외한 잘못된 query가 입력된 경우, 잘못된 형식이 입력되었다는 알림 메시지 출력
5.동일한 내용의 파일들이 중복출력되는 문제를 해결하기 위해서, indexer.py에서 seen_filenames라는 set을 정의 후 활용
6.searcher.py의 get_postings에서 기존에는 매번 파일을 읽고 List로 반환하던 방식을, 최초로 파일을 읽는 경우 Dictionary형태로 postings_cache에 저장하여서 캐시된 Dictionary를 반환하도록 수정 -> 파일 I/O횟수와 tf조회 복잡도 감소