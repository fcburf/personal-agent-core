"""구매 카테고리 고정 목록 (구매 에이전트 서경 · 구매·지출 데이터 예정 공용)

- 예정 님 normalize_node: 구매 이력의 category를 이 목록 중 하나로 정규화
- 서경 purchase_advisor: 구매 의도의 category를 이 목록 중 하나로 분류
- Critic PC1(무관한 카테고리 이력 사용)·PD2(같은 카테고리 거절 반복)는 문자열이 정확히 같아야 비교된다.
  → 항목을 추가·변경할 때는 두 담당이 함께 고치고 팀에 공지.
"""

PURCHASE_CATEGORIES = [
    "식품",          # 식사, 간식, 음료
    "건강·위생",      # 파스, 휴족시간, 상비약, 마스크, 위생용품
    "뷰티",          # 화장품, 스킨케어 (올리브영 등)
    "패션",          # 옷, 신발, 가방, 액세서리 (지그재그 등)
    "전자기기",       # 이어폰, 충전기, 보조배터리
    "기념품",         # 여행 기념품, 선물
    "생활용품",       # 우산, 수건, 문구 등
    "교통·입장권",     # 교통 패스, 입장권, 투어
    "기타",
]

DEFAULT_CATEGORY = "기타"

# 건강·안전에 꼭 필요한 '긴급 필요' 품목 예시 (Critic 조건표: PA2 충동구매 신호 미적용, PA5 최소 수량만 확인)
# 휴족시간·파스 같은 피로 회복 용품은 긴급 필요가 아니다.
URGENT_NEED_EXAMPLES = ["해열제", "진통제", "소화제", "지사제", "밴드", "소독약", "상처 연고", "멀미약", "알레르기약"]


def normalize_category(value) -> str:
    """목록에 없는 값은 '기타'로 돌린다 (공백·가운뎃점 표기 차이는 흡수)."""
    if not value:
        return DEFAULT_CATEGORY
    key = str(value).replace(" ", "").replace("/", "·").replace(",", "·")
    for cat in PURCHASE_CATEGORIES:
        if key == cat.replace(" ", ""):
            return cat
    return DEFAULT_CATEGORY
