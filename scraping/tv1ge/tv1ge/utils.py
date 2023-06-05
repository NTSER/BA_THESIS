import re

step = 1000

URLS = [
    f'https://1tv.ge/wp-json/witv/posts?page_id=1131&offset=1&per_page={step}'
]

def get_next_url(url):
    offset = re.search(r'(offset\=)(\d+)', url).group(2)
    next_offset = int(offset) + step
    next_url = re.sub(r'(offset\=)(\d+)', f'\\g<1>{next_offset}', url)

    return next_url