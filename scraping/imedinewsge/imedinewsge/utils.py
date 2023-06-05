import re

URL = [
    'https://imedinews.ge/ge/~/api/getnews/get?skipCount=1&portion=200&categoryId=3&pageId=18'
]

def get_next_url(url):
    current_skip_count = re.search(r'skipCount=(\d+)', url).group(1)
    next_skip_count = int(current_skip_count) + 200
    new_url = re.sub(r'(skipCount=)(\d+)', f'\\g<1>{next_skip_count}', url)

    return new_url