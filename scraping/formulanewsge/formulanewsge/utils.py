import re

URLS = [
    "https://formulanews.ge/index.php?module=news_category&id=4&name=politics&row=0"
]

def get_next_url(url):
    current_row = re.search(r'\d+$', url).group(0)
    next_row = str(int(current_row) + 12)
    next_url = re.sub(r'(\d+$)', next_row, url)

    return next_url

def content_from_quote(quote, quote_description):
    quote = ' '.join(quote).strip()
    quote_description = ' '.join(quote_description).strip()
    content = quote + ' ' + quote_description

    return content


